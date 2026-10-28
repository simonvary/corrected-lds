"""Score one bank (``corrected-lds evaluate``): pool x tauq -> the full corrected-LDS result as
one deterministic, JSON-able dict, versioned by ``schema``.  Diagnostics: ``corr(g, eps-diff)``
for additive noise; Gaussianity of the latent ``(g, F)`` pair via deconvolved marginals and the
Spearman-vs-bridge residual at two noise levels (flat in k is evidence of a Gaussian copula in
the k -> inf limit of ``rho_inf``); and the per-query excess kurtosis of the single-seed readout.
"""

import dataclasses

import numpy as np
from scipy import stats

from corrected_lds.banks.artifacts import predictions, predictions_tda
from corrected_lds.estimator import (
    battery,
    corrected_pearson,
    pearson_to_spearman,
    reconstruct_rho_R,
    seed_noise_variance_centred,
    seed_noise_variance_stacked,
    seed_offset_variance,
)
from corrected_lds.rank_debias import unbiased_spearman

__all__ = ["RESULT_SCHEMA", "evaluate_pool", "evaluate_pool_tda", "pool_tda_summary",
           "soft_threshold_fixed_k"]

RESULT_SCHEMA = 3        # 3: no analytic/parametric debias, pivot CI or Monte-Carlo params


def soft_threshold_fixed_k(tauq, k):
    """TRAK's Eq. 16 sparsification with a fixed k (App. A.2, the paper's ImageNet protocol, no
    CV): per query, ``lambda_q`` is the (k+1)-th largest ``|tau|`` and every entry becomes
    ``sign(tau) * max(|tau| - lambda_q, 0)``, leaving at most k nonzeros.  Evaluation-time only
    (tauq files stay raw); ``k >= n_train`` is the identity.  Returns float64, input untouched."""
    tauq = np.asarray(tauq, dtype=np.float64)
    k = int(k)
    if k < 1:
        raise ValueError(f"soft_threshold_fixed_k needs k >= 1 (got {k}).")
    if k >= tauq.shape[0]:
        return tauq.copy()
    a = np.abs(tauq)
    lam = -np.partition(-a, k, axis=0)[k]           # (Q,) the (k+1)-th largest per column
    return np.sign(tauq) * np.maximum(a - lam[None, :], 0.0)


# -- diagnostics ------------------------------------------------------------ #
def _colwise_corr(a, b):
    """Per-column Pearson correlation of two ``(n, Q)`` arrays; nan on degenerate columns."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    va = (ac * ac).sum(axis=0)
    vb = (bc * bc).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ac * bc).sum(axis=0) / np.sqrt(va * vb)
    return np.where((va > 0) & (vb > 0), r, np.nan)


def _bridge_residual(g, y):
    """Mean over queries of ``direct Spearman - bridge(Pearson)`` for the pair ``(g, y)``: zero
    iff the pair is bivariate normal, so it also catches non-Gaussian marginals under a Gaussian
    copula (ranks unchanged, Pearson shifted) and is stronger than any marginal kurtosis."""
    p = _colwise_corr(g, y)
    s = _colwise_corr(stats.rankdata(g, axis=0), stats.rankdata(y, axis=0))
    return float(np.nanmean(s - pearson_to_spearman(p)))


def _rank_correlations(g, y):
    """Per-query sample Spearman and Kendall tau-a, plus the tied-pair fraction."""
    M, Q = g.shape
    if M < 3:
        raise ValueError("rank-debiased evaluation requires at least three masks")

    spearman = _colwise_corr(stats.rankdata(g, axis=0), stats.rankdata(y, axis=0))
    kendall = np.zeros(Q)
    tied = 0
    for left in range(M - 1):
        pair_sign = np.sign(g[left] - g[left + 1:]) * np.sign(y[left] - y[left + 1:])
        kendall += pair_sign.sum(axis=0)
        tied += int(np.count_nonzero(pair_sign == 0))
    pairs = M * (M - 1) // 2
    return spearman, kendall / pairs, tied / (pairs * Q)


def _spearman_to_pearson(spearman):
    """Inverse Gaussian Pearson--Spearman bridge used by the rank-first composition."""
    return 2.0 * np.sin(np.pi * np.clip(spearman, -1.0, 1.0) / 6.0)


def _latent_shape(y_read, k, sigma2, d):
    """Marginal shape of the latent ``F(S)``, deconvolved from ``y_read = F + epsbar``: with
    independent ``epsbar`` (``Var = sigma2/k``) cumulants add; its excess kurtosis is
    ``2*exkurt(d)/k`` from the paired difference ``d``, whose symmetry leaves its skew
    unidentified (assumed zero).  ``latent_amplification = (1 + lambda/k)^2`` is the honest
    precision of the estimate."""
    V_y = y_read.var(axis=0, ddof=1)
    V_eps = np.asarray(sigma2, float) / k
    V_F = V_y - V_eps
    ok = V_F > 0
    if not ok.any():
        return {}
    ex_eps = 2.0 * stats.kurtosis(d, axis=0, fisher=True) / k
    with np.errstate(invalid="ignore", divide="ignore"):
        ex_F = (stats.kurtosis(y_read, axis=0, fisher=True) * V_y**2
                - ex_eps * V_eps**2) / V_F**2
        sk_F = stats.skew(y_read, axis=0) * (V_y / V_F) ** 1.5     # assumes symmetric eps
        amp = (V_y / V_F) ** 2
    return {
        "kurtosis_F_mean": float(np.nanmean(ex_F[ok])),
        "skew_F_mean": float(np.nanmean(sk_F[ok])),
        "latent_amplification": float(np.nanmean(amp[ok])),
        "latent_frac_identified": float(ok.mean()),
    }


def _jackknife_full(g, margins, clip, level, s2_fn=seed_noise_variance_stacked):
    """Delete-one-mask jackknife SE/CI of the full-pool corrected battery, recomputing sigma2
    (via ``s2_fn``, so a shared-seed pool is re-centred on its remaining masks) and the R-seed
    mean on every deletion; a frozen-sigma2 jackknife undercovers."""
    R, M, _ = margins.shape

    def stat(keep):
        s2 = s2_fn(margins[:, keep])
        rho, _ = corrected_pearson(g[keep], margins[:, keep].mean(axis=0), s2 / R, clip=clip)
        return battery(rho)

    theta = stat(np.ones(M, bool))
    loo = np.empty(M)
    for i in range(M):
        keep = np.ones(M, bool)
        keep[i] = False
        loo[i] = stat(keep)
    se = float(np.sqrt((M - 1) / M * np.sum((loo - loo.mean()) ** 2)))
    z = float(stats.norm.ppf(0.5 * (1.0 + level)))
    return theta, se, (theta - z * se, theta + z * se)


# -- the scorer ------------------------------------------------------------- #
def _seed_noise_fn(pool):
    """The seed-noise estimator the pool's stamped ``seed_mode`` calls for: the corrected
    estimator needs the seed noise that varies across masks, all of it in a ``"nested"`` pool
    (``seed_noise_variance_stacked``).  A ``"shared-per-replica"`` pool's offset ``c(r)``
    cancels in cross-mask correlations, so ``seed_noise_variance_centred`` leaves it out
    where the stacked estimator would over-subtract it."""
    mode = getattr(pool, "seed_mode", "nested")
    if mode == "nested":
        return seed_noise_variance_stacked
    if mode == "shared-per-replica":
        return seed_noise_variance_centred
    raise ValueError(f"[evaluate] no seed-noise estimator for seed_mode={mode!r}")


def evaluate_pool(pool, tau, *, clip=2.0, reconstruct_R=(2, 5, 10), level=0.95,
                  soft_threshold_k=None) -> dict:
    """Score one (pool, tauq) pairing into a JSON-able dict (``RESULT_SCHEMA``).  With
    ``soft_threshold_k``, :func:`soft_threshold_fixed_k` is applied to the tauq first (the
    ImageNet Eq. 16 protocol, k=1000), so every number scores the thresholded attributor."""
    if soft_threshold_k is not None:
        tau = dataclasses.replace(
            tau, tauq=soft_threshold_fixed_k(tau.tauq, soft_threshold_k))
    g = predictions(pool, tau)                      # pairing enforced here
    source = {"path": tau.path, "n_models": tau.n_models, "proj_dim": tau.proj_dim,
              "provenance": tau.provenance}
    return _score(pool, g, source, clip=clip, reconstruct_R=reconstruct_R, level=level,
                  soft_threshold_k=soft_threshold_k)


def evaluate_pool_tda(pool, pool_tda, *, replicas=None, clip=2.0,
                      reconstruct_R=(2, 5, 10), level=0.95) -> dict:
    """Score a (pool, pool_tda) pairing over a replica subset (``replicas=None`` = all); any
    subset is an exact smaller-ensemble estimate.  There is no ``soft_threshold_k``: Eq. 16
    sparsifies tauq entries, and pool_tda slices are already mask-projected."""
    g = predictions_tda(pool, pool_tda, replicas)   # pairing + mask bind enforced here
    reps = (list(range(pool_tda.R_tda)) if replicas is None
            else [int(r) for r in replicas])
    source = {"path": pool_tda.path, "method": pool_tda.method,
              "inner": pool_tda.inner, "e_features": pool_tda.e_features,
              "E": len(reps), "replicas": reps,
              "provenance": pool_tda.provenance}
    return _score(pool, g, source, clip=clip, reconstruct_R=reconstruct_R, level=level,
                  soft_threshold_k=None)


def pool_tda_summary(pool, pool_tda, *, clip=2.0, level=0.95) -> dict:
    """The block ``corrected-lds evaluate`` attaches when a bank ships a ``pool_tda``: the
    evaluation over all replicas (plain-mean methods reproduce the tauq route up to f32 storage;
    for trak's Eq. 15 units the difference is the unit-mean vs pooled gap), plus each replica's
    battery of raw per-query Pearson vs the R-seed mean (member noise, for ``tda_noise``)."""
    sub = evaluate_pool_tda(pool, pool_tda, clip=clip, level=level)
    ybar = pool.margins.mean(axis=0)
    per = [float(battery(_colwise_corr(pool_tda.g[k], ybar)))
           for k in range(pool_tda.R_tda)]
    return {"source": sub["tauq"], "results": sub["results"],
            "diagnostics": sub["diagnostics"], "replica_rho_R": per,
            "replica_rho_R_sd": (float(np.std(per, ddof=1)) if len(per) > 1 else None)}


def _score(pool, g, source, *, clip, reconstruct_R, level, soft_threshold_k) -> dict:
    """Scorer shared by :func:`evaluate_pool` and :func:`evaluate_pool_tda`; ``source`` (the
    attribution source's identity) becomes the result's ``"tauq"`` block."""
    margins = pool.margins
    R, M, Q = pool.R, pool.M, pool.Q

    s2_fn = _seed_noise_fn(pool)
    sigma2 = s2_fn(margins)
    shared = s2_fn is seed_noise_variance_centred
    if shared:
        # Drop each replica's offset c(r) before anything below reads a variance; correlations
        # across masks are invariant to a per-(replica, query) constant.
        offset = seed_offset_variance(margins)
        margins = margins - margins.mean(axis=1, keepdims=True)
    ybar = margins.mean(axis=0)
    spearman_q, kendall_q, rank_tied_frac = _rank_correlations(g, ybar)
    spearman = battery(spearman_q)
    debiased_spearman = battery(unbiased_spearman(spearman_q, kendall_q, M))
    rho_inf_q, diag = corrected_pearson(g, ybar, sigma2 / R, clip=clip)
    rho1_q, _ = corrected_pearson(g, margins[0], np.zeros(Q), clip=clip)
    lam_full = diag["lambda_q"] * R                 # diag's lambda is for the applied sigma2/R
    lam_med = float(np.nanmedian(lam_full))
    recon = {int(r): battery(reconstruct_rho_R(g, ybar, sigma2, k=R, R=int(r), clip=clip)[0])
             for r in reconstruct_R}
    theta, se, jk_ci = _jackknife_full(g, margins, clip, level, s2_fn)

    # One noise factor for all queries: per-query factors would bridge and clip every noisy
    # small-M rank estimate before averaging, which added bias in our experiments.
    rho_R = battery(_colwise_corr(g, ybar))
    noise_factor = theta / rho_R if abs(rho_R) > 1e-12 else 1.0
    debiased_noise_corrected_spearman = float(pearson_to_spearman(
        noise_factor * _spearman_to_pearson(debiased_spearman)))

    d = margins[0] - margins[1]                     # pure seed-noise differences
    diagnostics = {
        "corr_g_epsdiff_mean": float(np.nanmean(_colwise_corr(g, d))),
        "kurtosis_y1_mean": float(np.mean(stats.kurtosis(margins[0], axis=0, fisher=True))),
        "bridge_resid_k1": _bridge_residual(g, margins[0]),
        f"bridge_resid_k{R}": _bridge_residual(g, ybar),
        "rank_tied_pair_frac": rank_tied_frac,
        "rank_noise_factor": noise_factor,
        **_latent_shape(ybar, R, sigma2, d),
    }
    diagnostics["bridge_resid_slope"] = (diagnostics[f"bridge_resid_k{R}"]
                                         - diagnostics["bridge_resid_k1"])
    if shared:          # the discarded offset, so the total stays auditable
        diagnostics["seed_offset_share_mean"] = float(np.mean(offset / (offset + sigma2)))

    out = {
        "schema": RESULT_SCHEMA,
        "pool": {"path": pool.path, "dataset": pool.dataset, "M": M, "R": R, "Q": Q,
                 "n": pool.n, "alpha": pool.alpha, "train_hash": pool.train_hash,
                 "query_hash": pool.query_hash, "code_version": pool.code_version,
                 "recipe": pool.recipe,
                 # only when non-default, so every nested result stays byte-identical
                 **({"seed_mode": pool.seed_mode} if shared else {})},
        "tauq": source,
        "params": {"clip": clip, "level": level,
                   "soft_threshold_k": (int(soft_threshold_k)
                                        if soft_threshold_k is not None else None)},
        "results": {
            "naive_rho1": battery(rho1_q),
            "spearman": spearman,
            "debiased_spearman": debiased_spearman,
            "debiased_noise_corrected_spearman": debiased_noise_corrected_spearman,
            "corrected_rho_inf": theta,
            "corrected_se_jackknife": se,
            "corrected_ci_jackknife": [jk_ci[0], jk_ci[1]],
            "bridged_rho_inf": battery(pearson_to_spearman(rho_inf_q)),
            "bridged_rho1": battery(pearson_to_spearman(rho1_q)),
            "reconstructed_rho_R": recon,
            "sigma2_mean": float(sigma2.mean()),
            "lambda_median": lam_med,
            "floored_frac": diag["floored_frac"],
        },
        "diagnostics": diagnostics,
    }
    return out
