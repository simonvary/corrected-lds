"""Direct conversion-discrepancy checks for the two Spearman estimators the paper uses.

Both carry a Pearson-space noise correction to Spearman through the Gaussian bridge h, and
``_proxy_sweep``'s per-query bridge residual measures neither, so each is scored with its
figure's own function against a target that involves no conversion.

Combined (Figure 1(a), ensembles; ``_rmse_corrected._estimate``): ``h(theta/rho_k *
h^-1(S_deb))``, S_deb the U-statistic mask-debiased Spearman of the k-seed readout.  The
full-bank Pearson factor in place of the k-seed one leaves the conversion's share of D; the
``MATCHED`` steps are the size of the deployed, unobservable k = R -> infinity one.  Finite
K-seed targets are observable, so with no extrapolation (and exactly 0 at K = k)

    D(k -> K) = mean over disjoint k-seed groups of h(theta_K / rho_k * h^-1(S_deb,k))
                - mean over disjoint K-seed groups of S_deb,K.

Per-query (Figure 2(b,e), E=1; ``_shared._estimate_b``): D = mean over random 2-seed draws
at the full mask set minus rho^s_5 (sample Spearman of 5-seed means, 40 random subsets),
split into the conversion alone (h(5-seed Pearson) - 5-seed Spearman; exactly Gaussian
pairs at this M give its finite-mask part, the rest is non-Gaussian) and the reconstruction.

SEs: block jackknife over masks of one k-seed application (combined), SD over 2-seed draws
(per-query).  Each D: delete-a-block jackknife over masks (20 blocks) recomputing estimates
and targets, from per-block-pair Kendall sums (no O(M^2) sweep per deletion).  ~20 min, CPU.
"""

import os
import sys
from math import comb

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _assumptions as fa
import _rmse_corrected as f1a
import _shared
from _harness import bank_stamp, run
from _heldout import seed_noise

from corrected_lds import (
    battery,
    corrected_pearson,
    load_pool,
    load_pool_tda,
    pearson_to_spearman,
    predictions_tda,
    reconstruct_rho_R,
)
from corrected_lds.evaluate import _seed_noise_fn
from corrected_lds.rank_debias import unbiased_spearman

SEED = 0
CLIP = f1a.CLIP
NB = 20                          # mask blocks for every jackknife
G_MAX = 10                       # disjoint seed groups averaged per seed level
FOOTINGS = {"E1": [0], "full": None}
K_IN = {"cifar10": [2, 3, 5, 10, 20, 40], "qnli": [2, 3, 5, 10, 20],
        "wikitext2_gpt2_bae": [2, 3, 5, 10, 20]}
K_OUT = {"cifar10": [5, 10, 15, 20, 30, 40], "qnli": [4, 5, 6, 8, 10, 15, 20],
         "wikitext2_gpt2_bae": [4, 5, 6, 8, 10, 15, 20]}
# same-size step as R -> infinity
MATCHED = {"cifar10": (20, 40), "qnli": (10, 20), "wikitext2_gpt2_bae": (10, 20)}
DEPLOYED = K_IN                   # SE of one application at every input seed count
N_DRAWS = 200                    # random 2-seed draws for the per-query estimator
INF = "inf"


def _h(x):
    return pearson_to_spearman(x)


def _h_inv(s):
    return f1a._spearman_to_pearson(s)


# -- the combined estimator, with a block jackknife ------------------------- #
def _concordance_blocks(g, y, nb):
    """``C[a, b, q]`` = sum over mask pairs i < j in blocks a <= b of
    sign(g_i - g_j) sign(y_i - y_j).  Masks are ordered so blocks are contiguous."""
    m, q = g.shape
    bs = m // nb
    c = np.zeros((nb, nb, q))
    for i in range(m - 1):
        s = np.sign(g[i] - g[i + 1:]) * np.sign(y[i] - y[i + 1:])
        b0 = (i + 1) // bs
        bounds = [0] + [b * bs - (i + 1) for b in range(b0 + 1, nb)]
        c[i // bs, b0:] += np.add.reduceat(s, bounds, axis=0)
    return c


def _kendall_all(c, m, nb):
    """Kendall tau-a on the full mask set and with each block deleted: ``(1 + nb, Q)``."""
    bs = m // nb
    total = c.sum(axis=(0, 1))
    out = [total / comb(m, 2)]
    for d in range(nb):
        removed = c[d].sum(axis=0) + c[:, d].sum(axis=0) - c[d, d]
        out.append((total - removed) / comb(m - bs, 2))
    return np.array(out)


def _keeps(m, nb):
    bs = m // nb
    keep = [np.ones(m, bool)]
    for d in range(nb):
        k = np.ones(m, bool)
        k[d * bs:(d + 1) * bs] = False
        keep.append(k)
    return keep


def _level(g, rank_g, seeds_grp, keeps, targets, is_input, s2_fn):
    """Everything one seed group contributes, on the full mask set and each deletion:
    the debiased Spearman, and (input groups) rho_k and theta_K for every target."""
    m = g.shape[0]
    ybar = seeds_grp.mean(axis=0)
    tau = _kendall_all(_concordance_blocks(g, ybar, NB), m, NB)
    out = {"S_deb": [], "rho": [], "theta": {K: [] for K in targets}}
    for r, keep in enumerate(keeps):
        gk, yk = g[keep], ybar[keep]
        sp = f1a._colwise_corr(rank_g[r], stats.rankdata(yk, axis=0))
        out["S_deb"].append(float(battery(unbiased_spearman(sp, tau[r], int(keep.sum())))))
        if not is_input:
            continue
        k = seeds_grp.shape[0]
        s2 = s2_fn(seeds_grp[:, keep])
        out["rho"].append(float(battery(f1a._colwise_corr(gk, yk))))
        for K in targets:
            if K == INF:
                th = corrected_pearson(gk, yk, s2 / k, clip=CLIP)[0]
            else:
                th = reconstruct_rho_R(gk, yk, s2, k=k, R=K, clip=CLIP)[0]
            out["theta"][K].append(float(battery(th)))
    return {k: (np.array(v) if not isinstance(v, dict) else {kk: np.array(vv)
                                                              for kk, vv in v.items()})
            for k, v in out.items()}


def _combine(lv, K):
    """The composition for target K, on the full set and every deletion: ``(1 + nb,)``."""
    factor = lv["theta"][K] / lv["rho"]
    return _h(factor * _h_inv(lv["S_deb"]))


def _jk_se(reps):
    """Delete-a-block jackknife SE from the ``nb`` deletion replicates."""
    reps = np.asarray(reps, float)
    return float(np.sqrt((len(reps) - 1) / len(reps) * np.sum((reps - reps.mean()) ** 2)))


def _oracle_factor(g, margins, s2_fn):
    """The full-bank Pearson factor for k -> K seeds, from the full-bank rho_inf and lambda:
    ``battery(rho_inf/sqrt(1+lam/K)) / battery(rho_inf/sqrt(1+lam/k))``."""
    r_ = margins.shape[0]
    s2 = s2_fn(margins)
    rho_inf, diag = corrected_pearson(g, margins.mean(axis=0), s2 / r_, clip=CLIP)
    lam = np.asarray(diag["lambda_q"], float) * r_
    att = lambda n: float(battery(rho_inf / np.sqrt(1.0 + lam / n)))
    return lambda k, K: att(K) / att(k)


def _combined(key, g, margins, keeps, s2_fn):
    r_ = margins.shape[0]
    oracle = _oracle_factor(g, margins, s2_fn)
    rank_g = [stats.rankdata(g[k], axis=0) for k in keeps]
    levels = sorted(set(K_IN[key]) | set(K_OUT[key]))
    groups = {lv: [margins[j * lv:(j + 1) * lv] for j in range(min(r_ // lv, G_MAX))]
              for lv in levels}
    res = {}
    for lv in levels:
        is_in = lv in K_IN[key]
        targets = [K for K in K_OUT[key] if K > lv] + [INF] if is_in else []
        res[lv] = [_level(g, rank_g, grp, keeps, targets, is_in, s2_fn)
                   for grp in groups[lv]]
        print(f"[bridge_estimators] {key}: seed level {lv} ({len(groups[lv])} groups)",
              flush=True)
    pairs = []
    for k in K_IN[key]:
        for K in K_OUT[key]:
            if K <= k:
                continue
            est = np.mean([_combine(lv, K) for lv in res[k]], axis=0)
            tgt = np.mean([lv["S_deb"] for lv in res[K]], axis=0)
            d = est - tgt
            fac = float(np.mean([lv["theta"][K][0] / lv["rho"][0] for lv in res[k]]))
            conv = np.mean([_h(oracle(k, K) * _h_inv(lv["S_deb"])) for lv in res[k]],
                           axis=0) - tgt
            pairs.append({"k": k, "K": K, "D": float(d[0]), "D_se": _jk_se(d[1:]),
                          "D_conversion": float(conv[0]), "D_conversion_se": _jk_se(conv[1:]),
                          "estimate": float(est[0]), "target": float(tgt[0]),
                          "factor": fac, "factor_full_bank": oracle(k, K)})
    deployed = []
    for k in DEPLOYED[key]:
        one = _combine(res[k][0], INF)                  # one application, first group
        avg = np.mean([_combine(lv, INF) for lv in res[k]], axis=0)
        fac = float(np.mean([lv["theta"][INF][0] / lv["rho"][0] for lv in res[k]]))
        deployed.append({"k": k, "estimate": float(avg[0]), "se_one": _jk_se(one[1:]),
                         "factor": fac, "n_groups": len(res[k])})
    full = _combine(res[r_][0], INF)[0]
    ref = f1a._estimate("corrected", g, margins)
    if abs(full - ref) > 1e-10:
        raise SystemExit(f"[bridge_estimators] {key}: composition {full} != "
                         f"_rmse_corrected._estimate {ref}")
    return {"pairs": pairs, "deployed": deployed, "full_bank": float(full),
            "matched": MATCHED[key]}


# -- the per-query estimator of figure 2(b,e) ------------------------------- #
def _gaussian_conversion(g, margins, subsets5, rng, n_rep=20):
    """The conversion discrepancy ``battery(h(r_P)) - battery(r_S)`` of an exactly Gaussian
    pair at this M and the per-query 5-seed correlations: the finite-mask Spearman bias."""
    m = g.shape[0]
    r = np.mean([f1a._colwise_corr(g, margins[s].mean(axis=0)) for s in subsets5], axis=0)
    gaps = []
    for _ in range(n_rep):
        x = rng.standard_normal((m, r.size))
        y = r * x + np.sqrt(1.0 - r * r) * rng.standard_normal((m, r.size))
        sp = f1a._colwise_corr(stats.rankdata(x, axis=0), stats.rankdata(y, axis=0))
        gaps.append(float(battery(_h(f1a._colwise_corr(x, y)))) - float(battery(sp)))
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1) / np.sqrt(n_rep))


def _per_query(g, margins, keeps, rng, s2_fn):
    r_ = margins.shape[0]
    subsets5 = [rng.choice(r_, _shared.TARGET_R, replace=False) for _ in range(_shared.N_TRUTH)]
    draws2 = [rng.choice(r_, 2, replace=False) for _ in range(N_DRAWS)]
    out = []
    for keep in keeps:
        gk, mk = g[keep], margins[:, keep]
        rank_g = stats.rankdata(gk, axis=0)
        truth, conv = [], []
        for s in subsets5:
            yb = mk[s].mean(axis=0)
            sp = f1a._colwise_corr(rank_g, stats.rankdata(yb, axis=0))
            truth.append(float(battery(sp)))
            conv.append(float(battery(_h(f1a._colwise_corr(gk, yb)))) - truth[-1])
        est = np.array([_shared._estimate_b("recon", 2, gk, mk[s], s2_fn) for s in draws2])
        out.append({"truth": float(np.mean(truth)), "conv": float(np.mean(conv)),
                    "est_mean": float(est.mean()), "est_sd": float(est.std(ddof=1))})
    full = out[0]
    gauss, gauss_se = _gaussian_conversion(g, margins, subsets5, np.random.default_rng([SEED, 7]))
    d = np.array([o["est_mean"] - o["truth"] for o in out])
    cv = np.array([o["conv"] for o in out])
    return {"truth": full["truth"], "estimate_mean": full["est_mean"],
            "D": float(d[0]), "D_se": _jk_se(d[1:]),
            "D_mc_se": full["est_sd"] / np.sqrt(N_DRAWS),
            "conversion": float(cv[0]), "conversion_se": _jk_se(cv[1:]),
            "reconstruction": float(d[0] - cv[0]), "se_estimator": full["est_sd"],
            "conversion_gaussian": gauss, "conversion_gaussian_se": gauss_se,
            "conversion_non_gaussian": float(cv[0] - gauss),
            "n_draws": N_DRAWS, "n_truth_subsets": _shared.N_TRUTH}


# -- compute / plot --------------------------------------------------------- #
def compute(banks_root):
    rng = np.random.default_rng(SEED)
    banks, out = {}, {}
    for key, sweep_tda in fa.SWEEP_TDA.items():
        pool = load_pool(os.path.join(banks_root, key, "pool.npz"))
        s2_fn = _seed_noise_fn(pool)               # centred for a shared-per-replica pool
        banks[key] = bank_stamp(pool)
        perm = rng.permutation(pool.M)              # blocks = contiguous runs of this order
        margins = pool.margins[:, perm]
        keeps = _keeps(pool.M, NB)
        tda = load_pool_tda(os.path.join(banks_root, key, f"{sweep_tda}.npz"))
        comb_ = {}
        for f, rep in FOOTINGS.items():
            g = predictions_tda(pool, tda, rep)[perm]
            with seed_noise(pool):             # and in _rmse_corrected's check estimator
                comb_[f] = {"E": 1 if rep else int(tda.R_tda),
                            **_combined(key, g, margins, keeps, s2_fn)}
            print(f"[bridge_estimators] {key} {f}: full-bank {comb_[f]['full_bank']:.5f}",
                  flush=True)
        pq = {}
        for tda_name, label in _shared.B_METHODS[key]:
            g = predictions_tda(pool, load_pool_tda(os.path.join(banks_root, key,
                                                                 f"{tda_name}.npz")),
                                _shared.E1_REPLICAS)[perm]
            pq[tda_name] = {"label": label, **_per_query(g, margins, keeps, rng, s2_fn)}
            p = pq[tda_name]
            print(f"[bridge_estimators] {key} {label}: D {p['D']:+.5f} +- {p['D_se']:.5f} "
                  f"(conversion {p['conversion']:+.5f}); estimator SE {p['se_estimator']:.5f}",
                  flush=True)
        out[key] = {"label": _shared.BANK_LABEL[key], "color": _shared.BANK_COLOR[key],
                    "attributor": sweep_tda, "R": pool.R, "M": pool.M,
                    "combined": comb_, "per_query": pq}
        del pool, margins
    return {"banks": banks, "panels": out,
            "params": {"seed": SEED, "clip": CLIP, "nb": NB, "g_max": G_MAX, "k_in": K_IN,
                       "k_out": K_OUT, "matched": MATCHED, "deployed": DEPLOYED,
                       "n_draws": N_DRAWS, "per_query_methods": _shared.B_METHODS,
                       "combined_estimator": "_rmse_corrected._estimate('corrected')",
                       "per_query_estimator": "_shared._estimate_b('recon', 2) -> rho_5",
                       "se": "combined: block jackknife of one k-seed application; "
                             "per-query: SD over 2-seed draws at the full mask set"}}


if __name__ == "__main__":
    sys.exit(run("bridge_estimators", compute))
