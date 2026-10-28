"""Paired-seed bias-corrected LDS estimator (pure numpy, no torch).

The LDS correlates a TDA prediction ``g_q(S) = sum_{i in S} tau_q[i]`` with a seed-noisy
retrain readout ``y_q(S, xi) = F_q(S) + eps_q(S, xi)`` across masks S.  At R seeds, plain
Pearson tends to ``rho_R = rho_inf * sqrt(1/(1 + lambda/R))``, below the noise-free
``rho_inf`` for any number of masks.  If ``Cov(g, eps) = 0`` (the only assumption), only
``Var(y)`` is inflated, so subtract the seed-noise variance from paired retrains:

    sigma2_eps = mean over paired masks of (y1 - y2)**2 / 2        (exactly unbiased)
    rho_inf_hat = Cov(g, y) / ( s_g * sqrt( s_y**2 - sigma2_eps ) )

``g, y`` are ``(M, Q)`` (masks x queries); statistics are ``(Q,)``.  The pinned parity
numbers rely on ``ddof=1`` for every variance (across seeds too) and column-centred Pearson.
"""

import numpy as np

__all__ = [
    "battery",
    "corrected_pearson",
    "pearson_to_spearman",
    "reconstruct_rho_R",
    "seed_noise_variance_centred",
    "seed_noise_variance_stacked",
    "seed_offset_variance",
]

_TINY = 1e-12


# -- Pearson -> Spearman bridge (single source of truth; scalar and array) --- #
def pearson_to_spearman(rho_p):
    """Pearson -> Spearman for a bivariate normal pair: ``(6/pi) * arcsin(rho_P / 2)``.

    Vectorised; clips ``rho_p`` to [-1, 1].  Needs joint normality, not a mere Gaussian copula
    (other marginals move Pearson, not ranks).  Bridge last, from the corrected Pearson of the
    latent ``(g, F)``, as attenuation is a Pearson identity; evaluation reports the residual.
    """
    return 6.0 / np.pi * np.arcsin(np.clip(np.asarray(rho_p, float), -1.0, 1.0) / 2.0)


# -- column-centred (ddof=1) building blocks -------------------------------- #
def _cov_cols(g, y):
    """Per-column covariance Cov(g[:,q], y[:,q]) with ddof=1 (column-centred products)."""
    g = np.asarray(g, float)
    y = np.asarray(y, float)
    M = g.shape[0]
    gc = g - g.mean(axis=0)
    yc = y - y.mean(axis=0)
    return (gc * yc).sum(axis=0) / (M - 1)


def _var_cols(x):
    """Per-column variance Var(x[:,q]) with ddof=1."""
    x = np.asarray(x, float)
    M = x.shape[0]
    xc = x - x.mean(axis=0)
    return (xc * xc).sum(axis=0) / (M - 1)


# -- seed-noise variance from paired retrains ------------------------------- #
def seed_noise_variance_stacked(y_seeds):
    """Per-query seed-noise variance from ``S`` seeds per mask: ``(S, m, Q)`` -> ``(Q,)``.

    The mean over masks of the across-seed variance; for S=2, ``mean_masks((y1 - y2)**2 / 2)``,
    unbiased for ``sigma2_eps`` without normality, as F(S) cancels within a mask.  ``ddof=1``
    is fixed, not a parameter: the golden parity numbers were computed with it.
    """
    y = np.asarray(y_seeds, float)
    return np.var(y, axis=0, ddof=1).mean(axis=0)


def seed_noise_variance_centred(y_seeds):
    """Per-query seed-noise variance for a shared-per-replica pool: ``(S, m, Q)`` -> ``(Q,)``.

    One seed shared across a replica's masks (bergson's protocol) adds a per-replica offset:
    ``y[r, j, q] = mu(S_j) + c(r) + eps(r, j)``.  ``c(r)`` cancels in correlations across masks,
    so it does not attenuate, but :func:`seed_noise_variance_stacked` counts it and overstates
    the floor.  Removing each replica's mask-mean leaves ``Var(eps)``; ``m / (m - 1)`` restores
    the degree of freedom that costs.  Nested pools have no offset: use the stacked estimator.
    """
    y = np.asarray(y_seeds, float)
    if y.ndim != 3:
        raise ValueError(f"seed_noise_variance_centred needs (S, m, Q); got {y.shape}.")
    m = y.shape[1]
    if m < 2:
        raise ValueError("seed_noise_variance_centred needs m >= 2 masks to identify c(r).")
    yc = y - y.mean(axis=1, keepdims=True)
    return np.var(yc, axis=0, ddof=1).mean(axis=0) * m / (m - 1)


def seed_offset_variance(y_seeds):
    """Per-replica offset variance ``Var(c)``: the non-attenuating part, reported for audit."""
    y = np.asarray(y_seeds, float)
    return np.var(y.mean(axis=1), axis=0, ddof=1)


# -- the corrected (disattenuated) estimator -------------------------------- #
def _corrected_core(g, y, correction, clip):
    """Shared kernel: subtract ``correction`` (Q,) from Var(y), floor the denominator^2 at
    ``s_y^2 / clip^2`` (never a hard gate) and clip rho to [-1, 1]; returns ``(rho_q, diag)``."""
    g = np.asarray(g, float)
    y = np.asarray(y, float)
    correction = np.asarray(correction, float)
    cov, vg, vy = _cov_cols(g, y), _var_cols(g), _var_cols(y)
    sg = np.sqrt(vg)

    den2_raw = vy - correction
    floor = vy / (clip * clip)
    floored = den2_raw < floor
    den2 = np.maximum(den2_raw, floor)

    denom = sg * np.sqrt(den2)
    varF = np.where(np.abs(den2_raw) < _TINY, np.nan, den2_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.clip(np.where(denom > 0, cov / denom, 0.0), -1.0, 1.0)
        # inside errstate too: a constant readout column (vy == 0) gives nan, not a warning
        r1_q = np.where(vy > 0, den2_raw / vy, np.nan)   # signal fraction of this readout
        lambda_q = correction / varF                     # noise/signal implied by the applied correction
    diag = {
        "cov": cov,
        "s_g": sg,
        "s_y2": vy,
        "correction": correction,
        "den2_raw": den2_raw,        # s_y^2 - correction, before the clip floor
        "floored": floored,          # (Q,) bool: clip floor engaged
        "floored_frac": float(np.mean(floored)),
        "r1_q": r1_q,                # = r_1 when single-seed, full correction
        "lambda_q": lambda_q,        # = canonical lambda for single-seed, full correction
    }
    return rho, diag


def corrected_pearson(g, y, sigma2_q, clip=2.0):
    """Disattenuated per-query LDS: ``Cov(g, y) / (s_g * sqrt(s_y^2 - sigma2_q))``.

    ``g, y``: ``(M, Q)``; ``sigma2_q``: ``(Q,)`` seed-noise variance present in ``y`` (full
    ``sigma2_eps`` single-seed, ``sigma2_eps / k`` for a k-seed mean).  The denominator^2 is
    floored at ``s_y^2 / clip^2`` and rho clipped to [-1, 1].  Returns ``(rho_q, diag)``.
    """
    return _corrected_core(g, y, np.asarray(sigma2_q, float), clip)


def reconstruct_rho_R(g, ybar_k, sigma2_q, k, R, clip=2.0):
    """Reconstruct the finite-seed convention ``rho_R`` from a k-seed-mean readout.

    ``Var(ybar_k)`` exceeds ``Var(ybar_R)`` by ``(1/k - 1/R) * sigma2_eps``, so subtract that;
    ``sigma2_q`` is the full (R=1) variance and ``R -> inf`` gives :func:`corrected_pearson`.
    At ``k = R = 1`` ``sigma2_q`` is never read: plain single-seed Pearson to machine precision.
    """
    coef = (1.0 / k) - (1.0 / R)
    if coef == 0.0:
        correction = np.zeros_like(_var_cols(ybar_k))
    else:
        correction = coef * np.asarray(sigma2_q, float)
    rho, diag = _corrected_core(g, ybar_k, correction, clip)
    diag.update({"k": k, "R": R, "coef": coef})
    return rho, diag


# -- battery averaging over the query set ----------------------------------- #
def battery(rho_q):
    """Battery LDS: the plain mean of per-query rho.  Returns a float."""
    return float(np.mean(np.asarray(rho_q, float)))
