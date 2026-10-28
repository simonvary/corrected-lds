"""TDA-side noise: disattenuating the LDS for the attributor's own randomness.

The mirror of the retrain-side correction (``lambda_F / R``): an E-member attributor's
``tau`` is itself a draw, ``g_m(S) = s(S) + u_m(S)``, with per-member noise-to-signal ratio
(the TDA-side "lambda", in the mask-space metric the LDS lives in)

    c = E_m[Var_masks(u_m)] / Var_masks(s)

so the measured LDS is attenuated by ``1/sqrt(1 + c/E)`` and the two corrections compose:

    rho_inf_inf = rho_obs * sqrt((1 + c/E) * (1 + lambda_F/R))

Never reuse :func:`corrected_lds.estimator.seed_noise_variance_stacked` on member
replicates: unlike retrain noise, TDA noise ``u_m(S) = sum_{i in S} delta_i^(m)`` is smooth
in the mask, so at-fixed-mask pooling inflates ``c`` by ``1/(1 - alpha)`` (~2x at
alpha=0.5); :func:`member_noise_var` pools across masks instead (truth recovered to <1%).

Report the battery-level median ``c`` (:func:`battery_c`), never per-query, and the clip
rate of ``V_s``; intervals by delete-one-mask jackknife, never the mask bootstrap (on TRAK
its B=400 percentile interval missed its own point estimate).  Independent replicates (K
models, different seeds) are the preferred input: ``corr(g_a, g_b) = 1/(1 + c)`` then
holds with no extra assumptions, while a nested pair (first-20-of-100 vs the 100)
identifies ``c`` only under 1/E scaling.
"""

import numpy as np

__all__ = [
    "attenuation_factor",
    "battery_c",
    "disattenuate",
    "member_noise_ratio",
    "member_noise_var",
    "member_signal_var",
]

_TINY = 1e-300


def _as_kmq(g):
    g = np.asarray(g, np.float64)
    if g.ndim != 3:
        raise ValueError(f"expected g_members with shape (K, M, Q); got {g.shape}")
    if g.shape[0] < 2:
        raise ValueError(f"need K >= 2 members to separate noise from signal; got K={g.shape[0]}")
    return g


def _var_masks(x):
    """Variance along the mask axis (axis -2), ddof=1 -> drops that axis."""
    x = np.asarray(x, np.float64)
    m = x.shape[-2]
    xc = x - x.mean(axis=-2, keepdims=True)
    return (xc * xc).sum(axis=-2) / (m - 1)


def member_noise_var(g_members):
    """Per-member TDA noise variance ``sigma2`` in mask space, from independent replicates.

    ``g_members``: ``(K, M, Q)`` -- member m's LDS predictions ``g_m = S @ tau_m`` on the same
    M masks.  Returns ``(Q,)``: the mean over member pairs of ``Var_masks(g_a - g_b) / 2``, in
    which the signal cancels -- the retrain side's ``mean_masks((y1 - y2)**2 / 2)``, but with
    the variance taken across masks (see the module docstring)."""
    g = _as_kmq(g_members)
    K = g.shape[0]
    acc = np.zeros(g.shape[2])
    n = 0
    for a in range(K):
        for b in range(a + 1, K):
            acc += _var_masks(g[a] - g[b]) / 2.0
            n += 1
    return acc / n


def member_signal_var(g_members, sigma2=None):
    """Per-query signal variance ``V_s = Var_masks(mean_members g) - sigma2 / K``.

    Subtracting the K-member mean's ``sigma2 / K`` of noise targets the E->inf signal.  Returns
    ``(Q,)``, possibly negative from sampling noise (see :func:`member_noise_ratio`)."""
    g = _as_kmq(g_members)
    sigma2 = member_noise_var(g) if sigma2 is None else np.asarray(sigma2, np.float64)
    return _var_masks(g.mean(axis=0)) - sigma2 / g.shape[0]


def member_noise_ratio(g_members):
    """Per-query ``c = sigma2 / V_s`` plus diagnostics.  Returns ``(c, diag)``.

    ``c`` is floored at 0; ``clipped_frac`` is the fraction of queries with ``V_s <= 0``, and a
    high rate is the finding (noise at or above this design's resolution), not to be hidden."""
    g = _as_kmq(g_members)
    sigma2 = member_noise_var(g)
    varF = member_signal_var(g, sigma2)
    bad = ~(varF > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_raw = sigma2 / np.where(bad, np.nan, varF)
    c = np.where(bad, np.inf, c_raw)
    c = np.clip(np.nan_to_num(c, nan=0.0, posinf=np.inf), 0.0, None)
    diag = {
        "sigma2": sigma2,
        "V_s": varF,
        "K": int(g.shape[0]),
        "M": int(g.shape[1]),
        "c_raw": c_raw,
        "clipped": bad,
        "clipped_frac": float(np.mean(bad)),
    }
    return c, diag


def attenuation_factor(c, E):
    """``sqrt(1 + c/E)`` -- multiply a measured LDS by this to remove the TDA-side attenuation
    of an E-member attributor (or divide to predict what E members would measure)."""
    return np.sqrt(1.0 + np.asarray(c, np.float64) / float(E))


def disattenuate(rho_obs, c, E, lam_F=None, R=None):
    """Disattenuated LDS: ``rho_obs * sqrt(1 + c/E)``, and additionally ``* sqrt(1 + lam_F/R)``
    when the retrain-side ratio is supplied (the two noise sources are independent).  Result
    clipped to [-1, 1].  The ``E``-corrected value describes the E->inf attributor, which costs
    E x the attribution; it is not a free improvement to the attributor you hold, so label it
    as such wherever it is reported."""
    out = np.asarray(rho_obs, np.float64) * attenuation_factor(c, E)
    if lam_F is not None:
        if R is None:
            raise ValueError("disattenuate: pass R alongside lam_F (the retrain-side correction "
                             "is lam_F / R, not lam_F)")
        out = out * np.sqrt(1.0 + np.asarray(lam_F, np.float64) / float(R))
    return np.clip(out, -1.0, 1.0)


def battery_c(c_q):
    """Collapse per-query ``c`` to the single battery-level scalar the correction should use.

    The median, as per-query ``c`` is heavy-tailed (on TRAK its jackknife half-width was
    ~0.017 against ~0.19 for a per-query-shrunk correction).  Clipped queries (``V_s <= 0``)
    carry ``c = +inf`` and count by rank, giving ``1 / median_q(V_s / sigma2)`` (exact for an
    odd number of queries); dropping them would bias ``c`` low.  Returns ``inf`` when more
    than half the queries clip (not estimable); NaN entries (no defined ratio) are ignored."""
    c_q = np.asarray(c_q, np.float64)
    defined = c_q[~np.isnan(c_q)]
    if defined.size == 0:
        return float("nan")
    return float(np.median(defined))
