"""The TDA-side noise estimator (corrected_lds.tda_noise) against known truth, on synthetic
mask-space data with a controlled `c`: it recovers `c`, the ``1/E`` attenuation law holds, the
gold-side variance estimator is biased here by ``1/(1-alpha)`` (the ~2x pooling trap the
module exists to prevent), and degradation below the design's resolution is reported.
"""

import numpy as np
import pytest

from corrected_lds.estimator import seed_noise_variance_stacked
from corrected_lds.tda_noise import (
    attenuation_factor,
    battery_c,
    disattenuate,
    member_noise_ratio,
    member_noise_var,
    member_signal_var,
)


def _abstract(K=6, M=400, Q=200, c_true=3.0, seed=0):
    """g_m = s + u_m directly in mask space, with an exactly known noise/signal ratio."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(M, Q))                                   # unit signal variance
    u = rng.normal(scale=np.sqrt(c_true), size=(K, M, Q))         # per-member noise
    return s[None] + u, s


def _mask_structured(K=6, M=400, n=800, Q=60, alpha=0.5, noise_scale=0.7, seed=0):
    """The realistic generator: masks x attribution vectors, so the mask-space noise of a
    member is the smooth function ``sum_{i in S} delta_i`` rather than iid-per-mask draws."""
    rng = np.random.default_rng(seed)
    S = (rng.random((M, n)) < alpha).astype(np.float64)
    tau = rng.normal(size=(n, Q))
    delta = rng.normal(scale=noise_scale, size=(K, n, Q))
    g = np.stack([S @ (tau + delta[m]) for m in range(K)])        # (K, M, Q)
    return g, S, tau, delta


def _vm(x):
    """Across-mask ddof=1 variance, per query."""
    return ((x - x.mean(0)) ** 2).sum(0) / (x.shape[0] - 1)


# -- recovery of a known c -------------------------------------------------- #
@pytest.mark.parametrize("c_true", [0.25, 1.0, 4.0, 20.0])
def test_recovers_known_c_in_abstract_mask_space(c_true):
    g, _ = _abstract(c_true=c_true, seed=1)
    c, diag = member_noise_ratio(g)
    got = battery_c(c)
    assert diag["clipped_frac"] == 0.0
    assert got == pytest.approx(c_true, rel=0.15), f"c_true={c_true} -> estimated {got:.3f}"


def test_recovers_c_through_the_mask_times_attribution_generator():
    """The estimator must work on the structured (smooth-in-the-mask) noise it will meet."""
    g, S, tau, delta = _mask_structured(seed=2)
    c, _ = member_noise_ratio(g)
    # ground truth from the generator: both variances across masks, as the formula requires
    u_true = np.stack([S @ delta[m] for m in range(delta.shape[0])])
    s_true = S @ tau
    c_ref = np.median(np.mean([_vm(u_true[m]) for m in range(u_true.shape[0])], axis=0)
                      / _vm(s_true))
    assert battery_c(c) == pytest.approx(c_ref, rel=0.15)


def test_signal_var_removes_the_residual_noise_of_the_member_mean():
    """Var(mean of K members) overshoots the signal by sigma2/K, which must be subtracted or c
    is biased low (by a bias that shrinks with K, hiding itself)."""
    g, s = _abstract(K=4, c_true=6.0, seed=3)
    naive = np.median(_vm(g.mean(0)))                      # not corrected
    corrected = np.median(member_signal_var(g))
    truth = np.median(_vm(s))
    assert corrected == pytest.approx(truth, rel=0.15)
    assert naive > truth * 1.2, "the uncorrected member-mean variance should be inflated"


# -- the 1/E law ------------------------------------------------------------ #
def test_one_over_E_attenuation_law():
    """Averaging E members must attenuate the measured LDS by exactly 1/sqrt(1+c/E)."""
    c_true, K = 8.0, 16
    g, s = _abstract(K=K, M=4000, Q=40, c_true=c_true, seed=5)
    F = s + np.random.default_rng(6).normal(scale=0.3, size=s.shape)     # a noisy "margins"

    def cc(a, b):
        return np.median([np.corrcoef(a[:, q], b[:, q])[0, 1] for q in range(a.shape[1])])

    rho_inf = cc(s, F)
    for E in (1, 2, 4, 8, 16):
        rho_E = cc(g[:E].mean(0), F)
        assert rho_E == pytest.approx(rho_inf / attenuation_factor(c_true, E), rel=0.12), E
    # and the correction inverts it: disattenuating the E-member measurement recovers rho_inf
    c_hat = battery_c(member_noise_ratio(g)[0])
    for E in (1, 4, 16):
        assert disattenuate(cc(g[:E].mean(0), F), c_hat, E) == pytest.approx(
            rho_inf, rel=0.12), E


def test_disattenuate_composes_both_sides_and_needs_R():
    assert disattenuate(0.5, 3.0, 1) == pytest.approx(0.5 * np.sqrt(4.0))
    assert disattenuate(0.2, 3.0, 1, lam_F=1.0, R=2) == pytest.approx(
        0.2 * np.sqrt(4.0) * np.sqrt(1.5))
    assert disattenuate(0.9, 30.0, 1) == pytest.approx(1.0)              # clipped to [-1, 1]
    with pytest.raises(ValueError):
        disattenuate(0.5, 3.0, 1, lam_F=1.0)                             # lam_F without R


# -- the trap: the retrain-side estimator is the wrong pooling here --------- #
@pytest.mark.parametrize("alpha", [0.5, 0.25])
def test_retrain_side_pooling_is_biased_by_one_minus_alpha(alpha):
    """``seed_noise_variance_stacked`` pools across members at each mask, right only for noise iid
    across masks; on TDA-side noise it is ~1/(1-alpha) too large, inflating c ~2x at alpha=0.5."""
    g, S, tau, delta = _mask_structured(M=1500, n=1200, Q=40, alpha=alpha, seed=7)
    del S, tau, delta
    ours = np.median(member_noise_var(g))
    retrain_side = np.median(seed_noise_variance_stacked(g))
    assert retrain_side / ours == pytest.approx(1.0 / (1.0 - alpha), rel=0.15), (
        f"alpha={alpha}: retrain-side/ours = {retrain_side / ours:.3f}, "
        f"expected ~{1 / (1 - alpha):.3f}")


# -- degradation is reported, not silent ------------------------------------ #
def test_no_signal_case_is_clipped_and_reported():
    """With no signal at all, V_s is noise around zero: c must be reported as clipped
    rather than returned as a confident finite number."""
    rng = np.random.default_rng(8)
    g = rng.normal(size=(5, 200, 100))                     # pure noise, no shared signal
    c, diag = member_noise_ratio(g)
    assert diag["clipped_frac"] > 0.3, diag["clipped_frac"]
    assert np.all(c >= 0)
    assert np.isnan(battery_c(np.full(4, np.nan)))


def test_battery_c_counts_clipped_queries_as_infinite_not_dropped():
    """A clipped query's c exceeds every finite one, and the median must rank it so:
    dropping it would read median(1, 2, 3) = 2 here, not 3."""
    assert battery_c(np.array([1.0, 2.0, 3.0, np.inf, np.inf])) == 3.0
    assert battery_c(np.array([1.0, np.nan, 2.0, 3.0, np.inf, np.inf])) == 3.0   # NaN ignored


def test_battery_c_is_the_inverse_median_signal_to_noise():
    """With an odd number of queries the battery c is exactly 1 / median(V_s / sigma2),
    negative signal estimates included -- no clipping rule needed at all."""
    g, _ = _abstract(K=3, M=300, Q=401, c_true=30.0, seed=0)
    c, diag = member_noise_ratio(g)
    assert 0.05 < diag["clipped_frac"] < 0.5                  # the case that matters
    assert battery_c(c) == pytest.approx(1.0 / np.median(diag["V_s"] / diag["sigma2"]))


def test_battery_c_is_not_estimable_when_most_queries_clip():
    assert battery_c(np.array([1.0, 2.0, np.inf, np.inf, np.inf])) == np.inf


def test_battery_c_recovers_known_c_when_queries_clip():
    """A noisy design where ~18% of queries clip: the battery c stays on the truth, while
    the old rule (median of the finite c only) reads it systematically low."""
    new, old = [], []
    for seed in range(5):
        g, _ = _abstract(K=3, M=300, Q=401, c_true=30.0, seed=seed)
        c, _ = member_noise_ratio(g)
        new.append(battery_c(c) / 30.0)
        old.append(np.median(c[np.isfinite(c)]) / 30.0)
    assert 0.9 < np.mean(new) < 1.15, new
    assert np.mean(old) < 0.9, old                            # what the fix removes
