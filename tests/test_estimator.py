"""Synthetic, CPU-only tests of corrected_lds.estimator.  Every test seeds its own Generator;
the statistical thresholds were pinned by an offline MC sweep against this estimator.
"""

import itertools

import numpy as np
import pytest

from corrected_lds.estimator import (
    battery,
    corrected_pearson,
    pearson_to_spearman,
    reconstruct_rho_R,
    seed_noise_variance_stacked,
)


def _gen(rng, M, Q, rho, lam):
    """Bivariate-normal LDS sim: sigma2_F = 1, sigma2_eps = lam, Corr(g, F) = rho.
    Returns predictions g, and two independent single-seed readouts y1, y2, all (M, Q)."""
    F = rng.standard_normal((M, Q))
    g = rho * F + np.sqrt(1.0 - rho**2) * rng.standard_normal((M, Q))
    sig = np.sqrt(lam)
    y1 = F + sig * rng.standard_normal((M, Q))
    y2 = F + sig * rng.standard_normal((M, Q))
    return g, y1, y2


def _s2(y1, y2):
    """Seed-noise variance from two seeds of the same masks."""
    return seed_noise_variance_stacked(np.stack([y1, y2]))


def test_seed_noise_variance_unbiased():
    """The paired-diff sigma2_eps is unbiased: |mean(sigma2_hat) - sigma2| < 3 SE in 2000 sims."""
    rng = np.random.default_rng(101)
    sigma2_true = 2.3
    m, sims = 40, 2000
    ests = np.empty(sims)
    for s in range(sims):
        F = 1.5 * rng.standard_normal((m, 1))
        y1 = F + np.sqrt(sigma2_true) * rng.standard_normal((m, 1))
        y2 = F + np.sqrt(sigma2_true) * rng.standard_normal((m, 1))
        ests[s] = _s2(y1, y2)[0]
    se = ests.std(ddof=1) / np.sqrt(sims)
    assert abs(ests.mean() - sigma2_true) < 3 * se


def test_two_seed_stacked_is_the_paired_difference():
    """ddof=1 over two seeds is exactly (y1 - y2)^2 / 2 averaged over masks."""
    rng = np.random.default_rng(909)
    y1 = rng.standard_normal((30, 4))
    y2 = rng.standard_normal((30, 4))
    assert np.allclose(_s2(y1, y2), ((y1 - y2) ** 2 / 2).mean(axis=0))


def test_reconstruct_R1_is_plain_pearson_exact():
    """k=1, R=1 gives plain Pearson to machine precision even for garbage (inf/nan) seed noise:
    the 1/k - 1/R coefficient is exactly zero and short-circuits."""
    rng = np.random.default_rng(202)
    Q = 5
    g = rng.standard_normal((60, Q))
    y = rng.standard_normal((60, Q))
    plain = np.array([np.corrcoef(g[:, q], y[:, q])[0, 1] for q in range(Q)])

    rho, _ = reconstruct_rho_R(g, y, _s2(y[:20], y[:20] + 1), k=1, R=1)
    assert np.allclose(rho, plain, atol=1e-12)

    for garbage in (1e6 * rng.standard_normal(Q), np.full(Q, np.inf), np.full(Q, np.nan)):
        rho_g, _ = reconstruct_rho_R(g, y, garbage, k=1, R=1)
        assert np.allclose(rho_g, plain, atol=1e-12)


def test_reconstruct_Rinf_matches_full_correction():
    """As R -> inf, reconstruct_rho_R (k=1) becomes corrected_pearson's full correction."""
    rng = np.random.default_rng(808)
    g, y1, y2 = _gen(rng, 120, 8, 0.4, 2.0)
    s2 = _s2(y1, y2)
    full, _ = corrected_pearson(g, y1, s2)
    recon, _ = reconstruct_rho_R(g, y1, s2, k=1, R=10**9)
    assert np.allclose(full, recon, atol=1e-6)


def test_corrected_recovers_rho():
    """The corrected estimator recovers rho_inf: mean within 0.01 at M=400, R=2, over 2000 reps."""
    rng = np.random.default_rng(303)
    rho, lam, M, Q, reps = 0.35, 2.0, 400, 16, 2000
    means = np.empty(reps)
    for r in range(reps):
        g, y1, y2 = _gen(rng, M, Q, rho, lam)
        rho_q, _ = corrected_pearson(g, y1, _s2(y1, y2))
        means[r] = battery(rho_q)
    assert abs(means.mean() - rho) < 0.01


def test_clip_floored_fraction_monotone():
    """More pairs m -> less-noisy sigma2_eps -> fewer spurious over-corrections floored."""
    rng = np.random.default_rng(606)
    rho, lam, M, Q, reps = 0.35, 2.0, 200, 64, 200
    grid = [8, 20, 60, 200]
    fracs = []
    for m in grid:
        acc = 0.0
        for _ in range(reps):
            g, y1, y2 = _gen(rng, M, Q, rho, lam)
            _, diag = corrected_pearson(g, y1, _s2(y1[:m], y2[:m]))
            acc += diag["floored_frac"]
        fracs.append(acc / reps)
    assert all(a > b for a, b in itertools.pairwise(fracs)), fracs


def test_battery_is_the_plain_mean():
    assert battery(np.array([0.2, 0.4, 0.6])) == pytest.approx(0.4)


def test_pearson_to_spearman_vectorised_and_clipped():
    assert pearson_to_spearman(np.array([0.0, 1.0])).shape == (2,)
    assert pearson_to_spearman(0.0) == pytest.approx(0.0)
    assert pearson_to_spearman(1.0) == pytest.approx(1.0)          # 6/pi * arcsin(1/2) = 1
    assert pearson_to_spearman(5.0) == pytest.approx(pearson_to_spearman(1.0))   # clipped
