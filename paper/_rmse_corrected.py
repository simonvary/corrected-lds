"""Estimator error vs retrain budget on the cifar10/ResNet-9 bank (figure-1 panel).

At a budget of ``N = M * R`` retrains (M masks read out with R seeds each): the naive
R-seed-mean Spearman (standard practice) against the noise-corrected rank-first composition
(U-statistic mask-debiased Spearman times the battery-level seed-noise factor, what
``evaluate_pool`` reports as ``debiased_noise_corrected_spearman``), for the EK-FAC E=30
full-set bank (``pool_tda_ekfac``, damping 1e-8, the influence-function protocol).  Each
point is the RMSE over B subsamples of m masks (of M=300) and R seeds (of 40) to the truth,
the corrected composition on the full 300 x 40 pool.  Naive lines flatten at their
attenuation floor while corrected ones keep improving.  CPU-only; ~10 min at B=200.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import bank_stamp, run

from corrected_lds import (
    battery,
    corrected_pearson,
    load_pool,
    load_pool_tda,
    pearson_to_spearman,
    predictions_tda,
    seed_noise_variance_stacked,
)
from corrected_lds.rank_debias import unbiased_spearman

CLIP = 2.0
# masks per subsample; each line starts at its smallest m with N = m * R >= 9 and m >= 2 (3
# for the U-statistic).  Naive R=40 uses every seed: its spread is masks-only, 0 at m=300.
M_FILL = [5, 8, 12, 25, 50, 100, 200, 300]
M_MIN = {"naive": 2, "corrected": 3}
ESTIMATORS = [("naive", 5), ("naive", 10), ("naive", 20), ("naive", 40),
              ("corrected", 2), ("corrected", 3), ("corrected", 5)]
B = 200                                        # subsample replicates per point
SEED = 0


def _colwise_corr(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))


def _spearman_kendall(g, y):
    """Per-query sample Spearman and Kendall tau-a (mirrors evaluate._rank_correlations)."""
    m = g.shape[0]
    spearman = _colwise_corr(stats.rankdata(g, axis=0), stats.rankdata(y, axis=0))
    kendall = np.zeros(g.shape[1])
    for left in range(m - 1):
        kendall += (np.sign(g[left] - g[left + 1:]) * np.sign(y[left] - y[left + 1:])).sum(axis=0)
    return spearman, kendall / (m * (m - 1) // 2)


def _spearman_to_pearson(s):
    """Inverse Gaussian bridge (evaluate's rank-first composition)."""
    return 2.0 * np.sin(np.pi * np.clip(s, -1.0, 1.0) / 6.0)


def _estimate(kind, g_sub, margins_sub):
    """One estimate from a (masks, seeds) subsample.  ``margins_sub``: (r, m, Q)."""
    r = margins_sub.shape[0]
    ybar = margins_sub.mean(axis=0)
    if kind == "naive":
        sp = _colwise_corr(stats.rankdata(g_sub, axis=0), stats.rankdata(ybar, axis=0))
        return float(battery(sp))
    sigma2 = seed_noise_variance_stacked(margins_sub)
    theta = float(battery(corrected_pearson(g_sub, ybar, sigma2 / r, clip=CLIP)[0]))
    sp_q, kd_q = _spearman_kendall(g_sub, ybar)
    debiased = float(battery(unbiased_spearman(sp_q, kd_q, g_sub.shape[0])))
    rho_R = float(battery(_colwise_corr(g_sub, ybar)))
    factor = theta / rho_R if abs(rho_R) > 1e-12 else 1.0
    return float(pearson_to_spearman(factor * _spearman_to_pearson(debiased)))


def compute(banks_root):
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    g = predictions_tda(pool, load_pool_tda(os.path.join(bank, "pool_tda_ekfac.npz")))
    truth = _estimate("corrected", g, pool.margins)
    # Predicted naive RMSE floors (m -> inf): truth - bridge(bridge^-1(truth) / sqrt(1 +
    # lambda/R)), lambda the pool's battery median: a prediction, not a fit to these curves.
    sigma2 = seed_noise_variance_stacked(pool.margins)
    _, diag = corrected_pearson(g, pool.margins.mean(axis=0), sigma2 / pool.R,
                                clip=CLIP)
    lam_med = float(np.median(diag["lambda_q"] * pool.R))
    p_inf = _spearman_to_pearson(truth)
    floors = {str(R): float(truth - pearson_to_spearman(p_inf / np.sqrt(1 + lam_med / R)))
              for kind, R in ESTIMATORS if kind == "naive"}
    series = []
    for kind, R in ESTIMATORS:
        rows = []
        ms = [m for m in sorted({M_MIN[kind], *M_FILL})
              if m >= M_MIN[kind] and m * R >= 9]
        for m in ms:
            rng = np.random.default_rng([SEED, R, m, int(kind == "corrected")])
            ests = np.empty(B)
            for b in range(B):
                masks = rng.choice(pool.M, size=m, replace=False)
                seeds = rng.choice(pool.R, size=R, replace=False)
                ests[b] = _estimate(kind, g[masks], pool.margins[np.ix_(seeds, masks)])
            rows.append({"N": m * R, "m": m,
                         "rmse": float(np.sqrt(np.mean((ests - truth) ** 2))),
                         "bias": float(ests.mean() - truth),
                         "sd": float(ests.std(ddof=1))})
            print(f"[budget_rmse] {kind} R={R} m={m}: rmse {rows[-1]['rmse']:.4f}",
                  flush=True)
        series.append({"kind": kind, "R": R, "rows": rows})
    return {"banks": {"cifar10": bank_stamp(pool)},
            "naive_floors": floors, "lambda_med": lam_med,
            "params": {"clip": CLIP, "m_fill": M_FILL, "m_min": M_MIN, "B": B,
                       "seed": SEED,
                       "attributor": "ekfac E=30 full-set (pool_tda_ekfac, all replicas)",
                       "truth": "corrected composition on the full 300 x 40 pool"},
            "truth_rho_inf_spearman": truth, "series": series}


# linestyle and hue family tell naive from corrected; markers and an R label tell the R
_FAMILY = {
    "naive": {"colors": ["#dfa08a", "#c44536", "#8a3327", "#4e1d16"],
              "linestyle": (0, (4, 2.4)), "label": "naive"},
    "corrected": {"colors": ["#7fb3d8", "#1b6ca8", "#123f63"], "linestyle": "-",
                  "label": "corrected"},
}
_MARKERS = ["o", "s", "^", "D"]


_XTICKS = [10, 50, 200, 1000, 12000]     # thinned for a 1/3-page square (6000 collides with 12000)


if __name__ == "__main__":
    run("rmse_corrected", compute)
