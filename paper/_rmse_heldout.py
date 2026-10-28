"""Figure-1 panel (a), the paper's default: estimator RMSE against a held-out reference.

``_rmse_corrected``'s experiment (CIFAR-10 / ResNet-9, EK-FAC E=30, B=200 subsamples per
point), but each replicate splits the 40 seeds into random halves of 20 (``_heldout.halves``):
the reference is the corrected composition on all 300 masks from half A, the estimator
draws m masks and R seeds from half B.  Masks are shared, so the target is the noise-free
LDS on the bank's 300 subsets (finite population, as panel (c)).  Naive R=40 would need
every seed, so neither it nor its floor is drawn; the floors come from half A's reference
and lambda, averaged over the splits.  ``--recompute --banks-root <dir>`` needs
cifar10/{pool,pool_tda_ekfac}.npz; parallel processes, a few minutes on CPU.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rmse_corrected as base
from _harness import bank_stamp, run
from _heldout import halves

from corrected_lds import (
    corrected_pearson,
    load_pool,
    load_pool_tda,
    pearson_to_spearman,
    predictions_tda,
    seed_noise_variance_stacked,
)

ESTIMATORS = [("naive", 5), ("naive", 10), ("naive", 20),
              ("corrected", 2), ("corrected", 3), ("corrected", 5)]
FLOOR_RS = [5, 10, 20]           # the measured naive lines' floors; no R=40 line
_SHARED = {}                     # pool arrays, inherited by the forked workers


def _ref_and_floors(b):
    """Split b's half-A reference and the predicted naive floors it implies."""
    g, margins = _SHARED["g"], _SHARED["margins"]
    a, _ = halves(margins.shape[0], b)
    ma = margins[a]
    ref = base._estimate("corrected", g, ma)
    _, diag = corrected_pearson(g, ma.mean(axis=0), seed_noise_variance_stacked(ma) / len(a),
                                clip=base.CLIP)
    lam = float(np.median(diag["lambda_q"] * len(a)))
    p = base._spearman_to_pearson(ref)
    floors = {R: float(ref - pearson_to_spearman(p / np.sqrt(1 + lam / R))) for R in FLOOR_RS}
    return ref, floors, lam


def _series(job):
    kind, R = job
    g, margins, refs = _SHARED["g"], _SHARED["margins"], _SHARED["refs"]
    M = margins.shape[1]
    rows = []
    for m in [m for m in sorted({base.M_MIN[kind], *base.M_FILL})
              if base.M_MIN[kind] <= m <= M and m * R >= 9]:
        rng = np.random.default_rng([base.SEED, R, m, int(kind == "corrected"), 1])
        err = np.empty(base.B)
        for b in range(base.B):
            _, held = halves(margins.shape[0], b)
            masks = rng.choice(M, size=m, replace=False)
            seeds = rng.choice(held, size=R, replace=False)
            err[b] = base._estimate(kind, g[masks], margins[np.ix_(seeds, masks)]) - refs[b]
        rows.append({"N": m * R, "m": m, "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "bias": float(err.mean()), "sd": float(err.std(ddof=1))})
    print(f"[_rmse_heldout] {kind} R={R}: rmse at N={rows[-1]['N']} "
          f"{rows[-1]['rmse']:.4f}", flush=True)
    return {"kind": kind, "R": R, "rows": rows}


def compute(banks_root):
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    g = predictions_tda(pool, load_pool_tda(os.path.join(bank, "pool_tda_ekfac.npz")))
    truth = base._estimate("corrected", g, pool.margins)
    _SHARED.update(g=g, margins=pool.margins)
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1), mp_context=ctx) as ex:
        per_split = list(ex.map(_ref_and_floors, range(base.B)))
    refs = np.array([r for r, _, _ in per_split])
    _SHARED["refs"] = refs
    print(f"[_rmse_heldout] references: mean {refs.mean():.5f} sd {refs.std(ddof=1):.5f} "
          f"(40-seed value {truth:.5f})", flush=True)
    with ProcessPoolExecutor(max_workers=len(ESTIMATORS), mp_context=ctx) as ex:
        series = list(ex.map(_series, ESTIMATORS))
    return {"banks": {"cifar10": bank_stamp(pool)},
            "naive_floors": {str(R): float(np.mean([f[R] for _, f, _ in per_split]))
                             for R in FLOOR_RS},
            "lambda_med": float(np.mean([lam for _, _, lam in per_split])),
            "truth_rho_inf_spearman": truth,
            "reference": {"half": pool.R // 2, "mean": float(refs.mean()),
                          "sd": float(refs.std(ddof=1))},
            "params": {"clip": base.CLIP, "m_fill": base.M_FILL, "m_min": base.M_MIN,
                       "B": base.B, "seed": base.SEED,
                       "attributor": "ekfac E=30 full-set (pool_tda_ekfac, all replicas)",
                       "reference": "corrected composition on all masks from seed half A "
                                    "(_heldout.halves, a fresh split per replicate)",
                       "estimators": "m masks from all, R seeds from half B",
                       "floors": "from half A's reference and lambda, mean over splits"},
            "series": series}


if __name__ == "__main__":
    sys.exit(run("rmse_heldout", compute))
