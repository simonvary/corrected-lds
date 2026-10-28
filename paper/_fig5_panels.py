"""The bridge check ``fig5`` draws (CIFAR-10 / ResNet-9, QNLI / BERT): is the arcsin bridge
``(6/pi) arcsin(r/2)``, exact for a bivariate normal (score, readout) pair, safe on figure
2(b,e)'s reconstructed rho_5 route?  Same three methods (E=1), 5-seed readouts:

  (a, c) the consequence: reconstructed minus naive 5-seed Spearman on all masks (95%
         block-jackknife interval) against +-1 SE of one estimate (``bridge_bias.json``).
  (b, d) the condition: the chi-square (Mahalanobis) Q-Q plot of D^2, pooled over queries,
         averaged over 10 of figure 2's 5-seed subsets; the identity line means bivariate
         normal.  Kept for the appendix: the detrended ``curve`` and each method's 90%
         Gaussian ``band``, from 40 joint Haar rotations of the mask axis (each readout
         centred first, so its mean cannot become a spurious Gaussian component).

The null is exact at finite M (``_assumptions._qq_grid``).  ``--recompute`` needs the banks
(a few minutes on CPU).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _assumptions as fa
import _shared
from _harness import PLOTDATA_DIR, bank_stamp, run

from corrected_lds import load_pool, load_pool_tda, predictions_tda

N_SUBSETS = 10                   # of figure 2's 40 random 5-seed subsets
N_SURR = 40                      # rotation surrogates per bank
BAND_PCT = (5.0, 95.0)
SHARED_BAND_TOL = 0.25           # share one band if the methods' widths agree to 25%


def _haar(m, rng):
    q, r = np.linalg.qr(rng.standard_normal((m, m)))
    return q * np.sign(np.diag(r))


def _curve(g, seeds, subsets, probs, null_q):
    """Detrended D^2 quantile curve, pooled over queries, averaged over the subsets."""
    return np.mean([np.quantile(fa._mahalanobis2(g, seeds[s].mean(axis=0)), probs) - null_q
                    for s in subsets], axis=0)


def _qq_bank(banks_root, bank, rng):
    pool = load_pool(os.path.join(banks_root, bank, "pool.npz"))
    sub_rng = np.random.default_rng([_shared.SEED, 99])          # figure 2's own rho_5 draw
    subsets = [sub_rng.choice(pool.R, _shared.TARGET_R, replace=False)
               for _ in range(_shared.N_TRUTH)][:N_SUBSETS]
    probs, null_q = fa._qq_grid(pool.M)
    gs = {tda: predictions_tda(pool, load_pool_tda(os.path.join(banks_root, bank,
                                                                f"{tda}.npz")),
                               _shared.E1_REPLICAS)
          for tda, _ in _shared.B_METHODS[bank]}
    data = {tda: _curve(g, pool.margins, subsets, probs, null_q) for tda, g in gs.items()}
    seeds_c = pool.margins - pool.margins.mean(axis=1, keepdims=True)
    surr = {tda: [] for tda in gs}
    for i in range(N_SURR):
        o = _haar(pool.M, rng)
        rot = np.matmul(o, seeds_c)
        for tda, g in gs.items():
            surr[tda].append(_curve(o @ (g - g.mean(axis=0)), rot, subsets, probs, null_q))
        if (i + 1) % 10 == 0:
            print(f"[fig5] {bank}: {i + 1}/{N_SURR} surrogates", flush=True)
    methods = []
    for tda, label in _shared.B_METHODS[bank]:
        lo, hi = np.percentile(surr[tda], BAND_PCT, axis=0)
        exceed = float(((data[tda] < lo) | (data[tda] > hi)).mean())
        methods.append({"tda": tda, "label": label, "curve": data[tda].tolist(),
                        "band": [lo.tolist(), hi.tolist()], "exceed": exceed,
                        "max_dev": float(np.abs(data[tda]).max())})
        print(f"[fig5] {bank} {label}: max |dev| {np.abs(data[tda]).max():.3f},"
              f" outside the 90% band at {100 * exceed:.0f}% of points", flush=True)
    widths = np.array([np.mean(np.subtract(m["band"][1], m["band"][0])) for m in methods])
    spread = float(widths.max() / widths.min() - 1.0)
    shared = [np.min([m["band"][0] for m in methods], axis=0).tolist(),
              np.max([m["band"][1] for m in methods], axis=0).tolist()]
    return bank_stamp(pool), {"null_q": null_q.tolist(), "methods": methods,
                              "band_spread": spread, "shared_band": shared,
                              "use_shared": spread <= SHARED_BAND_TOL}


def compute(banks_root):
    with open(os.path.join(PLOTDATA_DIR, "bridge_bias.json")) as fh:
        bias = json.load(fh)
    rng = np.random.default_rng([_shared.SEED, 11])
    banks, qq = {}, {}
    for bank in _shared.B_METHODS:
        banks[bank], qq[bank] = _qq_bank(banks_root, bank, rng)
        if banks[bank] != bias["banks"][bank]:
            raise SystemExit(f"[fig5] {bank}: bank differs from bridge_bias")
        print(f"[fig5] {bank}: band widths agree to "
              f"{100 * qq[bank]['band_spread']:.0f}% -> "
              f"{'one shared band' if qq[bank]['use_shared'] else 'per-method bands'}",
              flush=True)
    return {"banks": banks, "bias": bias["panels"], "qq": qq,
            "params": {"derived_from": ["bridge_bias.json"], "n_subsets": N_SUBSETS,
                       "n_surrogates": N_SURR, "band_pct": BAND_PCT,
                       "shared_band_tol": SHARED_BAND_TOL, "E": "E1 replicas",
                       "readout": f"{_shared.TARGET_R}-seed mean, figure 2's subsets"}}


if __name__ == "__main__":
    sys.exit(run("fig5_panels", compute))
