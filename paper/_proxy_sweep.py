"""Appendix diagnostic: joint normality along the seed-averaging path toward (g, F).

The bridge needs the latent ``(g, F)`` bivariate normal; F is never observed, so the Q-Q
checks test the proxy ``(g, fbar_R)``.  The mean of k seeds, ``fbar_k``, carries noise share
``lambda/k``: a departure that grows with k belongs to (g, F), one that shrinks to the seed
noise, and the latent value is the ``k -> infinity`` extrapolation.  Query-mean statistics:

  bridge   Spearman - (6/pi) arcsin(Pearson/2), the relation the estimator uses
  mardia   Mardia's bivariate kurtosis b_{2,2}
  reject   per-query Mardia rejection rate at 0.05 (asymptotic p-values)
  lin      corr(resid, g^2): curvature of E[f | g]
  het      corr(resid^2, g^2): heteroscedasticity of f given g
  qq_rms   RMS deviation of the pooled D^2 quantiles from their exact Beta null
  skew_f, kurt_f   marginal skew / excess kurtosis of fbar_k

Each is read against a Gaussian null built the same way, not a formula (at finite M even a
Gaussian pair's bridge residual is not 0, and Mardia's textbook null assumes divisor M): one
Haar rotation of the mask axis, applied jointly to g and all R seed readouts before the
k-blocks (``min(R // k, G_MAX)`` disjoint seed means) are formed, keeps every per-query
covariance and imposes joint normality.  Footings: E=1 (figure 2's g) and the full ensemble.

CPU-only.  ``plotdata/proxy_sweep.json`` keeps every surrogate draw (arrays to 6 significant
figures), so other bands or views need no banks; it is rewritten after each bank with
``"complete": false``, and missing banks are skipped and named.  Per ``panels[bank]``:

  ks, lambda_median, null_q (the exact-null D^2 quantiles the curves are offsets from)
  footings[E1 | full]:  E (attributor ensemble size), per_k, k_inf
    per_k[i]:  k, blocks, noise_share (= lambda_median / k);
               <stat> in STATS: data, control, null (N_SURR surrogate values), null_mean,
                   null_sd, null_lo, null_hi (BAND_PCT percentiles of null);
               qq_curve (data D^2 quantiles - null_q), qq_lo, qq_hi (surrogate band),
               qq_exceed, qq_exceed_null (leave-one-out surrogate exceedance)
    k_inf[<stat>]:  excess (data intercept in 1/k minus the null's), null_sd, z,
                    null (the N_SURR surrogate intercepts, centred)
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _assumptions as fa
import _shared
from _harness import PLOTDATA_DIR, SCHEMA, bank_stamp, run, to_jsonable

from corrected_lds import (
    corrected_pearson,
    load_pool,
    load_pool_tda,
    pearson_to_spearman,
    predictions_tda,
    seed_noise_variance_stacked,
)

SEED = 0
CLIP = 2.0
N_SURR = 40                      # Gaussian rotation surrogates per bank
G_MAX = 10                       # disjoint k-seed blocks averaged at each k
BAND_PCT = (5.0, 95.0)
K_GRID = {"cifar10": [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40],
          "qnli": [1, 2, 4, 6, 8, 10, 15, 20]}
FOOTINGS = {"E1": [0], "full": None}     # replicas passed to predictions_tda
STATS = ["bridge", "mardia", "reject", "lin", "het", "qq_rms", "skew_f", "kurt_f"]
BANKS = {spec["key"]: spec for spec in fa.BANKS}


def _haar(m, rng):
    """A Haar-distributed orthogonal matrix (QR with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((m, m)))
    return q * np.sign(np.diag(r))


def _pair_stats(g, rank_g, y, probs, null_q):
    """``STATS`` for ``(g, y)``, each a query mean, and the pooled D^2 quantile deviation curve."""
    m = g.shape[0]
    gs = (g - g.mean(axis=0)) / g.std(axis=0, ddof=1)
    ys = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    pear = (gs * ys).sum(axis=0) / (m - 1)
    spear = _shared._colcorr(rank_g, stats.rankdata(y, axis=0))
    d2 = fa._mahalanobis2(g, y)
    b2 = (d2.reshape(m, -1) ** 2).mean(axis=0)
    z = (b2 - 8.0 * (m - 1) / (m + 1)) / np.sqrt(64.0 / m)
    resid, g2 = ys - pear * gs, gs ** 2
    curve = np.quantile(d2, probs) - null_q
    return {"bridge": float(np.nanmean(spear - pearson_to_spearman(pear))),
            "mardia": float(b2.mean()),
            "reject": float((2.0 * stats.norm.sf(np.abs(z)) < 0.05).mean()),
            "lin": float(np.nanmean(_shared._colcorr(resid, g2))),
            "het": float(np.nanmean(_shared._colcorr(resid ** 2, g2))),
            "qq_rms": float(np.sqrt(np.mean(curve ** 2))),
            "skew_f": float(np.mean(stats.skew(y, axis=0))),
            "kurt_f": float(np.mean(stats.kurtosis(y, axis=0)))}, curve


def _blocks(seeds, k):
    """Disjoint k-seed means: ``(G, M, Q)`` with ``G = min(R // k, G_MAX)``."""
    g_n = min(seeds.shape[0] // k, G_MAX)
    return [seeds[j * k:(j + 1) * k].mean(axis=0) for j in range(g_n)]


def _sweep(g, seeds, ks, probs, null_q, transform=None):
    """Block-averaged stats and D^2 curve at every k; ``transform`` is the negative control."""
    rank_g = stats.rankdata(g, axis=0)
    out = []
    for k in ks:
        rows, curves = [], []
        for y in _blocks(seeds, k):
            s, c = _pair_stats(g, rank_g, y if transform is None else transform(y),
                               probs, null_q)
            rows.append(s)
            curves.append(c)
        out.append(({name: float(np.mean([r[name] for r in rows])) for name in STATS},
                    np.mean(curves, axis=0)))
    return out


def _intercept(ks, values):
    """The k -> infinity limit: intercept of a straight line in 1/k, as appendix (d)."""
    return float(np.polyfit(1.0 / np.asarray(ks, float), values, 1)[1])


def _sig(x):
    """An array as a list at 6 significant figures — the bulk of the saved file."""
    return [float(f"{v:.6g}") for v in np.ravel(x)]


def _exceed(curve, lo, hi):
    return float(((curve < lo) | (curve > hi)).mean())


def _bank(banks_root, key, rng):
    spec = BANKS[key]
    pool = load_pool(os.path.join(banks_root, key, "pool.npz"))
    tda = load_pool_tda(os.path.join(banks_root, key, f"{spec['tda']}.npz"))
    ks = [k for k in K_GRID[key] if k <= pool.R]
    probs, null_q = fa._qq_grid(pool.M)
    seeds = pool.margins - pool.margins.mean(axis=1, keepdims=True)
    gs = {f: predictions_tda(pool, tda, rep) for f, rep in FOOTINGS.items()}

    sigma2 = seed_noise_variance_stacked(pool.margins)
    _, diag = corrected_pearson(gs["E1"], pool.margins.mean(axis=0), sigma2 / pool.R,
                                clip=CLIP)
    lam_q = np.asarray(diag["lambda_q"]) * pool.R
    lam = float(np.median(lam_q))

    data = {f: _sweep(g, seeds, ks, probs, null_q) for f, g in gs.items()}
    ctrl = {f: _sweep(g, seeds, ks, probs, null_q, fa._control) for f, g in gs.items()}
    surr = {f: [] for f in gs}
    for i in range(N_SURR):
        o = _haar(pool.M, rng)
        rot = np.matmul(o, seeds)
        for f, g in gs.items():
            surr[f].append(_sweep(o @ (g - g.mean(axis=0)), rot, ks, probs, null_q))
        if (i + 1) % 10 == 0:
            print(f"[proxy_sweep] {key}: {i + 1}/{N_SURR} surrogates", flush=True)

    footings = {}
    for f in gs:
        per_k = []
        for j, k in enumerate(ks):
            sv = {n: np.array([s[j][0][n] for s in surr[f]]) for n in STATS}
            sc = np.array([s[j][1] for s in surr[f]])
            lo, hi = np.percentile(sc, BAND_PCT, axis=0)
            loo = [_exceed(sc[i], *np.percentile(np.delete(sc, i, axis=0), BAND_PCT,
                                                  axis=0)) for i in range(len(sc))]
            row = {"k": k, "blocks": min(pool.R // k, G_MAX), "noise_share": lam / k,
                   "qq_exceed": _exceed(data[f][j][1], lo, hi),
                   "qq_exceed_null": float(np.mean(loo)),
                   "qq_curve": _sig(data[f][j][1]), "qq_lo": _sig(lo), "qq_hi": _sig(hi)}
            for n in STATS:
                row[n] = {"data": data[f][j][0][n], "control": ctrl[f][j][0][n],
                          "null": _sig(sv[n]),
                          "null_mean": float(sv[n].mean()), "null_sd": float(sv[n].std(ddof=1)),
                          "null_lo": float(np.percentile(sv[n], BAND_PCT[0])),
                          "null_hi": float(np.percentile(sv[n], BAND_PCT[1]))}
            per_k.append(row)
        limit = {}
        for n in STATS:
            d_int = _intercept(ks, [r[n]["data"] - r[n]["null_mean"] for r in per_k])
            s_int = np.array([_intercept(ks, [s[j][0][n] for j in range(len(ks))])
                              for s in surr[f]])
            s_int -= s_int.mean()
            limit[n] = {"excess": d_int, "null_sd": float(s_int.std(ddof=1)),
                        "z": d_int / float(s_int.std(ddof=1)), "null": _sig(s_int)}
        footings[f] = {"E": len(FOOTINGS[f]) if FOOTINGS[f] else tda.R_tda,
                       "per_k": per_k, "k_inf": limit}
    return bank_stamp(pool), {"label": spec["label"], "color": spec["color"],
                              "attributor": spec["tda"], "lambda_median": lam, "ks": ks,
                              "null_q": null_q.tolist(), "footings": footings}


def _checkpoint(result):
    """Write the banks done so far in the final format, so an interrupted run still renders."""
    path = os.path.join(PLOTDATA_DIR, "proxy_sweep.json")
    with open(path + ".tmp", "w") as fh:
        json.dump({"schema": SCHEMA, "figure": "proxy_sweep", **to_jsonable(result)}, fh,
                  indent=1)
    os.replace(path + ".tmp", path)
    print(f"[proxy_sweep] checkpoint: {list(result['panels'])} -> {path}", flush=True)


def compute(banks_root):
    rng = np.random.default_rng(SEED)
    banks, out, skipped = {}, {}, []
    for key in BANKS:
        if not os.path.exists(os.path.join(banks_root, key, "pool.npz")):
            skipped.append(key)
            print(f"[proxy_sweep] {key}: no pool under {banks_root} — skipped", flush=True)
            continue
        banks[key], out[key] = _bank(banks_root, key, rng)
        _report(key, out[key])
        _checkpoint(_result(banks, out, skipped, complete=False))
    return _result(banks, out, skipped, complete=True)


def _result(banks, out, skipped, complete):
    return {"complete": complete, "banks": banks, "skipped": skipped,
            "params": {"seed": SEED, "clip": CLIP, "n_surrogates": N_SURR,
                       "g_max": G_MAX, "band_pct": BAND_PCT, "k_grid": K_GRID,
                       "footings": {f: ("E=1" if r else "full ensemble")
                                    for f, r in FOOTINGS.items()},
                       "control_p": fa.CONTROL_P,
                       "null": "joint Haar rotation of g and all R seed readouts on the "
                               "mask axis; k-blocks formed after rotation"},
            "panels": out}


def _report(key, b):
    for f, ft in b["footings"].items():
        print(f"\n[proxy_sweep] {key}  footing {f} (E={ft['E']})  "
              f"lambda={b['lambda_median']:.3f}")
        print("   k  noise  " + "  ".join(f"{n:>15s}" for n in STATS) + "   qq_exceed")
        for r in ft["per_k"]:
            cells = "  ".join(f"{r[n]['data'] - r[n]['null_mean']:+9.5f}"
                              f"({(r[n]['data'] - r[n]['null_mean']) / r[n]['null_sd']:+5.1f})"
                              for n in STATS)
            print(f"  {r['k']:2d}  {r['noise_share']:5.3f}  {cells}   "
                  f"{r['qq_exceed']:.2f}/{r['qq_exceed_null']:.2f}")
        print("  k->inf excess (z): " + "  ".join(
            f"{n} {v['excess']:+.5f} ({v['z']:+.1f})" for n, v in ft["k_inf"].items()))


if __name__ == "__main__":
    sys.exit(run("proxy_sweep", compute))
