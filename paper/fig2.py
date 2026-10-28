"""Figure 2 with column (a, d, g) as RMSE to the noise-free LDS, in figure 1(a)'s form.

(a, d, g): EK-FAC at E=1 (one model), held out: corrected R=2 and R=5, and naive
R = 5, 10, 20 on CIFAR-10 and R = 3, 5, 10 on QNLI and WikiText-2.  Each point is the RMSE
over B=200 replicates; replicate b splits the seeds (``_heldout.split``) into N_REF
reference seeds and the rest, from which the estimator draws R seeds (m masks from all).
The reference is the corrected composition on all masks from the reference seeds, so a
few-seed reference adds its noise to the RMSE (``reference.sd``).  Dotted: each naive
line's predicted RMSE plateau, from the reference seeds' reference and lambda, averaged
over the splits.  Draws without a rank correlation (tied bf16 readouts at QNLI's smallest
m) are counted in ``n_defined``; a point is drawn only when all B are defined.

(b, e, h) and (c, f, i) are read from the committed ``fig2_panels.json``, whose banks are
stamped in ``fig2_banks``.  ``--recompute --banks-root <dir>`` needs every bank's pool.npz
and pool_tda_ekfac.npz; three parallel processes on CPU.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fig2_panels as panels
import _rmse_corrected as base
import _shared
from _harness import PLOTDATA_DIR, bank_stamp, run
from _heldout import seed_noise, split

from corrected_lds import (
    corrected_pearson,
    load_pool,
    load_pool_tda,
    load_tauq,
    pearson_to_spearman,
    predictions,
    predictions_tda,
)
from corrected_lds.evaluate import _seed_noise_fn

METHOD = ("pool_tda_ekfac", "EK-FAC")
# bank -> (mean-scores tauq read instead of pool_tda_ekfac, its ensemble size E)
TAUQ = {}
NAIVE_R = {"cifar10": [5, 10, 20], "qnli": [3, 5, 10], panels.BANK: [3, 5, 10]}
# reference seeds per replicate: figure 2's N_REF_A
N_REF = dict(panels.N_REF_A)
CORRECTED_R = [2, 5]
B = base.B                       # 200 replicates per point, as figure 1(a)
# figure 1(a)'s colours and markers, assigned in _style
_NAIVE = base._FAMILY["naive"]
_CORR = base._FAMILY["corrected"]
_CORR_IDX = {2: 0, 5: 2}
LABEL_FS = 11.0                  # the R labels at each curve's right end


def _series_keys(bank):
    return ([("naive", R) for R in NAIVE_R[bank]]
            + [("corrected", R) for R in CORRECTED_R])


def _m_grid(kind, R, M):
    """Figure 1(a)'s mask grid, cut to this pool and closed by its M."""
    return [m for m in sorted({base.M_MIN[kind], *base.M_FILL, M})
            if base.M_MIN[kind] <= m <= M and m * R >= 9]


def _ref_and_floors(g, ma, s2_fn, floor_rs):
    """The reference seeds' noise-free LDS, the naive plateaus it predicts, and lambda."""
    ref = base._estimate("corrected", g, ma)
    _, diag = corrected_pearson(g, ma.mean(axis=0), s2_fn(ma) / len(ma), clip=base.CLIP)
    lam = float(np.nanmedian(diag["lambda_q"] * len(ma)))
    p = base._spearman_to_pearson(ref)
    return ref, {str(R): float(ref - pearson_to_spearman(p / np.sqrt(1 + lam / R)))
                 for R in floor_rs}, lam


def _panel(job):
    banks_root, bank = job
    pool = load_pool(os.path.join(banks_root, bank, "pool.npz"))
    if bank in TAUQ:                     # the mean scores, projected onto every subset
        file, E = TAUQ[bank]
        g = predictions(pool, load_tauq(os.path.join(banks_root, bank, f"{file}.npz")))
    else:
        file, E = METHOD[0], len(_shared.E1_REPLICAS)                # E = 1: one model
        g = predictions_tda(pool, load_pool_tda(os.path.join(banks_root, bank, f"{file}.npz")),
                            _shared.E1_REPLICAS)
    s2_fn = _seed_noise_fn(pool)
    n_ref = N_REF[bank]
    splits = [split(pool.R, b, n_ref) for b in range(B)]
    keys = _series_keys(bank)
    if max(R for _, R in keys) > pool.R - n_ref:
        raise SystemExit(f"[fig2] {bank}: R up to {max(R for _, R in keys)} needs more "
                         f"than the {pool.R - n_ref} estimator seeds")
    with seed_noise(pool):
        per_split = [_ref_and_floors(g, pool.margins[a], s2_fn, NAIVE_R[bank])
                     for a, _ in splits]
        refs = np.array([r for r, _, _ in per_split])
        series = []
        for kind, R in keys:
            rows = []
            for m in _m_grid(kind, R, pool.M):
                rng = np.random.default_rng([base.SEED, R, m, int(kind == "corrected"), 2,
                                             len(bank)])
                est = np.full(B, np.nan)
                for b in range(B):
                    masks = rng.choice(pool.M, size=m, replace=False)
                    seeds = rng.choice(splits[b][1], size=R, replace=False)
                    est[b] = base._estimate(kind, g[masks], pool.margins[np.ix_(seeds, masks)])
                ok = np.isfinite(est)
                err = est[ok] - refs[ok]
                rows.append({"N": m * R, "m": m, "n_defined": int(ok.sum()),
                             "rmse": float(np.sqrt(np.mean(err ** 2))) if ok.any() else None,
                             "bias": float(err.mean()) if ok.any() else None,
                             "sd": float(err.std(ddof=1)) if ok.sum() > 1 else None})
            series.append({"kind": kind, "R": R, "rows": rows})
            print(f"[fig2] {bank} {kind} R={R}: "
                  + " ".join(f"{r['N']}:{r['rmse']:.4f}" for r in rows if r["rmse"] is not None),
                  flush=True)
    print(f"[fig2] {bank}: reference {refs.mean():.4f} (sd {refs.std(ddof=1):.4f}, "
          f"{n_ref} of {pool.R} seeds)", flush=True)
    return bank, {"stamp": bank_stamp(pool),
                  "panel": {"method": METHOD[1], "file": file, "E": E,
                            "n_ref": n_ref, "n_est": pool.R - n_ref,
                            "reference": {"mean": float(refs.mean()),
                                          "sd": float(refs.std(ddof=1))},
                            "naive_floors": {str(R): float(np.mean([f[str(R)]
                                                                    for _, f, _ in per_split]))
                                             for R in NAIVE_R[bank]},
                            "lambda_med": float(np.mean([lam for _, _, lam in per_split])),
                            "series": series}}


def compute(banks_root):
    with open(os.path.join(PLOTDATA_DIR, "fig2_panels.json")) as fh:
        f2 = json.load(fh)
    jobs = [(banks_root, bank) for bank in panels.ROWS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        out = dict(ex.map(_panel, jobs))
    # (a, d, g) may read another pool than fig2_panels.json's (b-i), so the stamps stay apart
    return {"banks": {b: o["stamp"] for b, o in out.items()},
            "fig2_banks": f2["banks"],
            "fig2": f2["fig2"], "fig3": f2["fig3"],
            "rmse": {b: o["panel"] for b, o in out.items()},
            "params": {"rows": list(panels.ROWS),
                       "derived_from": "fig2_panels.json for (b-i), stamped in fig2_banks",
                       "a": "RMSE to the noise-free LDS (corrected composition on all masks "
                            "from the N_REF reference seeds), EK-FAC E=1"
                            + "".join(f" ({b}: E={e} from {f})" for b, (f, e) in TAUQ.items())
                            + ", held out",
                       "n_ref": {b: N_REF[b] for b in panels.ROWS},
                       "naive_R": NAIVE_R, "corrected_R": CORRECTED_R, "B": B,
                       "m_fill": base.M_FILL, "m_min": base.M_MIN, "seed": base.SEED,
                       "clip": base.CLIP}}


def _style(kind, R, naive_rs):
    """Colour, line style and marker: naive by rank among the naive Rs, corrected as fig1(a)."""
    if kind == "naive":
        i = list(naive_rs).index(R)
        return _NAIVE["colors"][i], _NAIVE["linestyle"], base._MARKERS[i]
    i = _CORR_IDX[R]
    return _CORR["colors"][i], _CORR["linestyle"], base._MARKERS[i]


def draw_a(ax, data, bank):
    """One setup's RMSE panel: estimators, dotted naive plateaus, R at each curve's end."""
    from matplotlib.ticker import NullLocator
    p = data["rmse"][bank]
    naive_rs = p.get("naive_R", NAIVE_R[bank])
    ends = []
    for s in p["series"]:
        colour, ls, marker = _style(s["kind"], s["R"], naive_rs)
        rows = [r for r in s["rows"] if r["n_defined"] == B]
        ax.plot([r["N"] for r in rows], [r["rmse"] for r in rows], color=colour, linestyle=ls,
                marker=marker, markersize=3.6, linewidth=1.5, zorder=3)
        if s["kind"] == "naive":
            ax.axhline(p["naive_floors"][str(s["R"])], color=colour, linestyle=":",
                       linewidth=1.1, alpha=0.55, zorder=1)
        ends.append((rows[-1]["N"], rows[-1]["rmse"], f"$R{{=}}{s['R']}$", colour))
    n_all = [r["N"] for s in p["series"] for r in s["rows"]]
    y_all = ([r["rmse"] for s in p["series"] for r in s["rows"] if r["n_defined"] == B]
             + list(p["naive_floors"].values()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    xmax = max(n_all) * 3.2                    # the right-hand room holds the R labels
    ax.set_xlim(7, xmax)
    ax.set_ylim(0.8 * min(y_all), 1.25 * max(y_all))
    xt = [t for t in (10, 100, 1000) if t < max(n_all) * 1.5]
    ax.set_xticks(xt, [str(t) for t in xt])
    yt = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
          if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(yt, [f"{t:g}" for t in yt])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.set_ylabel(r"RMSE to $\rho^{\mathtt{s}}$")
    # R labels just right of each curve's last point, pushed apart vertically (in log y)
    lo, hi = np.log(ax.get_ylim())
    ends.sort(key=lambda e: e[1])
    pos = [(np.log(y) - lo) / (hi - lo) for _, y, _, _ in ends]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 0.11)
    if pos and pos[-1] > 0.97:
        pos = [v - (pos[-1] - 0.97) for v in pos]
    for (n, _, text, colour), y in zip(ends, pos):
        ax.text(n * 1.25, np.exp(lo + y * (hi - lo)), text, color=colour, fontsize=LABEL_FS,
                va="center", ha="left")


def plot(data, out_path):
    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=_NAIVE["colors"][1], linestyle=_NAIVE["linestyle"], marker="s",
                  markersize=4.0, linewidth=1.5, label="naive"),
           Line2D([], [], color=_CORR["colors"][1], linestyle="-", marker="o", markersize=4.0,
                  linewidth=1.5, label="corrected"),
           Line2D([], [], color="#777777", linestyle=":", linewidth=1.3, label="plateau")]
    panels.plot(data, out_path, draw_a=draw_a, key_a=key, xlabel_a=_shared._XLABEL_B)


if __name__ == "__main__":
    sys.exit(run("fig2", compute, plot))
