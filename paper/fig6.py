"""Figure 6, RMSE form: fig1(a)'s estimator error against budget for every experiment and
attribution method, against a held-out reference.

``fig7``'s 3x3 grid (every attributor at E=10).  Each point is the RMSE over B=200
replicates, each splitting the seeds into halves (``_heldout.halves``): the reference is the
corrected composition on all masks from half A, the estimates use m masks and R seeds of
half B (masks shared: a finite-population target); one y axis, as RMSE has the same units
everywhere.  Series are fig1(a)'s that fit in half B.  Dotted: each measured naive line's
predicted RMSE floor, from half A's reference and lambda, averaged over splits; the
reference's spread across splits (``reference.sd``) is part of the RMSE.  Each point also
keeps its first draw (split 0), with split 0's reference and plateaus, for ``fig7``.  Draws
without a rank correlation (tied bf16 readouts at QNLI's m=2 and 3) are counted in
``n_defined``; a point is drawn only when all B are defined.  ``--recompute`` needs each
bank's pool.npz and pool_tda files (banks-manifest.json); nine processes, ~20 min on CPU.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rmse_corrected as base
import fig7 as single
from _harness import bank_stamp, run
from _heldout import halves

from corrected_lds import corrected_pearson, pearson_to_spearman

B = base.B                       # 200 subsample replicates per point, as fig1(a)


def _ref_and_floors(g, ma, s2_fn, floor_rs):
    """Half A's reference, the naive floors it predicts, and its lambda."""
    ref = base._estimate("corrected", g, ma)
    _, diag = corrected_pearson(g, ma.mean(axis=0), s2_fn(ma) / len(ma), clip=base.CLIP)
    lam = float(np.nanmedian(diag["lambda_q"] * len(ma)))
    p = base._spearman_to_pearson(ref)
    return ref, {str(R): float(ref - pearson_to_spearman(p / np.sqrt(1 + lam / R)))
                 for R in floor_rs}, lam


def _panel(job):
    """One cell: references and floors per split, RMSE / bias / sd over B draws per point."""
    banks_root, bank, row = job
    stem, label = single.ATTRIBUTORS[bank][row]
    pool, pt, g, s2_fn = single._setup(banks_root, bank, stem)
    n_ref = pool.R // 2
    series_keys = [(k, R) for k, R in single._series(pool) if R <= pool.R - n_ref]
    # a predicted floor only for a measured naive R
    floor_rs = [R for k, R in series_keys if k == "naive"]
    with single._seed_noise(s2_fn):
        per_split = [_ref_and_floors(g, pool.margins[halves(pool.R, b)[0]], s2_fn, floor_rs)
                     for b in range(B)]
        refs = np.array([r for r, _, _ in per_split])
        series = []
        for kind, R in series_keys:
            rows = []
            for m in single._m_grid(kind, R, pool.M):
                rng = np.random.default_rng([base.SEED, R, m, int(kind == "corrected"), 1])
                est = np.full(B, np.nan)
                for b in range(B):
                    held = halves(pool.R, b)[1]
                    masks = rng.choice(pool.M, size=m, replace=False)
                    seeds = rng.choice(held, size=R, replace=False)
                    est[b] = base._estimate(kind, g[masks], pool.margins[np.ix_(seeds, masks)])
                ok = np.isfinite(est)
                err = est[ok] - refs[ok]
                rows.append({"N": m * R, "m": m, "n_defined": int(ok.sum()),
                             "rmse": float(np.sqrt(np.mean(err ** 2))) if ok.any() else None,
                             "bias": float(err.mean()) if ok.any() else None,
                             "sd": float(err.std(ddof=1)) if ok.sum() > 1 else None,
                             "first": float(est[0]) if ok[0] else None})
            series.append({"kind": kind, "R": R, "rows": rows})
    print(f"[fig6] {bank:18s} {label:8s} reference {refs.mean():.4f} "
          f"(sd {refs.std(ddof=1):.4f}, {n_ref} of {pool.R} seeds)", flush=True)
    return {"stamp": {**bank_stamp(pool), "seed_mode": getattr(pool, "seed_mode", "nested")},
            "panel": {"bank": bank, "row": row, "label": label, "file": stem, "E": single.E,
                      "E_available": pt.R_tda, "n_ref": n_ref,
                      "truth_rho_inf_spearman": float(refs.mean()),
                      "reference": {"mean": float(refs.mean()), "sd": float(refs.std(ddof=1)),
                                    "first": float(refs[0])},
                      "naive_floors": {str(R): float(np.mean([f[str(R)] for _, f, _ in
                                                               per_split]))
                                       for R in floor_rs},
                      "naive_floors_first": per_split[0][1],
                      "lambda_med": float(np.mean([lam for _, _, lam in per_split])),
                      "series": series}}


def compute(banks_root):
    jobs = [(banks_root, bank, row) for bank, _ in single.COLUMNS for row in single.ROWS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
        out = list(ex.map(_panel, jobs))
    return {"banks": {o["panel"]["bank"]: o["stamp"] for o in out},
            "panels": [o["panel"] for o in out],
            "params": {"clip": base.CLIP, "m_fill": base.M_FILL, "m_min": base.M_MIN,
                       "seed": base.SEED, "B": B, "series": single.SERIES,
                       "split": "_heldout.halves(R, b): reference half A, estimators half B",
                       "E": f"first {single.E} replicas of every attributor",
                       "reference": "corrected composition on all masks from half A",
                       "floors": "from half A's reference and lambda, mean over splits; "
                                 "only for measured naive Rs",
                       "undefined": "NaN draws excluded; n_defined per point; drawn only if "
                                    "all B"}}


def _draw_panel(ax, p, xmax, column, ylim):
    from matplotlib.ticker import NullLocator
    for s in p["series"]:
        colour, ls, marker = single._STYLE[(s["kind"], s["R"])]
        rows = [r for r in s["rows"] if r["n_defined"] == B]
        ax.plot([r["N"] for r in rows], [r["rmse"] for r in rows], color=colour,
                linestyle=ls, marker=marker, markersize=3.8, linewidth=1.5, zorder=3)
        # dotted: the predicted RMSE floor this naive line flattens onto (fig1(a))
        if s["kind"] == "naive":
            ax.axhline(p["naive_floors"][str(s["R"])], color=colour, linestyle=":",
                       linewidth=1.1, alpha=0.5, zorder=1)
    single._x_axis(ax, xmax)
    ax.set_yscale("log")
    ticks = [t for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3) if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(*ylim)
    if column:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.text(0.97, 0.96, f"{p['label']}, $E{{=}}{p['E']}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#444444")


def plot(data, out_path):
    from matplotlib.lines import Line2D
    vals = [r["rmse"] for p in data["panels"] for s in p["series"] for r in s["rows"]
            if r["n_defined"] == B]
    vals += [f for p in data["panels"] for f in p["naive_floors"].values()]
    ylim = (0.8 * min(vals), 1.25 * max(vals))
    handles = single._series_handles(data) + [
        Line2D([], [], color="#888888", linestyle=":", linewidth=1.1,
               label="predicted naive RMSE floor")]
    single._grid(data, lambda ax, p, xmax, j: _draw_panel(ax, p, xmax, j, ylim),
                 r"RMSE to $\rho^{\mathtt{s}}$", handles, out_path)


if __name__ == "__main__":
    sys.exit(run("fig6", compute, plot))
