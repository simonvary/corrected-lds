"""Shared figure code: the three settings' bank and method tables, seeds and grids (figures
2 and 5; ``_fig2_panels`` puts each row's TRAK unit in TRAK-1's slot), the rho_5 estimator
``_estimate_b``, column-wise correlation helpers, and figure 2's (b, e, h) drawer ``draw_b``
and column keys ``_column_legend``.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corrected_lds import (
    battery,
    pearson_to_spearman,
    reconstruct_rho_R,
    seed_noise_variance_stacked,
)

CLIP = 2.0
SEED = 0
N_SUB = 50                       # seed-subsample replicates per (series, R) in panel (a)
TARGET_R = 5                     # panel (b) estimates the rho_5 convention
# the RMSE every (b, e, h) speedup arrow is read at: one level, so xN compares across rows
ARROW_RMSE = 0.01
B_SUB = 300                      # subsample replicates per (series, m) in panel (b)
N_TRUTH = 40                     # random TARGET_R-seed subsets averaged into the rho_5 truth
# m <= M/2, never the full pool; m=100 on every bank so the rows share one budget axis
M_GRID = {"cifar10": [5, 10, 20, 40, 60, 100], "qnli": [5, 10, 20, 40, 60, 100],
          "wikitext2_gpt2_bae": [5, 10, 20, 40, 60, 100]}
# one-row column keys with short handles; type in points at fig2's scale (_fig2_panels.TYPE)
LEGEND_FS = 12
LEGEND_HL = 0.6
B_TICK_FS = 12.0                 # (b)'s tick labels
B_TITLE_FS = 13.5                # (b)'s method names
B_ARROW_FS = 12.0                # (b)'s speedup tags

BANK_COLOR = {"cifar10": "#1b6ca8", "qnli": "#c44536", "wikitext2_gpt2_bae": "#2f7d4a"}
BANK_LABEL = {"cifar10": "CIFAR-10 / ResNet-9", "qnli": "QNLI / BERT",
              "wikitext2_gpt2_bae": "WikiText-2 / GPT-2"}
# panels (a/d): the methods at E=1, one replica each, the only comparable footing (full
# ensembles confound the method with the model budget).  A TRAK E_f10/E_f20/E_f5 replica
# bakes in e_features models (the mask projection loses Eq. 15's per-example pieces).
E1_REPLICAS = [0]                # E=1 = the first replica (a prefix of the ensemble)
A_METHODS = {
    "cifar10": [("pool_tda_source_L3", "SOURCE", "#08306b", "D"),
                ("pool_tda_ekfac", "EK-FAC", "#3182bd", "s"),
                ("pool_tda_trak1", "TRAK", "#6aaed6", "^")],
    "qnli": [("pool_tda_ekfac", "EK-FAC", "#c44536", "s"),
             ("pool_tda_trak1", "TRAK", "#e0937f", "^"),
             ("pool_tda_tracin", "TracIn", "#7d2b20", "v")],
    # greens, darkest = highest LDS
    "wikitext2_gpt2_bae": [("pool_tda_trak1", "TRAK", "#238b45", "^"),
                           ("pool_tda_ekfac", "EK-FAC", "#00441b", "s"),
                           ("pool_tda_tracin", "TracIn", "#74c476", "D")],
}
A_R_GRID = {"cifar10": [1, 2, 3, 5, 10, 20, 40], "qnli": [1, 2, 3, 5, 10],
            "wikitext2_gpt2_bae": [1, 2, 3, 5, 10]}
# panels (b/e): the two routes to rho_5, one small panel per method, at E=1 as (a/d)
SERIES_B = [("naive", 5), ("recon", 2)]
STYLE_B = {("naive", 5): ("#c44536", "--", "o", 1.7, r"naive Spearman, $R{=}5$"),
           ("recon", 2): ("#1b6ca8", "-", "s", 1.8, r"reconstructed, $R{=}2$")}
B_METHODS = {
    "cifar10": [("pool_tda_ekfac", "EK-FAC"), ("pool_tda_trak1", "TRAK"),
                ("pool_tda_source_L3", "SOURCE")],
    "qnli": [("pool_tda_ekfac", "EK-FAC"), ("pool_tda_trak1", "TRAK"),
             ("pool_tda_tracin", "TracIn")],
    "wikitext2_gpt2_bae": [("pool_tda_ekfac", "EK-FAC"), ("pool_tda_trak1", "TRAK"),
                           ("pool_tda_tracin", "TracIn")],
}


def _colcorr(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)


def _spearman(a, b, rank_a=None):
    ra = stats.rankdata(a, axis=0) if rank_a is None else rank_a
    return float(battery(_colcorr(ra, stats.rankdata(b, axis=0))))


def _estimate_b(kind, r_seeds, g_sub, margins_sub, s2_fn=seed_noise_variance_stacked):
    """One rho_5 estimate; ``s2_fn`` is the pool's seed-noise estimator (``_seed_noise_fn``)."""
    if kind == "naive":
        return _spearman(g_sub, margins_sub.mean(axis=0))
    rho, _ = reconstruct_rho_R(g_sub, margins_sub.mean(axis=0), s2_fn(margins_sub),
                               k=r_seeds, R=TARGET_R, clip=CLIP)
    return float(battery(pearson_to_spearman(rho)))


# -- panels ----------------------------------------------------------------- #


def _b_limits(data, bank):
    """One RMSE and one budget scale per row: shared by its methods, separate between rows."""
    pts = [p for b in data["b"][bank] for ser in b["series"] for p in ser["points"]]
    rmse = [p["rmse"] for p in pts]
    n = [p["N"] for p in pts]
    # the floor is left out: a decade below the RMSEs (cifar10, E=1), it would flatten them
    return (0.82 * min(rmse), 1.55 * max(rmse)), (0.84 * min(n), 1.18 * max(n))


def draw_b(ax, data, bank, idx, show_x, show_ylabel, limits=None):
    """One method's budget frontier: two routes to rho_5, the arrow reads off the saving.
    ``limits``: (ylim, xlim), by default the row's (``_b_limits``)."""
    from matplotlib.ticker import NullLocator
    b = data["b"][bank][idx]
    label = B_METHODS[bank][idx][1]
    ylim, xlim = limits or _b_limits(data, bank)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=B_TICK_FS)
    every_n = [p["N"] for bb in data["b"][bank] for ser in bb["series"] for p in ser["points"]]
    ticks = [t for t in (25, 100, 500) if min(every_n) * 0.9 <= t <= max(every_n) * 1.1]
    ax.set_xticks(ticks, [str(t) for t in ticks] if show_x else [])
    lut = {(s["kind"], s["R"]): s for s in b["series"]}
    for key in SERIES_B:
        color, ls, mk, lw, _ = STYLE_B[key]
        pts = lut[key]["points"]
        ax.plot([p["N"] for p in pts], [p["rmse"] for p in pts], color=color, linestyle=ls,
                marker=mk, markersize=3.4, linewidth=lw, label=STYLE_B[key][4], zorder=3)
    # The reconstruction route's bias floor (plotdata "floor_bias") is not drawn: it fits the
    # axis for only some methods, where it would read as a panel property.  Arrow budgets come
    # off a log-log fit per series; two-point interpolation jitters enough to invent differences.
    lvl = ARROW_RMSE
    spans = {k: (min(p["rmse"] for p in lut[k]["points"]),
                 max(p["rmse"] for p in lut[k]["points"])) for k in SERIES_B}
    if not all(lo <= lvl <= hi for lo, hi in spans.values()):
        # never extrapolate: if a route stops short of the level, use the lowest RMSE both reach
        common = (max(lo for lo, _ in spans.values()), min(hi for _, hi in spans.values()))
        print(f"[fig2] {bank}/{label}: RMSE {lvl} outside the measured spans "
              f"{spans} -- arrow at {common[0]:.4f}, the lowest both reach", flush=True)
        lvl = common[0] if common[0] <= common[1] else None
    n_of, measured = {}, lvl is not None
    for k in SERIES_B if measured else ():
        pts = lut[k]["points"]
        slope, intercept = np.polyfit(np.log([p["N"] for p in pts]),
                                      np.log([p["rmse"] for p in pts]), 1)
        n_of[k] = float(np.exp((np.log(lvl) - intercept) / slope))
    if measured:
        ax.annotate("", xy=(n_of[("naive", 5)], lvl), xytext=(n_of[("recon", 2)], lvl),
                    arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.0},
                    zorder=5)
        tag = rf"$\times${n_of[('naive', 5)] / n_of[('recon', 2)]:.1f}"
        txt = ax.annotate(tag, (n_of[("naive", 5)], lvl), textcoords="offset points",
                          xytext=(3, 2), va="bottom", fontsize=B_ARROW_FS, zorder=5)
        # Tag above the arrow right of its naive end, else below it left of the reconstructed
        # end (both clear of the curves); if neither fits, it overhangs right, not over the y axis
        renderer = ax.figure.canvas.get_renderer()
        box = ax.get_window_extent(renderer)
        over = txt.get_window_extent(renderer).x1 - box.x1
        if over > 0:
            alt = ax.annotate(tag, (n_of[("recon", 2)], lvl), textcoords="offset points",
                              xytext=(-3, -2), ha="right", va="top", fontsize=B_ARROW_FS,
                              zorder=5)
            (alt if alt.get_window_extent(renderer).x0 < box.x0 else txt).remove()
    ax.set_title(label, fontsize=B_TITLE_FS, pad=4)
    # 1-2-5 ticks (decades alone can leave one label on a tight axis), thinned if crowded
    yt = [m * 10.0 ** e for e in range(-6, 2) for m in (1, 2, 5)
          if ylim[0] < m * 10.0 ** e < ylim[1]]
    if len(yt) > 6:
        yt = [v for v in yt if abs(np.log10(v) - round(np.log10(v))) < 1e-9]
    ax.set_yticks(yt, [f"{v:g}" for v in yt])
    ax.grid(axis="both", which="major", alpha=0.15, linewidth=0.6)
    if show_ylabel:
        ax.set_ylabel(rf"RMSE to $\rho_{{{b['target_R']}}}$")
    else:
        ax.tick_params(axis="y", labelleft=False)


_XLABEL_B = r"total retrains  $N = M \cdot R$"


def _column_legend(fig, handles, ncol, x0, width, tag, y=0.012, height=0.055):
    """One key under one column, pinned to its width.  Its natural width is measured without
    ``mode="expand"``, under which the extent is the bbox and overlaps go unreported."""
    style = {"handlelength": LEGEND_HL, "handletextpad": 0.3, "columnspacing": 0.3,
             "fontsize": LEGEND_FS}
    probe = fig.legend(handles=handles, ncol=ncol, frameon=False, **style)
    fig.canvas.draw()
    natural = probe.get_window_extent().transformed(fig.transFigure.inverted()).width
    probe.remove()
    if natural > width:
        print(f"[fig2] WARNING: the ({tag}) key needs {100 * natural / width:.0f}% of its "
              f"column — its labels will overlap", flush=True)
    fig.legend(handles=handles, ncol=ncol, frameon=False, mode="expand", borderaxespad=0.0,
               bbox_to_anchor=(x0, y, width, height), bbox_transform=fig.transFigure,
               **style)


