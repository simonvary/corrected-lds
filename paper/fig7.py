"""Figure 6, single-draw form: what one run of each estimator returns, for every experiment
and attribution method, against a held-out reference (``fig6`` is the RMSE form; this
module also hosts the grid's shared pieces).

Columns are the experiments, rows the methods.  Each panel shows, per budget ``N = M * R``,
replicate 0 of ``fig6``'s 200 (m masks and R seeds from seed half B of split 0) against
split 0's reference (the corrected composition on all masks from half A) and, dotted, the
naive plateaus half A predicts; the plotdata is derived from ``fig6.json``.  Series:
fig1(a)'s where half B has R seeds, a plateau only for a measured naive R.  Methods are
figure 2's (TRAK-10 on CIFAR-10, TRAK-5 elsewhere), each an ensemble of its first E=10
replicas in its own units (TRAK-10 at E=10 is 100 models, TRAK-5 50).  WikiText-2's pool
is seeded per replica (bergson's protocol), so its seed noise is read by the centred
estimator, chosen from ``seed_mode`` as ``evaluate`` does.
"""

import contextlib
import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rmse_corrected as base
import _style
from _harness import PLOTDATA_DIR, run

from corrected_lds import load_pool, load_pool_tda, predictions_tda
from corrected_lds.evaluate import _seed_noise_fn

COLUMNS = [("cifar10", "CIFAR-10 / ResNet-9"), ("qnli", "QNLI / BERT"),
           ("wikitext2_gpt2_bae", "WikiText-2 / GPT-2")]
# third row: SOURCE on CIFAR-10, TracIn on QNLI and WikiText-2 (the one ensemble each bank has)
ROWS = ["EK-FAC", "TRAK", "SOURCE / TracIn"]
ATTRIBUTORS = {                  # bank -> row -> (pool_tda file, label)
    "cifar10": {"EK-FAC": ("pool_tda_ekfac", "EK-FAC"),
                "TRAK": ("pool_tda_trak10", "TRAK-10"),
                "SOURCE / TracIn": ("pool_tda_source_L3", "SOURCE")},
    "qnli": {"EK-FAC": ("pool_tda_ekfac", "EK-FAC"),
             "TRAK": ("pool_tda_trak5", "TRAK-5"),
             "SOURCE / TracIn": ("pool_tda_tracin", "TracIn")},
    "wikitext2_gpt2_bae": {"EK-FAC": ("pool_tda_ekfac", "EK-FAC"),
                           "TRAK": ("pool_tda_trak5", "TRAK-5"),
                           "SOURCE / TracIn": ("pool_tda_tracin", "TracIn")},
}
SERIES = [("naive", 5), ("naive", 10), ("naive", 20), ("naive", 40),
          ("corrected", 2), ("corrected", 5)]          # fig1(a)'s shown series
E = 10                           # every attributor: its first E replicas


def _series(pool):
    """The SERIES this pool has the seeds for."""
    return [(kind, R) for kind, R in SERIES if R <= pool.R]


@contextlib.contextmanager
def _seed_noise(s2_fn):
    """fig1(a)'s estimator with the pool's own seed-noise estimator (figure 2's patch)."""
    with mock.patch.object(base, "seed_noise_variance_stacked", s2_fn):
        yield


def _setup(banks_root, bank, stem):
    """(pool, pool_tda, predictions of its first E replicas, seed-noise estimator)."""
    pool = load_pool(os.path.join(banks_root, bank, "pool.npz"))
    pt = load_pool_tda(os.path.join(banks_root, bank, f"{stem}.npz"))
    return pool, pt, predictions_tda(pool, pt, list(range(E))), _seed_noise_fn(pool)


def _m_grid(kind, R, M):
    """Figure 1(a)'s mask grid, cut to this pool and closed by its M (as fig2's)."""
    return [m for m in sorted({base.M_MIN[kind], *base.M_FILL, M})
            if base.M_MIN[kind] <= m <= M and m * R >= 9]


def compute(banks_root):
    """Replicate 0 of ``fig6``: its first draws, split 0's reference and plateaus."""
    with open(os.path.join(PLOTDATA_DIR, "fig6.json")) as fh:
        rmse = json.load(fh)
    panels = []
    for p in rmse["panels"]:
        panels.append({**{k: p[k] for k in ("bank", "row", "label", "file", "E",
                                             "E_available", "n_ref", "lambda_med")},
                       "truth_rho_inf_spearman": p["reference"]["first"],
                       "naive_floors": p["naive_floors_first"],
                       "series": [{"kind": s["kind"], "R": s["R"],
                                   "rows": [{"N": r["N"], "m": r["m"], "estimate": r["first"]}
                                            for r in s["rows"]]}
                                  for s in p["series"]]})
    return {"banks": rmse["banks"], "panels": panels,
            "params": {**rmse["params"], "derived_from": "fig6.json",
                       "draw": "replicate 0 (split 0): estimates from half B, reference "
                               "and plateaus from half A"}}


_STYLE = {}                      # (kind, R) -> fig1(a)'s colour and marker for that line
for _kind in ("naive", "corrected"):
    _Rs = [R for k, R in base.ESTIMATORS if k == _kind]
    for _i, _R in enumerate(_Rs):
        _STYLE[(_kind, _R)] = (base._FAMILY[_kind]["colors"][_i],
                               base._FAMILY[_kind]["linestyle"], base._MARKERS[_i])


def _x_axis(ax, xmax):
    """Log budget axis, decade ticks up to the column's largest N (10^4 on CIFAR-10)."""
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ticks = [t for t in (10, 100, 1000, 10000) if t <= xmax]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(8, xmax)


def _draw_panel(ax, p, xmax):
    from matplotlib.ticker import MaxNLocator
    truth = p["truth_rho_inf_spearman"]
    ax.axhline(truth, color="#333333", linewidth=1.1, zorder=2)
    for s in p["series"]:
        colour, ls, marker = _STYLE[(s["kind"], s["R"])]
        rows = [r for r in s["rows"] if r["estimate"] is not None]
        ax.plot([r["N"] for r in rows], [r["estimate"] for r in rows],
                color=colour, linestyle=ls, marker=marker, markersize=3.8, linewidth=1.5,
                zorder=3)
        if s["kind"] == "naive":
            ax.axhline(truth - p["naive_floors"][str(s["R"])], color=colour, linestyle=":",
                       linewidth=1.1, alpha=0.5, zorder=1)
    _x_axis(ax, xmax)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.text(0.97, 0.04, f"{p['label']}, $E{{=}}{p['E']}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#444444")


def _series_handles(data):
    """Legend handles for every series measured in some panel."""
    from matplotlib.lines import Line2D
    measured = {(s["kind"], s["R"]) for p in data["panels"] for s in p["series"]}
    return [Line2D([], [], color=_STYLE[(k, R)][0], linestyle=_STYLE[(k, R)][1],
                   marker=_STYLE[(k, R)][2], markersize=4.5, linewidth=1.7,
                   label=f"{k}, $R{{=}}{R}$") for k, R in SERIES if (k, R) in measured]


def _grid(data, draw_panel, ylabel, handles, out_path):
    """Both figure-6 forms' 3x3 layout; ``draw_panel(ax, panel, xmax, column)`` fills one cell."""
    import matplotlib.pyplot as plt
    _style.use_panel()
    by = {(p["bank"], p["row"]): p for p in data["panels"]}
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(10.6, 9.0))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.943, bottom=0.146, wspace=0.32,
                        hspace=0.14)
    for j, (bank, title) in enumerate(COLUMNS):
        xmax = 1.3 * max(r["N"] for p in data["panels"] if p["bank"] == bank
                         for s in p["series"] for r in s["rows"])
        for i, row in enumerate(ROWS):
            draw_panel(axes[i, j], by[(bank, row)], xmax, j)
            if i < len(ROWS) - 1:
                axes[i, j].tick_params(labelbottom=False)
        axes[-1, j].set_xlabel(r"total retrains  $N = M \cdot R$")
        axes[0, j].set_title(title, fontsize=12, pad=8)
    for i, row in enumerate(ROWS):
        axes[i, 0].set_ylabel(ylabel)
        pos = axes[i, 0].get_position()
        fig.text(0.012, pos.y0 + pos.height / 2, row, rotation=90, ha="left", va="center",
                 fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2, frameon=False)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


def plot(data, out_path):
    from matplotlib.lines import Line2D
    handles = _series_handles(data) + [
        Line2D([], [], color="#333333", linewidth=1.1,
               label=r"reference $\rho^{\mathtt{s}}$ (held-out seeds)"),
        Line2D([], [], color="#888888", linestyle=":", linewidth=1.1,
               label="predicted naive plateau")]
    _grid(data, lambda ax, p, xmax, _: _draw_panel(ax, p, xmax),
          r"one draw's  $\rho^{\mathtt{s}}$", handles, out_path)


if __name__ == "__main__":
    sys.exit(run("fig7", compute, plot))
