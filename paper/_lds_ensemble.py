"""Figure 1 panel (b): corrected Spearman LDS against attributor ensemble size E.

The series table, size grid and artifact lookup ``_fig1_panels`` computes (b) from, and
``draw``; each series reaches only as far as its artifact's replicas.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rmse_corrected as base

from corrected_lds import (
    pearson_to_spearman,
)

_E_MAX = 30                      # the panel's axis range (measured rows filtered too)


# The grid follows the artifact's R_tda, so an appended pool_tda is never truncated.  Unit
# series: every integer E; member series: E=1, even E to the cap and a stored 50/100 tail.
def _sizes(kind, R):
    if kind == "unit":
        return list(range(1, R + 1))
    return [e for e in (1, *range(2, _E_MAX + 1, 2), 50, 100) if e <= R]


# Each series lists the artifacts it accepts, most extended first: an append round (merged
# by `corrected-lds merge-pool-tda`) sits beside the untouched released file.
SERIES = [
    # TRAK is one family indexed by e_features; the name says which member.
    ("trak10", ("pool_tda_trak10.npz",), "unit"),
    ("trak20", ("pool_tda_trak20.npz",), "unit"),
    ("trak1", ("pool_tda_trak1.npz",), "member"),
    # EK-FAC: the full-set run (damping 1e-8, Bae et al.'s IF protocol).  No fallback, so a
    # missing artifact drops the series instead of silently swapping the estimator.
    ("ekfac", ("pool_tda_ekfac.npz",), "member"),
    ("source_L3", ("pool_tda_source_L3.npz",
                   "pool_tda_source_L3_e10_partial.npz"), "member"),
]


def _artifact(bank, names):
    """The first present candidate, or None, so an optional series simply drops out."""
    for name in names:
        path = os.path.join(bank, name)
        if os.path.exists(path):
            return path
    return None


# trak20 is computed and stored but not drawn
_SHOW = [("trak10", r"TRAK, $E_{\rm feat}{=}10$", "#1b6ca8", "o"),
         ("trak1", r"TRAK, $E_{\rm feat}{=}1$", "#a8890a", "s"),
         ("ekfac", "EK-FAC", "#c44536", "^"),
         ("source_L3", r"SOURCE, $L{=}3$", "#2e7d32", "D")]
_YTICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_XTICKS = [1, 10, 20, 30]
XLABEL = r"ensemble size $E$"


def draw(ax, data):
    """Render figure 1(b) into ``ax``."""
    ax.set_box_aspect(1)
    handles = []
    have = {x["key"]: x for x in data["series"]}
    for key, label, color, marker in _SHOW:
        s = have.get(key)
        if s is None:                # series absent from this plotdata
            continue
        pred = s["forecast"]
        K = pred["K"]
        kk = np.linspace(1, _E_MAX, 300)
        pK = base._spearman_to_pearson(pred["rho_sp_at_K"])
        rho_k = pearson_to_spearman(
            pK * np.sqrt((1 + pred["c"] / K) / (1 + pred["c"] / kk)))
        ax.plot(kk, rho_k, color=color, linewidth=1.4, linestyle=(0, (4, 2.4)),
                alpha=0.75, zorder=2)
        # ring the E=K anchor the forecast is fitted from
        ax.plot([K], [pred["rho_sp_at_K"]], marker="o", markerfacecolor="none",
                markeredgecolor=color, markersize=10, markeredgewidth=1.4,
                linestyle="none", zorder=4)
        rows = [r for r in s["rows"] if r["E"] <= _E_MAX]
        e = [r["E"] for r in rows]
        y = [r["mean"] for r in rows]
        sd = [r["sd"] or 0.0 for r in rows]
        # markers only: the dashed forecast is this panel's line
        handles.append(ax.errorbar(
            e, y, yerr=sd, color=color, marker=marker, markersize=5,
            linestyle="none", elinewidth=1.0, capsize=2, zorder=3, label=label))
    handles += ax.plot([], [], color="#777777", linewidth=1.4,
                       linestyle=(0, (4, 2.4)), label="predicted from circled $E$")
    ax.set_xticks(_XTICKS)
    ax.set_xlim(-0.3, _E_MAX + 1)
    ax.set_yticks(_YTICKS)
    ax.set_ylim(0.0, 0.71)
    # E counts replicas (Eq. 15 units), not models: the series cost E, 10E or 20E models at one x
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"LDS $\rho^{\mathtt{s}}$")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.legend(handles, [h.get_label() for h in handles], ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.8, columnspacing=0.9, borderaxespad=0.0,
              handletextpad=0.5)


