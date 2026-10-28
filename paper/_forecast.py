"""Figure 2 (c, f, i): the attributor-side forecast.  An E-member attributor's LDS is
attenuated by ``1/sqrt(1 + c/E)``, c the per-member noise-to-signal ratio, so c measured at
K members forecasts any larger E.  ``_fig2_panels`` anchors at the smallest grid K whose
clipped fraction clears ``CLIP_GATE``; ``draw_curves`` draws the panels.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corrected_lds import (
    pearson_to_spearman,
)

CLIP_GATE = 0.15          # the pre-flight check: anchor only where clipping is below this
K_GRID = [2, 3, 4, 5, 8, 15]
# Anchors overriding the gate: cifar10 TRAK-1, which the gate sends to K=15, is pinned at 5
# to show the failure the gate predicts; the gate's verdict stays in the plotdata.
ANCHOR_OVERRIDE = {("cifar10", "trak1"): 5}
MAX_GROUPS = 8            # disjoint groups averaged per size; caps the cost at large R_tda
# measured ensemble sizes; _fig2_panels adds each panel's E cap and each series' last E
SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 34, 50, 75, 100]
DATA_FRAC = 0.72          # of the axes width; the rest is the label gutter
# E axis drawn per setup, a display cap only, so TRAK-1's long tail does not crush the rest
E_SHOWN = {"cifar10": 20, "qnli": 20}
LABEL_GAP = 0.15          # the gutter labels' least vertical gap, in axes fraction

METHODS = {
    "cifar10": [("pool_tda_source_L3", "SOURCE", "#08306b", "D"),
                ("pool_tda_ekfac", "EK-FAC", "#3182bd", "s"),
                ("pool_tda_trak20", "TRAK-20", "#6aaed6", "v"),
                ("pool_tda_trak10", "TRAK-10", "#4a97cc", "^"),
                ("pool_tda_trak1", "TRAK-1", "#9ecae1", "o")],
    "qnli": [("pool_tda_ekfac", "EK-FAC", "#c44536", "s"),
             ("pool_tda_trak5", "TRAK-5", "#d4694f", "^"),
             ("pool_tda_trak1", "TRAK-1", "#e0937f", "v"),
             ("pool_tda_tracin", "TracIn", "#7d2b20", "D")],
}


def _inv_bridge(s):
    """Spearman -> Pearson: the inverse of the arcsin bridge that fig5 checks."""
    return 2.0 * np.sin(np.pi * np.clip(s, -1.0, 1.0) / 6.0)


def _forecast(rho_at_K, c, K, sizes):
    """The 1/E attenuation law anchored at K.  The law holds in Pearson space, so the
    Spearman anchor is bridged back, scaled and bridged forward again."""
    p = _inv_bridge(rho_at_K)
    return [float(pearson_to_spearman(p * np.sqrt((1.0 + c / K) / (1.0 + c / e))))
            for e in sizes]


def _label_of(bank, key):
    """The display name, looked up at draw time: the plotdata's ``label`` is never drawn."""
    return next(lab for f, lab, _, _ in METHODS[bank]
                if f.replace("pool_tda_", "") == key)


def _style_of(bank, key):
    return next((col, mk) for f, _, col, mk in METHODS[bank]
                if f.replace("pool_tda_", "") == key)


def _xticks(e_max):
    """A tick every ``nice`` members, plus E=1 — the left end of every curve."""
    step = next(v for v in (1, 2, 5, 10, 20, 25, 50, 100) if v >= e_max / 5.0)
    return [1, *range(step, int(e_max) + 1, step)]


def draw_curves(ax, data, bank, label_fs, label_x):
    ax.set_box_aspect(1)
    cap = E_SHOWN.get(bank, max(r["E"] for s in data["panels"][bank]["series"]
                                for r in s["rows"]))
    for s in data["panels"][bank]["series"]:
        c, mk = _style_of(bank, s["key"])
        # the forecast as a wide halo under the measurement: agreement reads as markers in it
        fe = [e for e in s["curve_E"] if e <= cap]
        ax.plot(fe, s["curve"][:len(fe)], color=c, linewidth=4.5, alpha=0.30,
                solid_capstyle="round", zorder=2)
        rows = [r for r in s["rows"] if r["E"] <= cap]
        ax.plot([r["E"] for r in rows], [r["mean"] for r in rows], color=c,
                marker=mk, markersize=3.4, linewidth=1.5, zorder=3)
        anchor = next(a for a in s["anchors"] if a["K"] == s["chosen_K"])
        ax.plot([s["chosen_K"]], [anchor["rho_at_K"]], marker="o", markersize=7,
                markerfacecolor="none", markeredgecolor=c, markeredgewidth=1.3, zorder=4)
    # linear E, a range per panel; the data ends at DATA_FRAC, leaving the label gutter
    ax.set_xlim(-0.02 * cap, cap / DATA_FRAC)
    ax.set_xticks(_xticks(cap))
    ax.set_ylabel("corrected LDS", fontsize=10)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    # push the labels apart: SOURCE and EK-FAC end within 0.001 of each other on cifar10
    tags = sorted(([r for r in s["rows"] if r["E"] <= cap][-1]["mean"],
                   _label_of(bank, s["key"]), _style_of(bank, s["key"])[0])
                  for s in data["panels"][bank]["series"])
    ylo, yhi = ax.get_ylim()
    pos = [(v - ylo) / (yhi - ylo) for v, _, _ in tags]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + LABEL_GAP)
    if pos and pos[-1] > 0.97:                   # slide the stack back inside, top ...
        pos = [v - (pos[-1] - 0.97) for v in pos]
    if pos and pos[0] < 0.03:                    # ... and bottom (TracIn sits on the axis)
        pos = [v + (0.03 - pos[0]) for v in pos]
    for y, (_, label, colour) in zip(pos, tags):
        ax.text(label_x, y, label, transform=ax.transAxes, fontsize=label_fs, color=colour,
                va="center", ha="left")


