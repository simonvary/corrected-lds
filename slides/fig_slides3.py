"""Talk figure 3: two attributor members forecast the LDS of the whole ensemble.

Paper figure 2(c): CIFAR-10 / ResNet-9's corrected Spearman LDS against attributor ensemble
size E for TRAK-20, TRAK-10, TRAK-1, SOURCE and EK-FAC, measured (line), forecast by the
1/E law from K members (halo) and its anchor (ring; K=2, TRAK-1 K=5).  The numbers are the
paper's committed ones (``--recompute`` re-reads paper/plotdata/fig2_panels.json; no banks),
drawn by figure 2's ``draw_curves`` with one hue per method and the axis from zero.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper"))
import _forecast as forecast
import _style
from _harness import PLOTDATA_DIR, run

SLIDES_DIR = os.path.dirname(os.path.abspath(__file__))
BANK = "cifar10"
# one hue per method: the paper's shades of one blue blur at slide size
COLORS = {"TRAK-20": "#1b9e77", "TRAK-10": "#d95f02", "TRAK-1": "#e6ab02",
          "SOURCE": "#1f78b4", "EK-FAC": "#e7298a"}
FIG_W, FIG_H = 3.6, 3.65                 # inches
AXES = [0.17, 0.255, 0.77, 0.715]        # left, bottom, width, height; drawn square inside
LABEL_GAP = 0.06                         # the gutter labels' least gap, for 11 pt labels


def compute(banks_root):
    """Figure 2(c)'s committed CIFAR-10 panel."""
    with open(os.path.join(PLOTDATA_DIR, "fig2_panels.json")) as fh:
        fig2 = json.load(fh)
    return {"banks": {BANK: fig2["banks"][BANK]},
            "panels": {BANK: fig2["fig3"]["panels"][BANK]},
            "params": {"derived_from": "paper/plotdata/fig2_panels.json, fig3.panels.cifar10",
                       "c": fig2["params"]["c"]}}


def plot(data, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style.use_panel()
    plt.rcParams["savefig.bbox"] = "standard"    # the page is the slide's
    # draw_curves' tables, this process only
    forecast.METHODS[BANK] = [(f, lab, COLORS[lab], mk) for f, lab, _, mk in forecast.METHODS[BANK]]
    forecast.LABEL_GAP = LABEL_GAP
    cap = forecast.E_SHOWN[BANK]
    top = max(max(r["mean"] for r in s["rows"] if r["E"] <= cap) for s in
              data["panels"][BANK]["series"])
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES)
    ax.set_ylim(0.0, 1.05 * top)                 # first: the gutter labels are placed in it
    forecast.draw_curves(ax, data, BANK, label_fs=11, label_x=0.735)
    ax.set_ylabel(r"corrected LDS $\rho^s$", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_xlabel(r"ensemble size  $E$")
    grey = "#555555"
    keys = [Line2D([], [], color=grey, marker="o", markersize=3.4, linewidth=1.5,
                   label="measured"),
            Line2D([], [], color=grey, linewidth=4.5, alpha=0.30, label="forecast"),
            Line2D([], [], color=grey, marker="o", markersize=7, markerfacecolor="none",
                   markeredgewidth=1.3, linestyle="none", label="anchor")]
    fig.legend(handles=keys, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig_slides3", compute, plot, root=SLIDES_DIR))
