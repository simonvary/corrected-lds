"""Talk figure 2: two seeds reconstruct the five-seed LDS for a fraction of the retrains.

Paper figure 2(b, e, h)'s TRAK panel per setting (CIFAR-10 / ResNet-9 TRAK-10; QNLI / BERT
and WikiText-2 / GPT-2 TRAK-5; one E=1 unit each): RMSE to the five-seed Spearman LDS
rho_5 against the budget ``N = M * R``, naive R=5 against reconstructed from R=2, the arrow
reading off the saving at RMSE 0.01.  The numbers are the paper's committed ones
(``--recompute`` re-reads paper/plotdata/fig2_panels.json; no banks), drawn by figure 2's
``draw_b`` on shared axes, two panels above one, in the ceilings figure's R=5 and R=2 colours.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper"))
import _rmse_corrected as base
import _shared
import _style
from _harness import PLOTDATA_DIR, run

SLIDES_DIR = os.path.dirname(os.path.abspath(__file__))
# (bank, figure 2's TRAK unit, its name, the setting)
PANELS = [("cifar10", "pool_tda_trak10", "TRAK-10", "CIFAR-10 / ResNet-9"),
          ("qnli", "pool_tda_trak5", "TRAK-5", "QNLI / BERT"),
          ("wikitext2_gpt2_bae", "pool_tda_trak5", "TRAK-5", "WikiText-2 / GPT-2")]
FIG_W, FIG_H = 5.6, 6.55                 # inches
PANEL_W, PANEL_H = 1.9, 1.75
CORNERS = [(1.15, 4.1), (3.35, 4.1), (2.25, 1.2)]    # each panel's lower left, in inches


def compute(banks_root):
    """Figure 2's committed (b, e, h) entries for each setting's TRAK unit."""
    with open(os.path.join(PLOTDATA_DIR, "fig2_panels.json")) as fh:
        fig2 = json.load(fh)
    b = {bank: [e for e in fig2["fig2"]["b"][bank] if e["attributor"] == unit]
         for bank, unit, _, _ in PANELS}
    return {"banks": {bank: fig2["banks"][bank] for bank in b}, "b": b,
            "params": {"derived_from": "paper/plotdata/fig2_panels.json, fig2.b",
                       "b": fig2["params"]["b"], "arrow_rmse": fig2["params"]["arrow_rmse"]}}


def plot(data, out_path):
    import matplotlib.pyplot as plt

    _style.use_panel()
    plt.rcParams["savefig.bbox"] = "standard"    # the page is the slide column
    # draw_b's tables, this process only: one method per row, the talk's colours
    naive, corrected = base._FAMILY["naive"], base._FAMILY["corrected"]
    _shared.STYLE_B.update({
        ("naive", 5): (naive["colors"][0], naive["linestyle"], "o", 2.0, r"naive, $R=5$"),
        ("recon", 2): (corrected["colors"][0], "-", "o", 2.0, r"reconstructed, $R=2$")})
    lims = []
    for bank, unit, name, _ in PANELS:
        _shared.B_METHODS[bank] = [(unit, name)]
        lims.append(_shared._b_limits(data, bank))
    limits = tuple((min(lim[i][0] for lim in lims), max(lim[i][1] for lim in lims))
                   for i in (0, 1))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    for (bank, _, name, setting), (x0, y0) in zip(PANELS, CORNERS):
        ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, PANEL_W / FIG_W, PANEL_H / FIG_H])
        _shared.draw_b(ax, data, bank, 0, True, x0 != CORNERS[1][0], limits=limits)
        ax.set_title(f"{setting}\n{name}, $E=1$", fontsize=_shared.B_TITLE_FS, pad=6)
    ax.set_xlabel(_shared._XLABEL_B)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), handlelength=2.6, fontsize=12)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig_slides2", compute, plot, root=SLIDES_DIR))
