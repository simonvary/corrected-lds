"""Figure 1 with panel (a) as the estimated Spearman LDS instead of its RMSE.

Panel (a) plots, at each budget ``N = M * R``, the mean of fig1(a)'s B=200 estimates (line)
and +-1 sd (band) on a linear LDS axis; solid: the held-out reference (the corrected
composition on all masks from seed half A, mean over the splits); dotted: the values the
naive lines are predicted to stall on (the attenuation law).  Series: naive R=5/10/20/40
and corrected R=2.  All but R=40 are ``_rmse_heldout``'s draws: mean = reference mean +
bias, band = the error's sd (the estimate's sd to within the reference's spread across
splits).  R=40 uses every seed and cannot be held out, but estimates need no reference:
its draws, centre and plateau are ``_rmse_corrected``'s full-pool ones.  Derived from
committed plotdata (``fig1_panels.json``, ``rmse_corrected.json``) and drawn by
``_fig1_panels.plot`` with (a)'s body swapped, so the layout cannot drift.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fig1_panels as panels
import _rmse_corrected as base
from _harness import PLOTDATA_DIR, run

SERIES = [("naive", 5), ("naive", 10), ("naive", 20), ("naive", 40),
          ("corrected", 2)]
FULL_POOL = {("naive", 40)}      # read from _rmse_corrected: they need every seed
_BAND_ALPHA = 0.16
LETTER_X_A = -0.33               # (a)'s letter, further left: its widest tick label is at the top


def compute(banks_root):
    with open(os.path.join(PLOTDATA_DIR, "fig1_panels.json")) as fh:
        data = json.load(fh)
    with open(os.path.join(PLOTDATA_DIR, "rmse_corrected.json")) as fh:
        full = json.load(fh)
    del data["schema"], data["figure"]
    held = data["a"]
    assert full["banks"] == held["banks"], "_rmse_corrected and fig1(a) differ in bank"
    assert full["truth_rho_inf_spearman"] == held["truth_rho_inf_spearman"], \
        "_rmse_corrected and fig1(a) disagree on the 40-seed value — recompute both"
    truth, ref = full["truth_rho_inf_spearman"], held["reference"]["mean"]
    series, plateaus = [], {}
    for kind, R in SERIES:
        src, centre = (full, truth) if (kind, R) in FULL_POOL else (held, ref)
        s = next(s for s in src["series"] if s["kind"] == kind and s["R"] == R)
        series.append({"kind": kind, "R": R,
                       "source": "rmse_corrected" if src is full else "rmse_heldout",
                       "rows": [{"N": r["N"], "m": r["m"], "mean": centre + r["bias"],
                                 "sd": r["sd"]} for r in s["rows"]]})
        if kind == "naive":
            plateaus[str(R)] = centre - src["naive_floors"][str(R)]
    data["a"] = {"banks": held["banks"], "reference": held["reference"],
                 "truth_rho_inf_spearman": truth, "plateaus": plateaus,
                 "params": {**held["params"],
                            "view": "mean estimate +- 1 sd over the B draws",
                            "full_pool": "naive R=40 from _rmse_corrected (all seeds)"},
                 "series": series}
    data["params"] = {**data["params"],
                      "derived_from": ["fig1_panels.json", "rmse_corrected.json"],
                      "a": "mean +- 1 sd of fig1(a)'s draws in LDS units; naive R=40 added"}
    return data


def draw(ax, data):
    """Panel (a) as mean +- 1 sd, fig1(a)'s colours and markers per (kind, R)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, NullLocator
    ax.set_box_aspect(1)
    ref = data["reference"]["mean"]
    ax.axhline(ref, color="#333333", linewidth=1.1, zorder=2)
    handles, labels = [], []
    for s in data["series"]:
        fam = base._FAMILY[s["kind"]]
        i = [R for kind, R in base.ESTIMATORS if kind == s["kind"]].index(s["R"])
        colour = fam["colors"][i]
        style = {"color": colour, "linestyle": fam["linestyle"], "marker": base._MARKERS[i],
                 "markersize": 4.5, "linewidth": 1.7}
        n = [r["N"] for r in s["rows"]]
        mean = [r["mean"] for r in s["rows"]]
        ax.fill_between(n, [m - r["sd"] for m, r in zip(mean, s["rows"])],
                        [m + r["sd"] for m, r in zip(mean, s["rows"])],
                        color=colour, alpha=_BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(n, mean, zorder=3, **style)
        # dotted: the predicted value this naive line stalls on
        if s["kind"] == "naive":
            ax.axhline(data["plateaus"][str(s["R"])], color=colour, linestyle=":",
                       linewidth=1.1, alpha=0.5, zorder=1)
        handles.append((Patch(color=colour, alpha=_BAND_ALPHA, linewidth=0),
                        Line2D([], [], **style)))
        labels.append(f"{fam['label']}, $R{{=}}{s['R']}$")
    # the reference goes in the key: an in-panel label would hit the corrected line's end
    handles.append(Line2D([], [], color="#333333", linewidth=1.1))
    labels.append(r"reference $\rho^{\mathtt{s}}$")
    ax.set_xscale("log")
    ax.set_xticks(base._XTICKS, [str(t) for t in base._XTICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(8, 14500)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel(r"total retrains  $N = M \cdot R$")
    ax.set_ylabel(r"estimated LDS $\rho^{\mathtt{s}}$")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.6, columnspacing=0.9, borderaxespad=0.0, handletextpad=0.5)


def plot(data, out_path):
    panels.plot(data, out_path, draw_a=draw, letter_x_a=LETTER_X_A)


if __name__ == "__main__":
    run("fig1", compute, plot)
