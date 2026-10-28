"""Figure 1's three-panel row, every reference held out (the paper's default).

(a) estimator RMSE vs budget against a held-out reference (``_rmse_heldout``: disjoint seed
halves, so naive R runs to 20).  (b) LDS vs attributor ensemble size: the forecast is
anchored on the last K replicas read out with seed half A of split 0 (``_heldout.halves``),
the measured curve on the replicas before them read out with half B, so it stops K short
of each artifact and the ringed anchor is not one of its points.  (c) finite-mask Spearman
debiasing for the released TRAK1 E=100 bank (subsets drawn from the bank the reference uses).

(a) and (c) come from ``rmse_heldout.json`` and ``spearman_debias.json``; (b) is computed
here (``--recompute --banks-root <dir>``: cifar10/pool.npz and (b)'s pool_tda files, a few
minutes on CPU).  ``plot`` lays the row out for ``fig1``, which supplies (a)'s body.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lds_ensemble as panel_b
import _rmse_corrected as panel_a
import _spearman_debias as panel_c
import _style
from _harness import PLOTDATA_DIR, bank_stamp, run
from _heldout import halves

from corrected_lds import (
    battery_c,
    load_pool,
    load_pool_tda,
    member_noise_ratio,
    predictions_tda,
)

# Type, in points: printed at ~0.5x (\textwidth from a ~10.8 in box), it reads 6-8 pt
TYPE = {"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 12}
LETTER_FS = 16
LETTER_X = -0.295                # the corner letters, in axes fraction left of the y axis
LEGEND_Y = -0.25                 # each key's top, in axes fraction below its panel
LEGEND_X = (0.44, 0.57, 0.5)     # key centres: (a)'s and (b)'s outgrow their panels, so lean apart


def _panel_b(banks_root):
    """(b), held out: anchors on the last K replicas (half A), ensembles before them (half B)."""
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    a_seeds, b_seeds = halves(pool.R, 0)
    m_a, m_b = pool.margins[a_seeds], pool.margins[b_seeds]
    series = []
    for key, names, kind in panel_b.SERIES:
        path = panel_b._artifact(bank, names)
        if path is None:
            continue
        ptda = load_pool_tda(path)
        K = min(5 if key == "trak1" else 2, ptda.R_tda - 1)   # fig1(b)'s anchor sizes
        anchors = list(range(ptda.R_tda - K, ptda.R_tda))     # the last K replicas
        e_max = ptda.R_tda - K                                # measured never touch them
        rows = []
        for s in panel_b._sizes(kind, e_max):
            v = panel_a._estimate("corrected", predictions_tda(pool, ptda, replicas=range(s)),
                                  m_b)
            rows.append({"E": s, "n_groups": 1, "values": [v], "mean": v, "sd": None})
        c_q, diag = member_noise_ratio(ptda.g[anchors])
        rho_spK = panel_a._estimate("corrected", predictions_tda(pool, ptda, replicas=anchors),
                                    m_a)
        series.append({"key": key, "e_features": int(ptda.e_features),
                       "artifact": os.path.basename(path), "rows": rows,
                       "measured_replicas": e_max,
                       "forecast": {"K": K, "c": float(battery_c(c_q)),
                                    "clipped_frac": diag["clipped_frac"],
                                    "rho_sp_at_K": rho_spK, "R_tda": int(ptda.R_tda),
                                    "anchor_replicas": anchors}})
        print(f"[fig1 b] {key}: anchor replicas {anchors[0]}..{anchors[-1]} "
              f"(half A) {rho_spK:.4f}; measured E<={e_max} (half B) ends "
              f"{rows[-1]['mean']:.4f}", flush=True)
    return {"banks": {"cifar10": bank_stamp(pool)}, "trak20_point": None,
            "params": {"clip": panel_a.CLIP, "E_max": panel_b._E_MAX,
                       "split": "seed halves of _heldout.halves(R, 0): anchor A, measured B",
                       "anchors": "the LAST K replicas; measured ensembles = prefixes of "
                                  "the replicas before them",
                       "estimator": "corrected Spearman composition"},
            "series": series}


def compute(banks_root):
    with open(os.path.join(PLOTDATA_DIR, "rmse_heldout.json")) as fh:
        a = json.load(fh)
    with open(os.path.join(PLOTDATA_DIR, "spearman_debias.json")) as fh:
        c = json.load(fh)
    return {"a": a, "b": _panel_b(banks_root), "c": c,
            "params": {"derived_from": ["rmse_heldout.json", "cifar10 bank (b)",
                                        "spearman_debias.json"],
                       "b": "held-out anchors (last K replicas, seed half A) vs measured "
                            "(earlier replicas, seed half B)",
                       "c": "finite-mask Spearman U-statistic, TRAK1 E=100"}}


def plot(data, out_path, draw_a, letter_x_a):
    """``fig1`` supplies panel (a): ``draw_a`` draws its body, ``letter_x_a`` places its letter."""
    import matplotlib.pyplot as plt
    _style.use_panel()
    plt.rcParams.update(TYPE)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55))
    draw_a(axes[0], data["a"])
    panel_b.draw(axes[1], data["b"])
    panel_c.draw(axes[2], data["c"])
    for ax, letter, x, lx in zip(axes, ("(a)", "(b)", "(c)"), LEGEND_X,
                                 (letter_x_a, LETTER_X, LETTER_X)):
        _style.panel_label(ax, letter, fontsize=LETTER_FS, x=lx)
        ax.get_legend().set_bbox_to_anchor((x, LEGEND_Y), transform=ax.transAxes)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.14, wspace=0.38)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run("fig1_panels", compute)
