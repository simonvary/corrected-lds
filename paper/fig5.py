"""Figure 5, the appendix to figure 3: the arcsin bridge in detail, on the reconstructed rho_5
route (three methods per setting at E=1, TRAK as TRAK-1; 5-seed readouts; all masks); rows
CIFAR-10 (a-c), QNLI (d-f) and WikiText-2 (g-i).

  (a, d, g) The bias, decomposed: reconstructed minus naive 5-seed Spearman (filled, 95%
         block-jackknife interval), the conversion alone (open: h(5-seed Pearson) minus
         5-seed Spearman) and its exactly Gaussian value at this M (cross: the finite-mask
         part, as the conversion yields the infinite-mask Spearman); grey: +-1 SE of one estimate.
  (b, e, h) Bivariate normality as seeds are averaged: EK-FAC's detrended D^2 Q-Q (empirical
         minus Gaussian quantile) at R = 1, 5 and all seeds; grey: figure 5's 90% Gaussian
         range at R = 5.  The negative control (the 5-seed readout through sign(f)|f|^0.7,
         moving only Pearson) is kept in the plotdata for the caption, not drawn.
  (c, f, i) Where single-seed non-normality comes from: the readout's excess kurtosis minus
         its Gaussian expectation against R, with the 90% range of exactly Gaussian data
         (rotation surrogates); the attribution method does not enter.

(a, c, d, f, g, i) derive from ``fig5_panels.json``, ``bridge_estimators.json`` and
``proxy_sweep.json``; (b, e, h) and the control need the banks for ``--recompute`` (about a
minute).  R is the paper's seeds per mask (plotdata ``k``), not the ensemble size E (= 1).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _assumptions as fa
import _shared
import _style
from _harness import PLOTDATA_DIR, run

from corrected_lds import battery, load_pool, load_pool_tda, pearson_to_spearman, predictions_tda

# Drawn at 7.0 in, printed at \textwidth (~5.5 in, ~79%): fonts >= 8.5 pt print at >= ~6.7 pt.
# Heights in inches, one row per bank: column headers above the top row, keys below the last.
FIG_W, ROW_IN, GAP_IN, TOP_IN, BOTTOM_IN = 7.0, 1.385, 0.62, 0.74, 0.97
HEADER_IN = 0.347            # the column headers' baseline above the top row
SE_GREY = "#9e9e9e"          # +-1 SE of the estimate, (a, d) — the darker grey
RANGE_GREY = "#d6d6d6"       # 90% Gaussian range, (b, c, e, f) — the lighter grey
KTICKS = (1, 10, 20, 30, 40)
KEY_TOP_IN = 0.485           # top of the keys, above the figure's bottom edge
FS = {"tick": 8.5, "label": 9.0, "legend": 8.5, "title": 9.5, "header": 9.5,
      "letter": 11.5}
HEADERS = ["bias of the estimate\n(decomposed)",
           "bivariate normality\n(Q-Q deviation, EK-FAC)",
           "readout tails\n(excess kurtosis)"]
QQ_METHOD = ("pool_tda_ekfac", "EK-FAC")      # the method both rows share
N_SUB = 10                                     # seed subsets averaged per k (as figure 5)


def _load(name):
    with open(os.path.join(PLOTDATA_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def compute(banks_root):
    f5, est, sweep = _load("fig5_panels"), _load("bridge_estimators"), _load("proxy_sweep")
    for bank in f5["banks"]:
        for other, name in ((est, "bridge_estimators"), (sweep, "proxy_sweep")):
            if other["banks"][bank] != f5["banks"][bank]:
                raise SystemExit(f"[fig5] {bank}: {name} is from a different bank")
    bias, kurt = {}, {}
    for bank, methods in _shared.B_METHODS.items():
        pq = est["panels"][bank]["per_query"]
        bias[bank] = [{"label": label, **{k: pq[tda][k] for k in
                                          ("D", "D_se", "conversion", "conversion_se",
                                           "conversion_gaussian", "se_estimator")}}
                      for tda, label in methods]
        rows = sweep["panels"][bank]["footings"]["E1"]["per_k"]
        kurt[bank] = {"k": [r["k"] for r in rows],
                      "excess": [r["kurt_f"]["data"] - r["kurt_f"]["null_mean"] for r in rows],
                      "lo": [r["kurt_f"]["null_lo"] - r["kurt_f"]["null_mean"] for r in rows],
                      "hi": [r["kurt_f"]["null_hi"] - r["kurt_f"]["null_mean"] for r in rows]}
    qqk = {bank: _qq_by_k(banks_root, bank, f5) for bank in _shared.B_METHODS}
    return {"banks": f5["banks"], "bias": bias, "kurt": kurt, "qqk": qqk,
            "params": {"derived_from": ["fig5_panels.json", "bridge_estimators.json",
                                        "proxy_sweep.json"],
                       "kurtosis": "readout averaged over k seeds; proxy_sweep's E1 rows "
                                   "(the readout alone, attributor-free)"}}


def _qq_by_k(banks_root, bank, f5):
    """EK-FAC's D^2 Q-Q deviation at k = 1 (ten random seeds), 5 (figure 5's subsets) and R,
    and the negative control at k = 5 with its bridge shift in mean_q(Spearman - h(Pearson))."""
    from scipy import stats
    pool = load_pool(os.path.join(banks_root, bank, "pool.npz"))
    g = predictions_tda(pool, load_pool_tda(os.path.join(banks_root, bank,
                                                         f"{QQ_METHOD[0]}.npz")),
                        _shared.E1_REPLICAS)
    rank_g = stats.rankdata(g, axis=0)
    probs, null_q = fa._qq_grid(pool.M)
    sub_rng = np.random.default_rng([_shared.SEED, 99])
    five = [sub_rng.choice(pool.R, _shared.TARGET_R, replace=False)
            for _ in range(_shared.N_TRUTH)][:N_SUB]
    ones = [[int(i)] for i in np.random.default_rng([_shared.SEED, 5]).choice(pool.R, N_SUB,
                                                                            replace=False)]
    groups = {1: ones, _shared.TARGET_R: five, pool.R: [list(range(pool.R))]}

    def curve(subsets, transform=None):
        cs, gaps = [], []
        for s in subsets:
            y = pool.margins[s].mean(axis=0)
            y = y if transform is None else transform(y)
            cs.append(np.quantile(fa._mahalanobis2(g, y), probs) - null_q)
            sp = _shared._colcorr(rank_g, stats.rankdata(y, axis=0))
            gaps.append(float(battery(sp - pearson_to_spearman(_shared._colcorr(g, y)))))
        return np.mean(cs, axis=0), float(np.mean(gaps))

    curves = {k: curve(sub)[0] for k, sub in groups.items()}
    ref = next(m for m in f5["qq"][bank]["methods"] if m["tda"] == QQ_METHOD[0])
    if np.max(np.abs(curves[_shared.TARGET_R] - np.asarray(ref["curve"]))) > 1e-9:
        raise SystemExit(f"[fig5] {bank}: the k=5 curve differs from figure 5's")
    ctrl, gap_c = curve(five, fa._control)
    _, gap_d = curve(five)
    print(f"[fig5] {bank}: control max|dev| {np.abs(ctrl).max():.3f} vs data "
          f"{np.abs(curves[_shared.TARGET_R]).max():.3f}; control shifts the bridge by "
          f"{gap_c - gap_d:+.5f}", flush=True)
    return {"label": QQ_METHOD[1], "null_q": null_q.tolist(), "ks": list(groups),
            "curves": [curves[k].tolist() for k in groups], "band": ref["band"],
            "control": ctrl.tolist(), "control_shift": gap_c - gap_d,
            "gap_data": gap_d, "gap_control": gap_c}


def _bias_xlim(data):
    """One range for both rows, so (a) and (d) compare directly."""
    ms = [m for rows in data["bias"].values() for m in rows]
    lo = min(min(m["D"] - 1.96 * m["D_se"], -m["se_estimator"], m["conversion"],
                 m["conversion_gaussian"]) for m in ms)
    hi = max(max(m["D"] + 1.96 * m["D_se"], m["se_estimator"], m["conversion"],
                 m["conversion_gaussian"]) for m in ms)
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def draw_bias(ax, data, bank, xlim):
    """Per method: bias (filled, 95% CI), conversion (open), Gaussian part (cross), +-1 SE."""
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    blue = _shared.STYLE_B[("recon", 2)][0]
    rows = data["bias"][bank]
    y = np.arange(len(rows))[::-1].astype(float)
    for yi, m in zip(y, rows):
        ax.fill_betweenx([yi - 0.34, yi + 0.34], -m["se_estimator"], m["se_estimator"],
                         color=SE_GREY, alpha=0.45, linewidth=0, zorder=1)
        ax.errorbar(m["D"], yi + 0.12, xerr=1.96 * m["D_se"], color=blue, marker="s",
                    markersize=4.4, elinewidth=1.1, capsize=0, linestyle="none", zorder=4)
        ax.plot(m["conversion"], yi - 0.2, marker="s", markersize=4.0, color=blue,
                markerfacecolor="white", markeredgewidth=1.0, linestyle="none", zorder=4)
        ax.plot(m["conversion_gaussian"], yi - 0.2, marker="x", markersize=4.4,
                color="#222222", markeredgewidth=1.1, linestyle="none", zorder=5)
    ax.axvline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([m["label"] for m in rows], fontsize=FS["label"])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)


def _tint(hex_color, t):
    """``t`` of the way from a light tint (t=0) to the full colour (t=1)."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(1.0 - (0.30 + 0.70 * t) * (1.0 - rgb))


def draw_qqk(ax, data, bank, ylim):
    """EK-FAC's Q-Q deviation at k = 1, 5, R (light to dark) over its Gaussian range at k = 5."""
    q = data["qqk"][bank]
    x = np.asarray(q["null_q"])
    ax.fill_between(x, *q["band"], color=RANGE_GREY, alpha=0.75, linewidth=0, zorder=1)
    color = _shared.BANK_COLOR[bank]
    for n, (k, c) in enumerate(zip(q["ks"], q["curves"])):
        ax.plot(x, c, color=_tint(color, n / (len(q["ks"]) - 1)), linewidth=1.4, zorder=3,
                label=f"$R$ = {k}")
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xlim(0.0, x.max())
    ax.set_ylim(-ylim, ylim)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=FS["legend"], handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.15, frameon=False)


def draw_kurt(ax, data, bank):
    k = data["kurt"][bank]
    ax.fill_between(k["k"], k["lo"], k["hi"], color=RANGE_GREY, alpha=0.75, linewidth=0,
                    zorder=1)
    ax.plot(k["k"], k["excess"], color=_shared.BANK_COLOR[bank], linewidth=1.4, marker="o",
            markersize=3.2, zorder=3)
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    # each row's own seed range
    k_max = max(k["k"])
    ax.set_xlim(0.0 if k_max > 10 else 0.5, k_max + (1.5 if k_max > 10 else 0.5))
    ax.set_xticks([t for t in KTICKS if t <= k_max] if k_max > 10 else k["k"])
    ax.grid(alpha=0.15, linewidth=0.5)


def plot(data, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _style.use()
    plt.rcParams.update({"font.size": FS["label"], "axes.labelsize": FS["label"],
                         "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
                         "legend.fontsize": FS["legend"]})
    banks = list(data["bias"])
    fig_h = TOP_IN + len(banks) * ROW_IN + (len(banks) - 1) * GAP_IN + BOTTOM_IN
    fig, axes = plt.subplots(len(banks), 3, figsize=(FIG_W, fig_h))
    fig.subplots_adjust(left=0.115, right=0.99, top=1 - TOP_IN / fig_h,
                        bottom=BOTTOM_IN / fig_h, wspace=0.62, hspace=GAP_IN / ROW_IN)
    qq_lim = 1.25 * max(max(np.max(np.abs(q["band"])),
                            max(np.max(np.abs(c)) for c in q["curves"]))
                        for q in data["qqk"].values())
    xlim = _bias_xlim(data)
    k_lo = min(min(min(k["excess"]), min(k["lo"])) for k in data["kurt"].values())
    k_hi = max(max(max(k["excess"]), max(k["hi"])) for k in data["kurt"].values())
    k_pad = 0.08 * (k_hi - k_lo)
    for i, bank in enumerate(banks):
        ax_b, ax_q, ax_k = axes[i]
        draw_bias(ax_b, data, bank, xlim)
        ax_b.set_title(_shared.BANK_LABEL[bank], loc="left", color=_shared.BANK_COLOR[bank],
                       fontsize=FS["title"], pad=4)
        draw_qqk(ax_q, data, bank, qq_lim)
        ax_q.set_ylabel(r"empirical $-$ Gaussian")
        draw_kurt(ax_k, data, bank)
        ax_k.set_ylim(k_lo - k_pad, k_hi + k_pad)
        ax_k.set_ylabel("excess kurtosis")
        if i == len(banks) - 1:
            ax_b.set_xlabel(r"reconstructed $-$ naive")
            ax_q.set_xlabel(r"Gaussian quantile of $D^2$")
            ax_k.set_xlabel("retrains per mask  $R$")
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.50 if i % 3 == 0 else -0.36, 1.10, f"({chr(ord('a') + i)})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=FS["letter"],
                fontweight="bold")
    # column headers above the top row
    top_y = axes[0, 0].get_position().y1
    for ax, text in zip(axes[0], HEADERS):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2.0, top_y + HEADER_IN / fig_h, text, ha="center", va="bottom",
                 fontsize=FS["header"], color="#333333", linespacing=1.15)
    # each key under the columns it explains (the Gaussian range under the second and third)
    blue = _shared.STYLE_B[("recon", 2)][0]
    key_bias = [Line2D([], [], color=blue, marker="s", markersize=4.4, linestyle="-",
                       linewidth=1.1, label="total bias, 95% CI"),
                Line2D([], [], color=blue, marker="s", markersize=4.0, linestyle="none",
                       markerfacecolor="white", markeredgewidth=1.0, label="conversion only"),
                Patch(facecolor=SE_GREY, alpha=0.45,
                      label="$\\pm$1 SE of a single estimate"),
                Line2D([], [], color="#222222", marker="x", markersize=4.4, linestyle="none",
                       markeredgewidth=1.1, label="Gaussian pair, finite $M$")]
    key_range = [Patch(facecolor=RANGE_GREY, alpha=0.75, label="90% Gaussian range")]
    # (a, d)'s key is wider than its column, so it starts at the figure's left edge
    col = [axes[-1, j].get_position() for j in range(3)]
    for handles, loc, x, ncol in ((key_bias, "upper left", 0.0, 2),
                                  (key_range, "upper center", (col[1].x0 + col[2].x1) / 2.0,
                                   1)):
        fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False,
                   fontsize=FS["legend"], bbox_to_anchor=(x, KEY_TOP_IN / fig_h), borderaxespad=0.0,
                   handlelength=1.3, columnspacing=0.8, handletextpad=0.4,
                   labelspacing=0.35)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig5", compute, plot))
