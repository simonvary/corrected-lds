"""Talk figure 1: naive seed averaging stalls below the LDS; two seeds and the correction do not.

CIFAR-10 / ResNet-9 with one EK-FAC model (the first replica of ``pool_tda_ekfac``), Pearson
LDS on figure 1(a)'s design (``_rmse_corrected``: its mask grid, B=200 subsamples of m masks
and R seeds, and its seeds): each point is the mean estimate at a budget of ``N = m * R``
retrains, naive for R = 5, 10, 20, 40 and noise-corrected from R = 2.  Dotted: each naive
line's ceiling ``rho_inf / sqrt(1 + lambda / R)``, labelled with its fraction of ``rho_inf``,
the LDS a perfect attributor would reach under naive averaging; ``rho_inf`` and ``lambda``
(its median over queries) come from all 300 masks and 40 seeds.

Writes ``fig_slides1.pdf`` (every naive line, each ceiling labelled with its fraction),
``_step1`` to ``_step4`` (one to four lines on the same axes, unlabelled: a slide's builds) and
``_corrected`` (plus the corrected line).
``--recompute --banks-root <dir>`` needs cifar10's pool.npz and pool_tda_ekfac.npz; about a
minute on CPU.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper"))
import _rmse_corrected as base
import _style
from _harness import bank_stamp, run

from corrected_lds import (
    battery,
    corrected_pearson,
    load_pool,
    load_pool_tda,
    predictions_tda,
    seed_noise_variance_stacked,
)

SLIDES_DIR = os.path.dirname(os.path.abspath(__file__))
SERIES = [("naive", 5), ("naive", 10), ("naive", 20), ("naive", 40), ("corrected", 2)]
# figure 1's colours and markers: the naive reds darken with R, the corrected line is blue
NAIVE = base._FAMILY["naive"]
CORRECTED = base._FAMILY["corrected"]["colors"][0]
CEILING_GREY = "#555555"
FIG_W, FIG_H = 4.1, 4.35                 # inches; one size for every build, so axes stay put
AXES = [0.19, 0.315, 0.64, 0.595]        # left, bottom, width, height (figure fraction)


def _estimate(kind, g, margins):
    """Pearson LDS of predictions ``g`` (m, Q) against readouts ``margins`` (r, m, Q): the
    r-seed mean's correlation or its noise-corrected value, averaged over queries."""
    ybar = margins.mean(axis=0)
    if kind == "naive":
        return float(battery(base._colwise_corr(g, ybar)))
    sigma2 = seed_noise_variance_stacked(margins)
    rho, _ = corrected_pearson(g, ybar, sigma2 / margins.shape[0], clip=base.CLIP)
    return float(battery(rho))


def compute(banks_root):
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    g = predictions_tda(pool, load_pool_tda(os.path.join(bank, "pool_tda_ekfac.npz")), [0])
    sigma2 = seed_noise_variance_stacked(pool.margins)
    rho, diag = corrected_pearson(g, pool.margins.mean(axis=0), sigma2 / pool.R, clip=base.CLIP)
    rho_inf, lam = float(battery(rho)), float(np.median(diag["lambda_q"] * pool.R))
    series = []
    for kind, R in SERIES:
        rows = []
        for m in sorted({base.M_MIN[kind], *base.M_FILL}):
            if m < base.M_MIN[kind] or m * R < 9:
                continue
            rng = np.random.default_rng([base.SEED, R, m, int(kind == "corrected")])
            ests = np.empty(base.B)
            for b in range(base.B):
                masks = rng.choice(pool.M, size=m, replace=False)
                seeds = rng.choice(pool.R, size=R, replace=False)
                ests[b] = _estimate(kind, g[masks], pool.margins[np.ix_(seeds, masks)])
            rows.append({"N": m * R, "m": m, "mean": float(ests.mean()),
                         "sd": float(ests.std(ddof=1))})
        series.append({"kind": kind, "R": R, "rows": rows})
        print(f"[fig_slides1] {kind} R={R}: {rows[-1]['mean']:.4f} at N={rows[-1]['N']}",
              flush=True)
    return {"banks": {"cifar10": bank_stamp(pool)}, "rho_inf": rho_inf, "lambda_med": lam,
            "series": series,
            "params": {"attributor": "ekfac E=1 (pool_tda_ekfac replica 0)",
                       "lds": "Pearson, mean over queries", "B": base.B, "seed": base.SEED,
                       "clip": base.CLIP, "m_fill": base.M_FILL, "m_min": base.M_MIN,
                       "point": "mean of B estimates, each on m masks and R seeds drawn "
                                "without replacement (figure 1(a)'s draws)"}}


def _draw(fig, data, naive_shown, corrected, labelled, ylim):
    """The naive lines for the first ``naive_shown`` Rs with their ceilings; ``labelled``
    marks each ceiling's fraction, ``corrected`` adds the two-seed line and its own."""
    from matplotlib.lines import Line2D
    ax = fig.add_axes(AXES)
    rho_inf, lam = data["rho_inf"], data["lambda_med"]
    handles = []
    naive = [s for s in data["series"] if s["kind"] == "naive"]
    for i, s in enumerate(naive[:naive_shown]):
        color, frac = NAIVE["colors"][i], 1.0 / np.sqrt(1.0 + lam / s["R"])
        handles += ax.plot([r["N"] for r in s["rows"]], [r["mean"] for r in s["rows"]],
                           color=color, linestyle=NAIVE["linestyle"], linewidth=2.0,
                           marker=base._MARKERS[i], markersize=5.5,
                           label=rf"naive, $R={s['R']}$")
        ax.axhline(rho_inf * frac, color=color, linestyle=":", linewidth=1.4, zorder=1)
        if labelled:
            ax.text(1.03, rho_inf * frac, f"{frac:.2f}", transform=ax.get_yaxis_transform(),
                    color=color, va="center", fontsize=11)
    if corrected:
        s = next(s for s in data["series"] if s["kind"] == "corrected")
        handles += ax.plot([r["N"] for r in s["rows"]], [r["mean"] for r in s["rows"]],
                           color=CORRECTED, linewidth=2.0, marker="o", markersize=5.5,
                           label=rf"corrected, $R={s['R']}$")
        ax.axhline(rho_inf, color=CORRECTED, linestyle=":", linewidth=1.4, zorder=1)
        if labelled:
            ax.text(1.03, rho_inf, "1.00", transform=ax.get_yaxis_transform(),
                    color=CORRECTED, va="center", fontsize=11)
    if labelled:
        ax.text(1.03, 1.02, "perfect\nattributor", transform=ax.transAxes, color=CEILING_GREY,
                va="bottom", ha="left", fontsize=8.5, linespacing=0.95)
    handles.append(Line2D([], [], color=CEILING_GREY, linestyle=":", linewidth=1.4,
                          label="ceiling"))
    ax.set_xscale("log")
    ax.set_xlim(8, 15000)
    ax.set_xticks([10, 100, 1000, 10000], ["10", "100", "1000", "10000"])
    ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.set_xlabel(r"total retrains  $N = M \cdot R$")
    ax.set_ylabel("LDS")
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(AXES[0] + AXES[2] / 2, 0.175),
               handlelength=2.0, columnspacing=1.2)


def plot(data, out_path):
    import matplotlib.pyplot as plt

    _style.use_panel()
    plt.rcParams["savefig.bbox"] = "standard"    # fixed page size, so the builds line up
    vals = [r["mean"] for s in data["series"] for r in s["rows"]] + [data["rho_inf"]]
    pad = 0.04 * (max(vals) - min(vals))
    ylim = (min(vals) - pad, max(vals) + pad)
    stem = out_path[:-4]
    n_naive = sum(s["kind"] == "naive" for s in data["series"])
    # (file, naive lines shown, corrected line, ceilings labelled)
    builds = [(f"{stem}_step{k}", k, False, False) for k in range(1, n_naive + 1)]
    builds += [(stem, n_naive, False, True), (f"{stem}_corrected", n_naive, True, True)]
    for path, shown, corrected, labelled in builds:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        _draw(fig, data, shown, corrected, labelled, ylim)
        fig.savefig(path + ".pdf")
        fig.savefig(path + ".png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig_slides1", compute, plot, root=SLIDES_DIR))
