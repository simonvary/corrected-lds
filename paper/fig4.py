"""Bias--variance decomposition for Figure 1(c)'s Spearman estimators.

Exactly Figure 1(c)'s setting (CIFAR-10 / ResNet-9, TRAK1 E=100, the R=40 seed mean,
Q=2,000 queries, M=300 population masks, B=2,500 mask-panel draws), derived from
``plotdata/spearman_debias.json`` so the draws are identical.  One panel per estimator,
each showing the terms of ``RMSE^2 = bias^2 + variance``; the stored ddof=1 variance is
scaled by ``(B - 1) / B`` for this exact Monte Carlo identity.  The U-statistic is exactly
unbiased here (no tied pair-query comparisons), so its tiny bias^2 is Monte Carlo error.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style
from _harness import PLOTDATA_DIR, run

SOURCE = "spearman_debias.json"
ESTIMATORS = (
    ("raw", "sample Spearman"),
    ("unbiased_u", "U-statistic corrected"),
)
METRICS = (
    ("bias_squared", r"bias$^2$", "#c44536", "s", (0, (4, 2.4))),
    ("variance", "variance", "#1b6ca8", "o", "-"),
)


def compute(_banks_root):
    with open(os.path.join(PLOTDATA_DIR, SOURCE)) as handle:
        source = json.load(handle)

    draws = int(source["params"]["B"])
    series = []
    for key, label, *_ in ESTIMATORS:
        rows = []
        for source_row in source["rows"]:
            summary = source_row[key]
            bias_squared = summary["bias"] ** 2
            variance = summary["variance"] * (draws - 1) / draws
            mse = summary["rmse"] ** 2
            if abs(mse - bias_squared - variance) > 1e-14:
                raise ValueError("source summary does not satisfy the MSE decomposition")
            rows.append(
                {
                    "M": source_row["M"],
                    "bias": summary["bias"],
                    "bias_squared": bias_squared,
                    "variance": variance,
                    "variance_ddof1": summary["variance"],
                    "mse": mse,
                }
            )
        series.append({"key": key, "label": label, "rows": rows})

    return {
        "banks": source["banks"],
        "tauq": source["tauq"],
        "reference_spearman": source["reference_spearman"],
        "params": {
            **source["params"],
            "derived_from": SOURCE,
            "decomposition": "MSE = bias_squared + variance (ddof=0)",
        },
        "series": series,
    }


def plot(data, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    _style.use_panel()
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.4), sharex=True, sharey=True)
    series = {item["key"]: item for item in data["series"]}

    for ax, (key, label) in zip(axes, ESTIMATORS):
        for metric, metric_label, color, marker, linestyle in METRICS:
            rows = series[key]["rows"]
            ax.plot(
                [row["M"] for row in rows],
                [row[metric] for row in rows],
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.7,
                linestyle=linestyle,
                label=metric_label,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks((4, 8, 16, 32, 64, 128), labels=(4, 8, 16, 32, 64, 128))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", alpha=0.15, linewidth=0.6)
        ax.set_title(label)

    axes[-1].set_xlabel(r"sampled subsets $M$")
    fig.supylabel(r"contribution to MSE of $\rho^{\mathtt{s}}$", x=0.025)
    handles, labels = axes[0].get_legend_handles_labels()
    # the key hangs below the x label (the tight bbox grows to hold it), as in the paper
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.0522),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.95, bottom=0.14, hspace=0.20)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run("fig4", compute, plot)
