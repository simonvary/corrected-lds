"""Spearman bias--variance decomposition as the number of queries changes.

Figure 1(c)'s CIFAR-10 / ResNet-9 TRAK1 setup: E=100 independent TRAK features, the R=40
seed mean, M=300 population masks, B=2,500 mask panels drawn uniformly without replacement
and shared by every Q.  The panels use nested subsets of one seeded query permutation
(Q=1 within Q=100 within Q=1,000); per Q the target is the mean ordinary Spearman LDS over
those queries on the full mask population.  Terms: the exact Monte Carlo decomposition
``MSE = bias^2 + variance``, with population (ddof=0) variance over the B draws.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style
from _harness import bank_stamp, run
from _spearman_debias import (
    MASK_COUNTS,
    SEED,
    B,
    _panel_spearman,
    _rank_correlation,
)

from corrected_lds import load_pool, load_tauq
from corrected_lds.banks.artifacts import predictions

QUERY_COUNTS = (1, 100, 1_000)
QUERY_SEED = 0
METRICS = (
    ("bias_squared", r"bias$^2$", "#c44536", "s", (0, (4, 2.4))),
    ("variance", "variance", "#1b6ca8", "o", "-"),
)


def _decomposition(values, reference):
    errors = np.asarray(values, dtype=np.float64) - reference
    return {
        "bias": float(errors.mean()),
        "bias_squared": float(errors.mean() ** 2),
        "variance": float(values.var(ddof=0)),
        "mse": float(np.mean(errors**2)),
    }


def compute(banks_root):
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    tauq_name = "tauq_trak1_e100.npz"
    tauq = load_tauq(os.path.join(bank, tauq_name))
    e_features = int(tauq.provenance.get("e_features", 1))
    if pool.Q != 2_000 or tauq.n_targets != pool.Q:
        raise ValueError("this figure requires the full Q=2,000 CIFAR-10 bank")
    if tauq.n_models != 100 or e_features != 1:
        raise ValueError("this figure requires TRAK1 with E=100")

    query_order = np.random.default_rng(QUERY_SEED).permutation(pool.Q)
    query_indices = query_order[: max(QUERY_COUNTS)]
    predicted = predictions(pool, tauq)[:, query_indices]
    observed = pool.margins.mean(axis=0)[:, query_indices]

    full_spearman = _rank_correlation(predicted, observed, axis=0)
    references = {
        queries: float(np.mean(full_spearman[:queries])) for queries in QUERY_COUNTS
    }

    rows = {queries: [] for queries in QUERY_COUNTS}
    rng = np.random.default_rng(SEED)
    for masks in MASK_COUNTS:
        mask_panels = np.vstack(
            [rng.choice(pool.M, masks, replace=False) for _ in range(B)]
        )
        sample_q = _panel_spearman(predicted, observed, mask_panels)
        for queries in QUERY_COUNTS:
            raw = np.mean(sample_q[:, :queries], axis=1, dtype=np.float64)
            rows[queries].append(
                {
                    "M": masks,
                    "raw": _decomposition(raw, references[queries]),
                }
            )
        print(f"[fig8] M={masks}", flush=True)

    return {
        "banks": {"cifar10": bank_stamp(pool)},
        "tauq": {
            "artifact": tauq_name,
            "method": tauq.provenance.get("method"),
            "aggregation": tauq.provenance.get("aggregation"),
            "E": int(tauq.n_models),
            "e_features": e_features,
        },
        "params": {
            "mask_counts": MASK_COUNTS,
            "B": B,
            "mask_seed": SEED,
            "query_counts": QUERY_COUNTS,
            "query_seed": QUERY_SEED,
            "query_indices": query_indices,
            "sampling": "uniform without replacement",
            "observed": f"mean of all R={pool.R} retraining replicas",
            "target": f"ordinary Spearman LDS on the full finite M={pool.M} bank",
            "decomposition": "MSE = bias_squared + variance (ddof=0)",
        },
        "panels": [
            {
                "Q": queries,
                "reference_spearman": references[queries],
                "rows": rows[queries],
            }
            for queries in QUERY_COUNTS
        ],
    }


def plot(data, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    _style.use_panel()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.15), sharex=True, sharey=True)

    for ax, panel in zip(axes, data["panels"]):
        masks = [row["M"] for row in panel["rows"]]
        for metric, component, color, marker, linestyle in METRICS:
            ax.plot(
                masks,
                [row["raw"][metric] for row in panel["rows"]],
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.4,
                linestyle=linestyle,
                label=component,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks((4, 8, 16, 32, 64, 128), labels=(4, 8, 16, 32, 64, 128))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", alpha=0.15, linewidth=0.6)
        ax.set_title(f"$Q={panel['Q']:,}$")
        ax.set_xlabel(r"sampled subsets $M$")

    axes[0].set_ylabel(r"Contribution to MSE of $\rho^{\mathtt{s}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.01),
        ncol=2,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.89, bottom=0.30, wspace=0.12)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run("fig8", compute, plot)
