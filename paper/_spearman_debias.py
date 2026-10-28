"""Figure-1 panel (c): finite-mask Spearman debiasing for TRAK1, E=100.

Target: ordinary Spearman LDS of the released ``tauq_trak1_e100.npz`` on the full M=300
mask population, readout fixed to the R=40 seed mean, all Q=2,000 queries.  Per smaller
mask count, B panels drawn uniformly without replacement feed both estimators: sample
Spearman and its exact finite-population U-statistic correction, whose no-tie identity
holds exactly (the CIFAR-10 bank has no tied pair-query comparisons).  CPU-only.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import bank_stamp, run

from corrected_lds import load_pool, load_tauq
from corrected_lds.banks.artifacts import predictions
from corrected_lds.rank_debias import finite_population_unbiased_spearman

MASK_COUNTS = (4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
B = 2_500
SEED = 0


def _rank_correlation(x, y, axis):
    x = stats.rankdata(x, axis=axis)
    y = stats.rankdata(y, axis=axis)
    x -= x.mean(axis=axis, keepdims=True)
    y -= y.mean(axis=axis, keepdims=True)
    return np.sum(x * y, axis=axis) / np.sqrt(
        np.sum(x**2, axis=axis) * np.sum(y**2, axis=axis)
    )


def _panel_spearman(predicted, observed, panels, trial_chunk=32, query_chunk=256):
    """Per-panel, per-query Spearman correlations without a full B x M x Q copy."""
    output = np.empty((len(panels), predicted.shape[1]), dtype=np.float64)
    for t in range(0, len(panels), trial_chunk):
        selected = panels[t : t + trial_chunk]
        for q in range(0, predicted.shape[1], query_chunk):
            values = _rank_correlation(
                predicted[selected, q : q + query_chunk],
                observed[selected, q : q + query_chunk],
                axis=1,
            )
            output[t : t + len(selected), q : q + values.shape[1]] = values
    return output


def _pairwise_mean_concordance(predicted, observed):
    """Mean Kendall sign kernel over Q for every pair of population masks."""
    population = len(predicted)
    kernel = np.zeros((population, population), dtype=np.float64)
    ties = 0
    for left in range(population - 1):
        dx = np.sign(predicted[left + 1 :] - predicted[left])
        dy = np.sign(observed[left + 1 :] - observed[left])
        ties += int(np.count_nonzero((dx == 0) | (dy == 0)))
        values = np.mean(dx * dy, axis=1, dtype=np.float64)
        kernel[left, left + 1 :] = values
        kernel[left + 1 :, left] = values
    return kernel, ties


def _panel_mean_kendall(kernel, panels, chunk=128):
    masks = panels.shape[1]
    left, right = np.triu_indices(masks, 1)
    output = np.empty(len(panels), dtype=np.float64)
    for start in range(0, len(panels), chunk):
        selected = panels[start : start + chunk]
        output[start : start + len(selected)] = np.mean(
            kernel[selected[:, left], selected[:, right]], axis=1, dtype=np.float64
        )
    return output


def _summary(values, reference):
    values = np.asarray(values, dtype=np.float64)
    errors = values - reference
    squared_errors = errors**2
    rmse = float(np.sqrt(squared_errors.mean()))
    return {
        "mean": float(values.mean()),
        "bias": float(errors.mean()),
        "variance": float(values.var(ddof=1)),
        "rmse": rmse,
        "rmse_mc_se": float(
            squared_errors.std(ddof=1) / np.sqrt(len(values)) / (2.0 * rmse)
        ),
    }


def compute(banks_root):
    bank = os.path.join(banks_root, "cifar10")
    pool = load_pool(os.path.join(bank, "pool.npz"))
    tauq_name = "tauq_trak1_e100.npz"
    tauq = load_tauq(os.path.join(bank, tauq_name))
    e_features = int(tauq.provenance.get("e_features", 1))
    if pool.Q != 2_000 or tauq.n_targets != pool.Q:
        raise ValueError("Figure 1(c) requires the full Q=2,000 bank")
    if tauq.n_models != 100 or e_features != 1:
        raise ValueError("Figure 1(c) requires TRAK1 with E=100")

    predicted = predictions(pool, tauq)
    observed = pool.margins.mean(axis=0)
    reference = float(np.mean(_rank_correlation(predicted, observed, axis=0)))
    kernel, ties = _pairwise_mean_concordance(predicted, observed)
    if ties:
        raise ValueError(f"exact U-statistic panel requires no ties; found {ties}")

    rng = np.random.default_rng(SEED)
    rows = []
    for masks in MASK_COUNTS:
        panels = np.vstack(
            [rng.choice(pool.M, masks, replace=False) for _ in range(B)]
        )
        sample_q = _panel_spearman(predicted, observed, panels)
        raw = np.mean(sample_q, axis=1, dtype=np.float64)
        kendall = _panel_mean_kendall(kernel, panels)
        unbiased = finite_population_unbiased_spearman(raw, kendall, masks, pool.M)
        rows.append(
            {
                "M": masks,
                "raw": _summary(raw, reference),
                "unbiased_u": _summary(unbiased, reference),
            }
        )
        print(
            f"[_spearman_debias] M={masks}: "
            f"raw {rows[-1]['raw']['rmse']:.5f}, "
            f"U {rows[-1]['unbiased_u']['rmse']:.5f}",
            flush=True,
        )

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
            "seed": SEED,
            "queries": pool.Q,
            "sampling": "uniform without replacement",
            "observed": f"mean of all R={pool.R} retraining replicas",
            "target": f"ordinary Spearman LDS on the full finite M={pool.M} bank",
            "rank_debias": "finite_population_unbiased_spearman",
            "population_pair_query_ties": ties,
        },
        "reference_spearman": reference,
        "rows": rows,
    }


def draw(ax, data):
    """Render figure 1(c) into ``ax``."""
    from matplotlib.ticker import NullLocator

    ax.set_box_aspect(1)
    styles = (
        ("raw", "sample Spearman", "#c44536", "s", (0, (4, 2.4))),
        ("unbiased_u", "U-statistic corrected", "#1b6ca8", "o", "-"),
    )
    masks = [row["M"] for row in data["rows"]]
    for key, label, color, marker, linestyle in styles:
        ax.plot(
            masks,
            [row[key]["rmse"] for row in data["rows"]],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.7,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128])
    yticks = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value:g}" for value in yticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(3.6, 142)
    ax.set_ylim(0.0009, 0.11)
    ax.set_xlabel("sampled subsets $M$")
    ax.set_ylabel(r"RMSE to full-bank $\rho^{\mathtt{s}}$")
    ax.grid(axis="y", which="major", alpha=0.15, linewidth=0.6)
    ax.legend(
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        handlelength=1.8,
        borderaxespad=0.0,
        handletextpad=0.5,
    )


if __name__ == "__main__":
    run("spearman_debias", compute)
