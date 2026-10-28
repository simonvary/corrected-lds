"""Finite-mask Spearman debiasing across datasets and attribution methods.

Two 2x3 grids: rows MNIST and CIFAR-10, columns TRAK1 E=100, EK-FAC E=30 and SOURCE L=3
E=30, all on the ``M=300, R=40, Q=2,000`` pool geometry with the retraining budget fixed at
the first ``R=5`` nested replicas.  Per cell and mask count, ``B`` panels drawn uniformly
without replacement are shared by all estimators.

``figures/fig9.pdf`` compares sample Spearman with the finite-population U-statistic
correction against ordinary Spearman on the full 300-mask bank at R=5 (exact for continuous
pairs; the plotdata records the negligible tied pair-query comparisons).
``figures/fig9_noise_corrected.pdf`` applies the rank-first noise composition to both:
Spearman mapped to latent Pearson, times the battery factor ``corrected Pearson / finite-R
Pearson`` (seed noise from the panel's five replicas), mapped back; the target is the same
composition on the full bank.  One computation feeds both, so their panels pair exactly.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _spearman_debias as rank
import _style
from _harness import bank_stamp, run

from corrected_lds import load_pool, load_tauq, pearson_to_spearman
from corrected_lds.banks.artifacts import predictions
from corrected_lds.rank_debias import finite_population_unbiased_spearman

MASK_COUNTS = (4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
B = 600
R = 5
SEED = 0
CLIP = 2.0
TRIAL_CHUNK = 16

DATASETS = (
    ("mnist", "MNIST"),
    ("cifar10", "CIFAR-10"),
)
METHODS = (
    ("trak", "TRAK", "tauq_trak1_e100.npz", r"$E=100$"),
    ("ekfac", "EK-FAC", "tauq_ekfac_e30.npz", r"$E=30$"),
    ("source", "SOURCE", "tauq_source_L3_e30.npz", r"$L=3,\ E=30$"),
)


def _noise_factors(predicted, observed, seed_variance, panels):
    """Battery-level retraining-noise factor per sampled mask panel: ``evaluate_pool``'s
    ``corrected_rho_inf / rho_R``, batched to avoid thousands of ``corrected_pearson``
    calls.  ``seed_variance``: per-mask, per-query variance across the R replicas."""
    factors = np.empty(len(panels), dtype=np.float64)
    for start in range(0, len(panels), TRIAL_CHUNK):
        selected = panels[start : start + TRIAL_CHUNK]
        g = predicted[selected]
        y = observed[selected]
        g -= g.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)

        scale = selected.shape[1] - 1
        covariance = np.sum(g * y, axis=1) / scale
        variance_g = np.sum(g * g, axis=1) / scale
        variance_y = np.sum(y * y, axis=1) / scale
        finite_r_noise = seed_variance[selected].mean(axis=1) / R

        pearson_denominator = np.sqrt(variance_g * variance_y)
        pearson = np.divide(
            covariance,
            pearson_denominator,
            out=np.zeros_like(covariance),
            where=pearson_denominator > 0,
        )

        signal_variance = np.maximum(
            variance_y - finite_r_noise, variance_y / (CLIP * CLIP)
        )
        corrected_denominator = np.sqrt(variance_g * signal_variance)
        corrected = np.clip(
            np.divide(
                covariance,
                corrected_denominator,
                out=np.zeros_like(covariance),
                where=corrected_denominator > 0,
            ),
            -1.0,
            1.0,
        )

        rho_r = pearson.mean(axis=1)
        rho_inf = corrected.mean(axis=1)
        factors[start : start + len(selected)] = np.divide(
            rho_inf,
            rho_r,
            out=np.ones_like(rho_inf),
            where=np.abs(rho_r) > 1e-12,
        )
    return factors


def _combine_rank_and_noise(spearman, noise_factor):
    """The rank-first composition used by ``evaluate_pool``."""
    latent_pearson = 2.0 * np.sin(np.pi * np.clip(spearman, -1.0, 1.0) / 6.0)
    return pearson_to_spearman(noise_factor * latent_pearson)


def _cell(pool, bank, method_key, label, tauq_name, setup, panels):
    tauq = load_tauq(os.path.join(bank, tauq_name))
    predicted = predictions(pool, tauq)
    margins = pool.margins[:R]
    observed = margins.mean(axis=0)
    seed_variance = np.var(margins, axis=0, ddof=1)

    reference = float(np.mean(rank._rank_correlation(predicted, observed, axis=0)))
    full_panel = np.arange(pool.M, dtype=np.int64)[None, :]
    full_noise_factor = float(
        _noise_factors(predicted, observed, seed_variance, full_panel)[0]
    )
    combined_reference = float(_combine_rank_and_noise(reference, full_noise_factor))
    kernel, ties = rank._pairwise_mean_concordance(predicted, observed)

    rows = []
    for masks in MASK_COUNTS:
        selected = panels[masks]
        sample_q = rank._panel_spearman(predicted, observed, selected)
        raw = np.mean(sample_q, axis=1, dtype=np.float64)
        kendall = rank._panel_mean_kendall(kernel, selected)
        unbiased = finite_population_unbiased_spearman(
            raw, kendall, masks, pool.M
        )
        noise_factor = _noise_factors(
            predicted, observed, seed_variance, selected
        )
        raw_combined = _combine_rank_and_noise(raw, noise_factor)
        unbiased_combined = _combine_rank_and_noise(unbiased, noise_factor)
        rows.append(
            {
                "M": masks,
                "finite_r": {
                    "raw": rank._summary(raw, reference),
                    "unbiased_u": rank._summary(unbiased, reference),
                },
                "noise_corrected": {
                    "raw": rank._summary(raw_combined, combined_reference),
                    "unbiased_u": rank._summary(
                        unbiased_combined, combined_reference
                    ),
                    "noise_factor_mean": float(noise_factor.mean()),
                    "noise_factor_sd": float(noise_factor.std(ddof=1)),
                },
            }
        )
        print(
            f"[fig9] {pool.dataset}/{method_key} M={masks}: "
            f"finite {rows[-1]['finite_r']['raw']['rmse']:.5f} -> "
            f"{rows[-1]['finite_r']['unbiased_u']['rmse']:.5f}; "
            f"combined {rows[-1]['noise_corrected']['raw']['rmse']:.5f} -> "
            f"{rows[-1]['noise_corrected']['unbiased_u']['rmse']:.5f}",
            flush=True,
        )

    return {
        "dataset": pool.dataset,
        "method": method_key,
        "label": label,
        "setup": setup,
        "tauq": {
            "artifact": tauq_name,
            "method": tauq.provenance.get("method"),
            "n_models": tauq.n_models,
            "proj_dim": tauq.proj_dim,
            "e_features": tauq.provenance.get("e_features", 1),
        },
        "population_pair_query_ties": ties,
        "reference_finite_r_spearman": reference,
        "full_bank_noise_factor": full_noise_factor,
        "reference_noise_corrected_spearman": combined_reference,
        "rows": rows,
    }


def compute(banks_root):
    banks = {}
    cells = []
    for dataset_index, (dataset, _) in enumerate(DATASETS):
        bank = os.path.join(banks_root, dataset)
        pool = load_pool(os.path.join(bank, "pool.npz"))
        if pool.M != 300 or pool.R < R or pool.Q != 2_000:
            raise ValueError(
                f"{dataset} must have M=300, R>={R}, Q=2,000; "
                f"found M={pool.M}, R={pool.R}, Q={pool.Q}"
            )
        banks[dataset] = bank_stamp(pool)
        rng = np.random.default_rng([SEED, dataset_index])
        panels = {
            masks: np.vstack(
                [rng.choice(pool.M, masks, replace=False) for _ in range(B)]
            )
            for masks in MASK_COUNTS
        }
        for method in METHODS:
            cells.append(_cell(pool, bank, *method, panels))

    return {
        "banks": banks,
        "params": {
            "mask_counts": MASK_COUNTS,
            "B": B,
            "R": R,
            "seed": SEED,
            "clip": CLIP,
            "sampling": "uniform without replacement",
            "retraining_replicas": f"first {R} nested pool replicas",
            "finite_r_target": "ordinary Spearman on the complete finite mask bank",
            "noise_composition": (
                "rank-first: inverse Gaussian Spearman bridge, multiply by "
                "corrected-Pearson / finite-R-Pearson battery factor, bridge back"
            ),
        },
        "cells": cells,
    }


def _plot_grid(data, mode, out_path):
    import matplotlib.pyplot as plt

    _style.use()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.9), sharex=True, sharey=True)
    cells = {(cell["dataset"], cell["method"]): cell for cell in data["cells"]}
    styles = (
        ("raw", "sample Spearman", "#c44536", "s", (0, (4, 2.4))),
        ("unbiased_u", "U-statistic corrected", "#1b6ca8", "o", "-"),
    )
    if mode == "noise_corrected":
        styles = (
            ("raw", "sample Spearman + noise", "#c44536", "s", (0, (4, 2.4))),
            ("unbiased_u", "U-statistic + noise", "#1b6ca8", "o", "-"),
        )
    for row, (dataset, _) in enumerate(DATASETS):
        for column, (method, method_label, _, _) in enumerate(METHODS):
            ax = axes[row, column]
            cell = cells[dataset, method]
            masks = [point["M"] for point in cell["rows"]]
            for key, label, color, marker, line in styles:
                rmse = [point[mode][key]["rmse"] for point in cell["rows"]]
                ax.plot(
                    masks,
                    rmse,
                    color=color,
                    marker=marker,
                    linestyle=line,
                    linewidth=1.8,
                    markersize=4.2,
                    label=label,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks((4, 8, 16, 32, 64, 128), labels=(4, 8, 16, 32, 64, 128))
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            ax.text(
                0.97,
                0.96,
                cell["setup"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            ax.text(
                -0.13,
                1.04,
                f"({chr(ord('a') + row * 3 + column)})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
            if row == 0:
                ax.set_title(method_label, fontweight="bold")

    for (_, dataset_label), y in zip(DATASETS, (0.70, 0.31)):
        fig.text(0.018, y, dataset_label, rotation=90, ha="center", va="center",
                 fontweight="bold", fontsize=11)
    fig.supxlabel(r"sampled subsets $M$", y=0.095)
    target = (rf"finite-$R$ ($R={data['params']['R']}$) full bank" if mode == "finite_r"
              else "noise-corrected full bank")
    fig.supylabel(rf"RMSE to {target} $\rho^s$", x=0.057)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.005),
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.20,
                        wspace=0.18, hspace=0.27)
    out = Path(out_path)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot(data, out_path):
    _plot_grid(data, "finite_r", out_path)
    combined = Path(out_path).with_name(
        Path(out_path).stem + "_noise_corrected" + Path(out_path).suffix
    )
    _plot_grid(data, "noise_corrected", combined)
    print(f"[fig9] wrote {combined}")


if __name__ == "__main__":
    run("fig9", compute, plot)
