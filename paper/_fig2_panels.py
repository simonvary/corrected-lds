"""Figure 2: CIFAR-10, QNLI and WikiText-2 / GPT-2 (the Bae et al. bank), one row each.

(a, d, g) naive vs corrected Pearson LDS against seeds R, the attenuation law as a halo;
(b, e, h) the budget frontier for the rho_5 target, one sub-panel per method; (c, f, i)
corrected LDS against attributor ensemble size E, forecast from K members.

All held out (``_heldout``; masks shared), so no estimate shares a seed with its law or
reference: (a, d, g) fit the law on N_REF_A reference seeds and take the points from R-seed
subsets of the rest, averaged over N_SUB=50 splits; (b, e, h) take rho_5 from five-seed
groups of seed half A and the estimators from half B (300 replicates over 30 splits);
(c, f, i) anchor on the last K replicas with half A of split 0 and measure disjoint-group
means of the earlier replicas with half B, so each curve stops K short of its artifact (K
from ``_forecast``'s clipping gate).

WikiText-2's bergson pool is ``seed_mode="shared-per-replica"``, so it uses the centred
seed-noise estimator (``_seed_noise_fn``), as ``evaluate`` does.  (a, b) show CIFAR-10's
TRAK-10, (d, e, g, h) TRAK-5, each E=1 point one Eq. 15 unit.  ``--recompute --banks-root
<dir>`` needs every bank's pool.npz and pool_tda files (banks-manifest.json); ~15 min on CPU.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _forecast as forecast
import _rmse_corrected as base
import _shared
import _style
from _harness import bank_stamp, run
from _heldout import seed_noise, split

from corrected_lds import (
    battery,
    battery_c,
    corrected_pearson,
    load_pool,
    load_pool_tda,
    member_noise_ratio,
    predictions_tda,
)
from corrected_lds.evaluate import _seed_noise_fn

# Type, in points: the 12.2 in canvas prints at 0.45x (\linewidth); _shared and _forecast match
TYPE = {"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5}
TITLE_FS = 13.0                  # the row titles over (a, d, g)
LETTER_FS = 17.0                 # the panel letters
LETTER_RAISE = 8.0               # their lift over the axes, so (b, h) clear a top tick label
LABEL_FS_C = 12.0                # (c, f, i)'s gutter method labels
YLABEL_FS_C = 13.0               # (c, f, i)'s y label is the longest; at 15 pt it overruns
KEY_PAD = 0.015                  # a key may run this far (figure fraction) past its column
# labels start at or right of forecast.DATA_FRAC (where the curves end), clear of last markers
LABEL_X_C = 0.735
_XLABEL_C = r"ensemble size  $E$"


BANK = "wikitext2_gpt2_bae"
ROWS = ("cifar10", "qnli", BANK)

# Geometry: flat rows (ROW_ASPECT) so three fit the page, wide gaps so labels and keys fit
FIG_W = 12.2
ROW_ASPECT = 0.58
TOP_IN, BOTTOM_IN, GAP_IN = 0.39, 1.04, 0.60
LEGEND_Y_IN, LEGEND_H_IN = 0.062, 0.286

# (a, b, d, e, g, h) show each row's TRAK unit in TRAK-1's slot, with its (c, f, i) name and
# style (this process only: figure 5 keeps TRAK-1)
TRAK1 = "pool_tda_trak1"
ROW_TRAK = {"cifar10": "pool_tda_trak10", "qnli": "pool_tda_trak5", BANK: "pool_tda_trak5"}
for _bank, _unit in ROW_TRAK.items():
    _sty = {t: (lab, col, mk) for t, lab, col, mk in forecast.METHODS[_bank]}
    _shared.A_METHODS[_bank] = [(t, *_sty[t]) for t in
                             (_unit if t == TRAK1 else t
                              for t, _, _, _ in _shared.A_METHODS[_bank])]
    _shared.B_METHODS[_bank] = [(_unit, _sty[_unit][0]) if t == TRAK1 else (t, lab)
                             for t, lab in _shared.B_METHODS[_bank]]


# (a, d, g): reference seeds per bank for the law's fit; the rest feed the plotted points
N_REF_A = {"cifar10": 20, "qnli": 5, BANK: 5}
N_SPLITS_B = 30                  # distinct seed splits cycled through (b)'s replicates
_ROOT = {}


def _pool(bank):
    return load_pool(os.path.join(_ROOT["root"], bank, "pool.npz"))


def _attr(bank, tda):
    return load_pool_tda(os.path.join(_ROOT["root"], bank, f"{tda}.npz"))


# -- (a, d, g) -------------------------------------------------------------- #
def _panel_a(job):
    """The R-sweep: per split, the law (rho_inf, lambda) from n_ref reference seeds and the
    points from R-seed subsets of the rest, averaged over splits (``n_ref=None``: halves)."""
    bank, si, tda, n_ref = job
    pool = _pool(bank)
    s2_fn = _seed_noise_fn(pool)
    n_ref = pool.R // 2 if n_ref is None else n_ref
    n_est = pool.R - n_ref
    grid = sorted({R for R in _shared.A_R_GRID[bank] if R <= n_est} | {n_est})
    g = predictions_tda(pool, _attr(bank, tda), _shared.E1_REPLICAS)  # E = 1: one model/unit
    r_fine = np.geomspace(1.0, float(n_est), 60)
    _, diag_full = corrected_pearson(g, pool.margins.mean(axis=0),
                                     s2_fn(pool.margins) / pool.R, clip=_shared.CLIP)
    lam_full = float(np.nanmedian(diag_full["lambda_q"] * pool.R))
    theory, rho_inf, lam_med = [], [], []
    naive = {R: [] for R in grid}
    corr = {R: [] for R in grid}
    for b in range(_shared.N_SUB):
        a_seeds, b_seeds = split(pool.R, b, n_ref)
        ma = pool.margins[a_seeds]
        rho_q, diag = corrected_pearson(g, ma.mean(axis=0), s2_fn(ma) / len(a_seeds),
                                        clip=_shared.CLIP)
        lam_q = diag["lambda_q"] * len(a_seeds)
        theory.append([float(battery(rho_q / np.sqrt(1.0 + lam_q / r))) for r in r_fine])
        rho_inf.append(float(battery(rho_q)))
        lam_med.append(float(np.nanmedian(lam_q)))
        rng = np.random.default_rng([_shared.SEED, si, b, len(bank), 1])
        for R in grid:
            seeds = b_seeds if R == len(b_seeds) else rng.choice(b_seeds, R, replace=False)
            sub = pool.margins[seeds]
            ybar = sub.mean(axis=0)
            naive[R].append(float(battery(_shared._colcorr(g, ybar))))
            if R >= 2:
                corr[R].append(float(battery(corrected_pearson(
                    g, ybar, s2_fn(sub) / R, clip=_shared.CLIP)[0])))
    rows = [{"R": R, "naive": float(np.mean(naive[R])),
             "naive_sd": float(np.std(naive[R], ddof=1)),
             "corrected": float(np.mean(corr[R])) if R >= 2 else None,
             "corrected_sd": float(np.std(corr[R], ddof=1)) if R >= 2 else 0.0}
            for R in grid]
    print(f"[fig2 a] {bank}/{tda} ({n_ref}/{n_est} split): naive {rows[0]['naive']:.4f} -> "
          f"{rows[-1]['naive']:.4f}, law at R={n_est} {np.mean(theory, axis=0)[-1]:.4f}",
          flush=True)
    return {"bank": bank, "tda": tda, "E": len(_shared.E1_REPLICAS),
            # title lambda: the bank's (all seeds), as quoted in the text; fits' mean beside it
            "lambda_median": lam_full, "lambda_median_fit": float(np.mean(lam_med)),
            "R": n_est, "rows": rows,
            "theory_R": r_fine.tolist(), "theory": np.mean(theory, axis=0).tolist(),
            "rho_inf": float(np.mean(rho_inf)), "splits": _shared.N_SUB}


# -- (b, e, h) -------------------------------------------------------------- #
def _panel_b(job):
    """The budget frontier against a rho_5 reference from seed half A, estimators from B."""
    bank, midx, tda, label = job
    pool = _pool(bank)
    s2_fn = _seed_noise_fn(pool)
    g = predictions_tda(pool, _attr(bank, tda), _shared.E1_REPLICAS)
    rank_g = stats.rankdata(g, axis=0)
    splits = [split(pool.R, k, pool.R // 2) for k in range(N_SPLITS_B)]
    refs = []
    for k, (a_seeds, _) in enumerate(splits):
        rng = np.random.default_rng([_shared.SEED, 99, k])
        n_groups = 1 if len(a_seeds) == _shared.TARGET_R else _shared.N_TRUTH
        refs.append(float(np.mean([_shared._spearman(
            g, pool.margins[rng.choice(a_seeds, _shared.TARGET_R, replace=False)]
            .mean(axis=0), rank_g) for _ in range(n_groups)])))
    series = []
    for si, (kind, r_seeds) in enumerate(_shared.SERIES_B):
        pts = []
        for m_masks in _shared.M_GRID[bank]:
            rng = np.random.default_rng([_shared.SEED, si, m_masks, len(bank), midx, 1])
            err = np.empty(_shared.B_SUB)
            for b in range(_shared.B_SUB):
                k = b % N_SPLITS_B
                masks = rng.choice(pool.M, m_masks, replace=False)
                seeds = rng.choice(splits[k][1], r_seeds, replace=False)
                err[b] = _shared._estimate_b(kind, r_seeds, g[masks],
                                             pool.margins[np.ix_(seeds, masks)], s2_fn) - refs[k]
            pts.append({"N": m_masks * r_seeds, "m": m_masks,
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "bias": float(err.mean()), "sd": float(err.std(ddof=1))})
        series.append({"kind": kind, "R": r_seeds, "points": pts})
    rng = np.random.default_rng([_shared.SEED, 7, 1])
    floor = [_shared._estimate_b("recon", 2, g, pool.margins[rng.choice(
        splits[b % N_SPLITS_B][1], 2, replace=False)], s2_fn) - refs[b % N_SPLITS_B]
        for b in range(_shared.B_SUB)]
    print(f"[fig2 b] {bank}/{label}: reference {np.mean(refs):.4f} "
          f"(sd {np.std(refs, ddof=1):.4f}); " + "  ".join(
              f"{s['kind']}{s['R']} {s['points'][-1]['rmse']:.4f}" for s in series),
          flush=True)
    return {"truth_rho5": float(np.mean(refs)), "truth_rho5_sd": float(np.std(refs, ddof=1)),
            "series": series, "target_R": _shared.TARGET_R, "attributor": tda,
            "E": len(_shared.E1_REPLICAS), "floor_bias": float(np.mean(floor)),
            "label": label,
            "reference": "rho_5 from seed half A (single five-seed group when A has five)"}


# -- (c, f, i) -------------------------------------------------------------- #
def _anchors(pool, ptda, members_of, margins):
    """The anchor grid: c and the corrected LDS of each K-member anchor."""
    anchors = []
    for K in [k for k in forecast.K_GRID if k < ptda.R_tda]:
        members = members_of(K)
        c_q, diag = member_noise_ratio(ptda.g[members])
        anchors.append({"K": K, "c": float(battery_c(c_q)),
                        "clipped_frac": float(diag["clipped_frac"]),
                        "rho_at_K": base._estimate(
                            "corrected", predictions_tda(pool, ptda, replicas=members),
                            margins),
                        "members": members})
    return anchors


def _series_c(bank, key, label, ptda, anchors, e_max, rows, sizes):
    ok = [a for a in anchors if a["clipped_frac"] <= forecast.CLIP_GATE]
    gated = (ok[0] if ok else anchors[-1])["K"]
    chosen = forecast.ANCHOR_OVERRIDE.get((bank, key), gated)
    for a in anchors:
        a["forecast_at_max"] = forecast._forecast(a["rho_at_K"], a["c"], a["K"], [e_max])[0]
    a = next(x for x in anchors if x["K"] == chosen)
    curve_e = [e for e in sizes if e >= chosen]
    print(f"[fig2 c] {bank}/{key}: anchor K={chosen}, measured to E={e_max}: forecast "
          f"{a['forecast_at_max']:.4f} vs {rows[-1]['mean']:.4f} "
          f"({a['forecast_at_max'] - rows[-1]['mean']:+.4f})", flush=True)
    return {"key": key, "label": label, "e_features": int(ptda.e_features),
            "E_max": e_max, "R_tda": int(ptda.R_tda), "sizes": sizes, "rows": rows,
            "target": rows[-1]["mean"], "anchors": anchors, "chosen_K": chosen,
            "gated_K": gated,
            "curve": forecast._forecast(a["rho_at_K"], a["c"], chosen, curve_e),
            "curve_E": curve_e}


def _panel_c(job):
    """Held out: anchors on the last K replicas (half A, split 0), earlier ones measured (B)."""
    bank, fname, label = job
    pool = _pool(bank)
    a_seeds, b_seeds = split(pool.R, 0, pool.R // 2)
    ptda = _attr(bank, fname)
    key = fname.replace("pool_tda_", "")
    with seed_noise(pool):
        anchors = _anchors(pool, ptda, lambda K: list(range(ptda.R_tda - K, ptda.R_tda)),
                           pool.margins[a_seeds])
        gated = next((a["K"] for a in anchors if a["clipped_frac"] <= forecast.CLIP_GATE),
                     anchors[-1]["K"])
        chosen = forecast.ANCHOR_OVERRIDE.get((bank, key), gated)
        e_max = ptda.R_tda - chosen              # measured never touch the anchors
        cap = forecast.E_SHOWN.get(bank, e_max)
        sizes = sorted({s for s in (*forecast.SIZES, cap) if s <= e_max} | {e_max})
        rows = []
        for size in sizes:
            groups = [list(range(k * size, (k + 1) * size))
                      for k in range(min(e_max // size, forecast.MAX_GROUPS))]
            vals = [base._estimate("corrected", predictions_tda(pool, ptda, replicas=gr),
                                   pool.margins[b_seeds]) for gr in groups]
            rows.append({"E": size, "n_groups": len(vals), "mean": float(np.mean(vals)),
                         "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None})
    return _series_c(bank, key, label, ptda, anchors, e_max, rows, sizes)


def _run(job):
    kind, *rest = job
    return {"a": _panel_a, "b": _panel_b, "c": _panel_c}[kind](tuple(rest))


def _build(banks_root):
    """All nine panels, each held out as the module docstring describes."""
    _ROOT["root"] = banks_root
    jobs = []
    for bank in ROWS:
        jobs += [("a", bank, si, tda, N_REF_A[bank])
                 for si, (tda, _, _, _) in enumerate(_shared.A_METHODS[bank])]
        jobs += [("b", bank, midx, tda, lab)
                 for midx, (tda, lab) in enumerate(_shared.B_METHODS[bank])]
        jobs += [("c", bank, fname, lab) for fname, lab, _, _ in forecast.METHODS[bank]]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1),
                             mp_context=ctx) as ex:
        res = dict(zip([j[:3] for j in jobs], ex.map(_run, jobs)))
    banks = {bank: bank_stamp(_pool(bank)) for bank in ROWS}
    d2 = {"a": {"series": [res[j[:3]] for j in jobs if j[0] == "a"]},
          "b": {bank: [res[j[:3]] for j in jobs if j[0] == "b" and j[1] == bank]
                for bank in ROWS},
          "banks": banks}
    d3 = {"panels": {bank: {"label": _shared.BANK_LABEL[bank],
                            "series": [res[j[:3]] for j in jobs
                                       if j[0].startswith("c") and j[1] == bank]}
                     for bank in ROWS},
          "banks": banks}
    return {"banks": banks, "fig2": d2, "fig3": d3,
            "params": {"rows": list(ROWS), "seed_split": "_heldout.split",
                       "n_sub_a": _shared.N_SUB, "B_sub_b": _shared.B_SUB,
                       "n_splits_b": N_SPLITS_B,
                       "a": f"law fitted on {N_REF_A} reference seeds, points from the "
                            f"rest, mean over splits",
                       "b": "rho_5 reference from seed half A, estimators from half B",
                       "c": "anchors = last K replicas with half A of split 0; measured = "
                            "earlier replicas with half B, disjoint-group means",
                       "m_grid": {b: _shared.M_GRID[b] for b in ROWS},
                       "arrow_rmse": _shared.ARROW_RMSE}}


def compute(banks_root):
    return _build(banks_root)


def _check_fit(fig, axes_c):
    """Do the gutter method labels stay inside their panel?  Warn, do not fail."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    worst = max((t.get_window_extent(renderer=r).x1 - ax.get_window_extent().x1
                 for ax in axes_c for t in ax.texts), default=-1.0)
    if worst > 0:
        print(f"[fig2] WARNING: method labels overflow the panel by "
              f"{worst:.1f} px — lower LABEL_FS_C or LABEL_X_C", flush=True)
    else:
        print(f"[fig2] method labels clear the panel edge by {-worst:.1f} px",
              flush=True)


def _c_top(d3, bank):
    """(c, f, i)'s top y limit: 5% over all ``forecast.draw_curves`` draws up to its E cap."""
    series = d3["panels"][bank]["series"]
    cap = forecast.E_SHOWN.get(bank, max(r["E"] for s in series for r in s["rows"]))
    vals = []
    for s in series:
        vals += [r["mean"] for r in s["rows"] if r["E"] <= cap]
        vals += [v for e, v in zip(s["curve_E"], s["curve"]) if e <= cap]
        vals += [a["rho_at_K"] for a in s["anchors"] if a["K"] == s["chosen_K"]]
    return 1.05 * max(vals)


def plot(data, out_path, draw_a, key_a, xlabel_a):
    """``fig2`` supplies column (a, d, g): ``draw_a(ax, data, bank)``, ``key_a``, ``xlabel_a``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d2, d3 = data["fig2"], data["fig3"]
    rows = data["params"]["rows"]
    _style.use_panel()
    plt.rcParams.update(TYPE)
    # canvas height from the rows: a row is as tall as column (a)'s box (ROW_ASPECT)
    left, right, wspace, ratios = 0.070, 0.988, 0.32, [1.75, 2.6, 1.75]
    col_avg = (right - left) * FIG_W / (len(ratios) + (len(ratios) - 1) * wspace)
    row_in = len(ratios) * col_avg * ratios[0] / sum(ratios) * ROW_ASPECT
    h = TOP_IN + BOTTOM_IN + len(rows) * row_in + (len(rows) - 1) * GAP_IN
    fig = plt.figure(figsize=(FIG_W, h))
    outer = fig.add_gridspec(len(rows), 3, width_ratios=ratios, left=left, right=right,
                             top=1.0 - TOP_IN / h, bottom=BOTTOM_IN / h, wspace=wspace,
                             hspace=GAP_IN / row_in)
    corners, grid, axes_c = [], [], []
    for row, bank in enumerate(rows):
        last_row = row == len(rows) - 1
        ax_a = fig.add_subplot(outer[row, 0])
        draw_a(ax_a, data, bank)
        ax_a.set_box_aspect(ROW_ASPECT)
        inner = outer[row, 1].subgridspec(1, 3, wspace=0.12)
        axes_b = [fig.add_subplot(inner[i]) for i in range(3)]
        for i, ax in enumerate(axes_b):
            _shared.draw_b(ax, d2, bank, i, show_x=True, show_ylabel=(i == 0))
        ax_c = fig.add_subplot(outer[row, 2])
        # from 0, set before drawing: draw_curves places its labels from the limits it finds
        ax_c.set_ylim(0.0, _c_top(d3, bank))
        forecast.draw_curves(ax_c, d3, bank, label_fs=LABEL_FS_C, label_x=LABEL_X_C)
        ax_c.set_box_aspect(ROW_ASPECT)
        ax_c.set_ylabel(r"corrected LDS $\rho^{\mathtt{s}}$", fontsize=YLABEL_FS_C)
        axes_c.append(ax_c)
        if last_row:
            ax_a.set_xlabel(xlabel_a)
            axes_b[1].set_xlabel(_shared._XLABEL_B)
            ax_c.set_xlabel(_XLABEL_C)
        ser = next(x for x in d2["a"]["series"] if x["bank"] == bank)
        ax_a.set_title(f"{_shared.BANK_LABEL[bank]} ($\\lambda$ = {ser['lambda_median']:.2f})",
                       loc="left", color=_shared.BANK_COLOR[bank], fontsize=TITLE_FS, pad=6)
        letters = "adgbehcfi"
        corners += [(ax_a, f"({letters[row]})"), (axes_b[0], f"({letters[3 + row]})"),
                    (ax_c, f"({letters[6 + row]})")]
        grid.append([ax_a, *axes_b, ax_c])

    key_b = [Line2D([], [], color=_shared.STYLE_B[k][0], linestyle=_shared.STYLE_B[k][1],
                    marker=_shared.STYLE_B[k][2], markersize=3.6,
                    linewidth=_shared.STYLE_B[k][3], label=_shared.STYLE_B[k][4])
             for k in _shared.SERIES_B]
    key_c = [Line2D([], [], color="#555555", linewidth=1.5, marker="o", markersize=3.4,
                    label="measured"),
             Line2D([], [], color="#bbbbbb", linewidth=4.5, label="forecast"),
             Line2D([], [], color="#555555", linestyle="none", marker="o", markersize=7,
                    markerfacecolor="none", markeredgewidth=1.3, label="anchor")]
    last = grid[-1]
    for handles, ncol, left, right, tag in zip(
            (key_a, key_b, key_c), (3, 2, 3), (last[0], last[1], last[4]),
            (last[0], last[3], last[4]), ("a, d, g", "b, e, h", "c, f, i")):
        # into the gutters either side, which are empty at the key's height
        x0 = max(left.get_position().x0 - KEY_PAD, 0.005)
        x1 = min(right.get_position().x1 + KEY_PAD, 0.995)
        _shared._column_legend(fig, handles, ncol, x0, x1 - x0, tag,
                            y=LEGEND_Y_IN / h, height=LEGEND_H_IN / h)

    # a row borrows the tick labels of the row below only where the two axes are identical
    for r in range(len(grid) - 1):
        for col in range(len(grid[r])):
            if grid[r][col].get_xlim() == grid[r + 1][col].get_xlim():
                grid[r][col].tick_params(axis="x", labelbottom=False)
    em = (LETTER_FS + LETTER_RAISE) / (fig.get_figheight() * 72.0)
    for ax, letter in corners:
        bb = ax.get_position()
        fig.text(bb.x0 - 0.048, bb.y1 + em, letter, ha="left", va="top", fontsize=LETTER_FS,
                 fontweight="bold")
    _check_fit(fig, axes_c)
    fig.savefig(out_path)
    fig.savefig(out_path[:-4] + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig2_panels", compute))
