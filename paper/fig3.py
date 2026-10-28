"""Figure 3: Pearson against Spearman LDS for many estimators and models, and the Gaussian curve.

One point per setting (bank, attribution method, ensemble of its first E members, readout
averaged over its first R seeds): the query-averaged Pearson and Spearman LDS over all
masks, a battery average with little noise.  Over this range ``h(r) = (6/pi) arcsin(r/2)``
is within 0.018 of the identity, so both panels take that diagonal out:

  (a)  Spearman - Pearson against Pearson; the curve ``h(r) - r`` is the Gaussian
       prediction (points translucent so it shows through).
  (b)  Spearman minus the expected sample Spearman of an exactly Gaussian pair with the
       same per-query correlations at the same M (``gaussian_spearman``, Moran 1948); zero
       means consistent with Gaussian.  This removes two offsets (a) keeps: the query
       average of h(r_q) is not h of the average (h is convex), and a Gaussian sample
       Spearman at finite M sits below h.  Large markers: the full ensemble, all seeds; their
       jackknife SEs (``resid_se``) stay undrawn, as the text claims only a size below 0.006.

Points within a bank share masks and seeds (coverage, not replication), and WikiText-2's
EK-FAC and SOURCE full points are near-copies; this checks the observed
(attribution, R-seed readout) pair, like figure 5.  Banks as banks-manifest.json pins them
(QNLI and WikiText-2 at M=400, R=20; WikiText-2 read out in eval mode, Q=481); ``--recompute``
needs them (about two minutes on CPU), rendering does not.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style
from _harness import bank_stamp, run

from corrected_lds import load_pool, load_pool_tda, pearson_to_spearman, predictions_tda
from corrected_lds.banks.artifacts import load_tauq, predictions

# bank -> methods; "tauq:" = a single tauq, not a pool_tda.  One TRAK per bank, figure 2's
# unit: TRAK-10 on CIFAR-10 (it has no TRAK-5), TRAK-5 elsewhere.
SETUPS = {
    "cifar10": ["ekfac", "source_L3", "trak10"],
    "qnli": ["ekfac", "tracin", "trak5"],
    # plus the bank's single-model SOURCE tauq
    "wikitext2_gpt2_bae": ["ekfac", "tracin", "trak5", "tauq:source_L3_e1"],
}
TRAK_UNIT = {"cifar10": 10, "qnli": 5, "wikitext2_gpt2_bae": 5}   # for the legend
E_GRID = (1, 2, 5, 10, 30)       # first E members; the full ensemble is always added
R_GRID = (1, 2, 5, 10)           # first R seeds; all seeds are always added
NB = 20                          # jackknife blocks over masks, for the full points
SEED = 20261009

BANK_COLOR = {"cifar10": "#1b6ca8", "qnli": "#c44536", "wikitext2_gpt2_bae": "#2f7d4a"}
BANK_NAME = {"cifar10": "CIFAR-10 / ResNet-9", "qnli": "QNLI / BERT",
             "wikitext2_gpt2_bae": "WikiText-2 / GPT-2"}
FAMILY = {"trak": "TRAK", "ekfac": "EK-FAC", "source": "SOURCE", "tracin": "TracIn"}
MARKER = {"EK-FAC": "s", "TRAK": "o", "SOURCE": "D", "TracIn": "v"}

# Drawn at its printed width (a 0.5\textwidth wrapfigure, 2.75 in), so the type is as printed;
# inches for frames, axes units for the rest; one y-label offset, (b)'s lowered clear of (a)'s
FRAME_W, FRAME_H, GAP = 1.78, 1.3, 0.15
LEFT = 0.95                  # left of the frames: letters, y labels, tick labels
TYPE = {"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5}
LETTER_FS = 9.5
NOTE_FS = 7.5
YLABEL_X = -0.311
YLABEL_Y = (0.5, 0.42)       # (a), (b)
LETTER_X = -0.531
LEGEND_AT = (0.235, -0.3)


def _family(method):
    return next(f for prefix, f in FAMILY.items() if method.startswith(prefix))


def gaussian_spearman(r, n):
    """Expected sample Spearman of a bivariate normal pair whose sample Pearson over n
    points is ``r`` (first-order Pearson debias, then Moran's exact formula)."""
    r = np.clip(np.asarray(r, dtype=np.float64), -1.0, 1.0)
    rho = np.clip(r * (1.0 + (1.0 - r * r) / (2.0 * n)), -1.0, 1.0)
    return 6.0 / (np.pi * (n + 1)) * (np.arcsin(rho) + (n - 2) * np.arcsin(rho / 2.0))


def _z(x):
    x = x - x.mean(axis=0)
    return x / np.sqrt((x * x).sum(axis=0))


def _point(g, y):
    """Per-query Pearson and Spearman over the masks (rows) -> battery numbers."""
    p = (_z(g) * _z(y)).sum(axis=0)
    s = (_z(stats.rankdata(g, axis=0)) * _z(stats.rankdata(y, axis=0))).sum(axis=0)
    n = g.shape[0]
    return {"P": float(p.mean()), "S": float(s.mean()),
            "H": float(pearson_to_spearman(p).mean()),
            "G": float(gaussian_spearman(p, n).mean())}


def _jackknife(g, y, blocks):
    """Delete-a-block jackknife SE of the residual S - G over mask blocks."""
    vals = []
    for b in blocks:
        keep = np.ones(g.shape[0], bool)
        keep[b] = False
        pt = _point(g[keep], y[keep])
        vals.append(pt["S"] - pt["G"])
    vals = np.asarray(vals)
    return float(np.sqrt((len(vals) - 1) / len(vals) * np.sum((vals - vals.mean()) ** 2)))


def _ensembles(banks_root, bank, method, pool):
    if method.startswith("tauq:"):
        name = method[len("tauq:"):]
        tau = load_tauq(os.path.join(banks_root, bank, f"tauq_{name}.npz"))
        # suffixed so a tauq never shares a method name with a pool_tda of the same bank
        return f"{name}_tauq", {1: predictions(pool, tau)}
    pt = load_pool_tda(os.path.join(banks_root, bank, f"pool_tda_{method}.npz"))
    es = sorted({e for e in E_GRID if e < pt.R_tda} | {pt.R_tda})
    return method, {e: predictions_tda(pool, pt, list(range(e))) for e in es}


def compute(banks_root):
    banks, points = {}, []
    for bank, methods in SETUPS.items():
        pool = load_pool(os.path.join(banks_root, bank, "pool.npz"))
        banks[bank] = bank_stamp(pool)
        rs = sorted({r for r in R_GRID if r < pool.R} | {pool.R})
        ys = {r: pool.margins[:r].mean(axis=0) for r in rs}
        perm = np.random.default_rng([SEED, pool.M]).permutation(pool.M)
        blocks = np.array_split(perm, NB)
        for method in methods:
            name, gs = _ensembles(banks_root, bank, method, pool)
            e_full = max(gs)
            for e, g in gs.items():
                for r, y in ys.items():
                    pt = {"bank": bank, "method": name, "family": _family(name), "E": e,
                          "R": r, "M": pool.M, "full": bool(e == e_full and r == pool.R),
                          **_point(g, y)}
                    if pt["full"]:
                        pt["resid_se"] = _jackknife(g, y, blocks)
                    points.append(pt)
                    print(f"[fig3] {bank:18s} {name:9s} E={e:3d} "
                          f"R={r:2d}  P={pt['P']:.4f}  S-P={pt['S'] - pt['P']:+.4f}  "
                          f"S-G={pt['S'] - pt['G']:+.4f}"
                          + (f" +- {pt['resid_se']:.4f}" if pt["full"] else ""), flush=True)
        del pool
    return {"banks": banks, "points": points,
            "params": {"setups": SETUPS, "E_grid": list(E_GRID) + ["all"],
                       "R_grid": list(R_GRID) + ["all"],
                       "selection": "first E members (replica order), first R seeds",
                       "P, S": "query-averaged Pearson / Spearman over all masks",
                       "H": "query-averaged (6/pi) arcsin(r_q / 2)",
                       "G": "query-averaged finite-M Gaussian expected Spearman "
                            "(gaussian_spearman)",
                       "resid_se": f"delete-a-block jackknife of S - G over {NB} mask "
                                   f"blocks (seed [{SEED}, M]), full points only"}}


def plot(data, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style.use()
    plt.rcParams.update(TYPE)
    pts = data["points"]
    w, h = LEFT + FRAME_W + 0.1, 0.1 + 2 * FRAME_H + GAP + 0.5
    fig, (b, c) = plt.subplots(
        2, 1, figsize=(w, h), sharex=True,
        gridspec_kw={"left": LEFT / w, "right": (LEFT + FRAME_W) / w, "top": 1 - 0.1 / h,
                     "bottom": 0.5 / h, "hspace": GAP / FRAME_H})
    hi = 0.72
    xs = np.linspace(0.0, hi, 300)
    pred = {"color": "#222222", "linewidth": 1.2, "zorder": 2}
    for ax in (b, c):
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.set_xlim(0.0, hi)
    c.set_xlabel("Pearson LDS")

    b.plot(xs, pearson_to_spearman(xs) - xs, **pred)
    b.set_ylim(-0.0215, 0.001)
    b.set_ylabel("Spearman $-$ Pearson")
    x0 = 0.36                    # label the curve from the empty lower-left corner
    b.annotate("Gaussian\nprediction", xy=(x0, float(pearson_to_spearman(x0)) - x0),
               xytext=(0.05, -0.0172), fontsize=NOTE_FS, color="#222222", ha="left",
               va="center", arrowprops={"arrowstyle": "-", "color": "#222222",
                                        "linewidth": 0.6, "shrinkB": 2})

    c.axhline(0.0, color="#222222", linewidth=1.0, zorder=2)
    c.set_ylim(-0.0017, 0.0066)  # holds every point
    c.set_yticks([0.0, 0.002, 0.004, 0.006])
    c.set_ylabel("Spearman $-$ prediction")

    for p in pts:
        kw = {"marker": MARKER[p["family"]], "color": BANK_COLOR[p["bank"]],
              "linestyle": "none", "markeredgecolor": "white", "markeredgewidth": 0.4,
              "zorder": 3}
        small = dict(kw, markersize=3.3, alpha=0.85)
        b.plot(p["P"], p["S"] - p["P"], **dict(small, alpha=0.4, markeredgewidth=0))
        if p["full"]:
            c.plot(p["P"], p["S"] - p["G"], markersize=5.0, **dict(kw, zorder=4))
        else:
            c.plot(p["P"], p["S"] - p["G"], **dict(small, alpha=0.35))

    for ax, letter, y in zip((b, c), ("(a)", "(b)"), YLABEL_Y):
        ax.yaxis.set_label_coords(YLABEL_X, y)    # after set_ylabel, which resets y
        _style.panel_label(ax, letter, fontsize=LETTER_FS, x=LETTER_X)

    # three-row legend filled by column: models first, then methods two by two, padded apart
    blank = Line2D([], [], linestyle="none", label=" ")
    banks = [k for k in BANK_COLOR if k in data["banks"]]
    handles = [Line2D([], [], color=BANK_COLOR[k], marker="o", linestyle="none",
                      markersize=4.5, label=BANK_NAME[k]) for k in banks]
    units = sorted({TRAK_UNIT[k] for k in banks})
    trak = "TRAK-" + "/".join(map(str, units))            # "TRAK-5/10": named by unit
    methods = [Line2D([], [], color="#555555", marker=MARKER[f], linestyle="none",
                      markersize=4.5, label=trak if f == "TRAK" else f) for f in MARKER]
    handles += [blank] * (-len(handles) % 3)
    for i in range(0, len(methods), 2):
        handles += methods[i:i + 2] + [blank] * (3 - len(methods[i:i + 2]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=LEGEND_AT,
               bbox_transform=c.transAxes, ncol=len(handles) // 3,
               handlelength=1.0, handletextpad=0.35, columnspacing=0.8)
    fig.savefig(out_path, pad_inches=0.02)        # the page is the printed width
    fig.savefig(out_path[:-4] + ".png", dpi=300, pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run("fig3", compute, plot))
