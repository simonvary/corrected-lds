"""One place for the paper's matplotlib style — every figure calls ``use()`` first."""

import matplotlib

matplotlib.use("Agg")                       # render headless; scripts write PDFs only
import matplotlib.pyplot as plt


def use():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def use_panel():
    """``use()`` plus a larger type scale, for the panel figures."""
    use()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 10})


def panel_label(ax, letter, fontsize=14, x=-0.27):
    """The bold corner letter — left of the y-axis, top-aligned with the frame."""
    ax.text(x, 1.0, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold")
