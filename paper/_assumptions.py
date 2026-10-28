"""Bivariate-normality helpers for ``fig5``'s bridge checks and their stages: the squared
Mahalanobis distance, its exact-null Q-Q grid, the banks and the negative control.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CONTROL_P = 0.7                  # the shared negative control: y -> sign(y) |y|^p
QQ_N = 180                       # Q-Q grid points, spaced evenly along the axis
QQ_P_MAX = 0.999                 # deepest quantile resolved (see _qq_grid)

BANKS = [
    {"key": "cifar10", "label": "CIFAR-10 / ResNet-9", "color": "#1b6ca8",
     "tda": "pool_tda_ekfac"},
    {"key": "qnli", "label": "QNLI / BERT", "color": "#c44536",
     "tda": "pool_tda_trak5"},
]


# -- shared numerics (column-wise, ddof=1 — the package convention) --------- #


def _mahalanobis2(a, b):
    """Squared Mahalanobis distance of the bivariate pair, per query then pooled; each query
    has its own 2x2 sample covariance, so queries need not share a scale."""
    m = a.shape[0]
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    saa = (ac * ac).sum(axis=0) / (m - 1)
    sbb = (bc * bc).sum(axis=0) / (m - 1)
    sab = (ac * bc).sum(axis=0) / (m - 1)
    return ((sbb * ac**2 - 2 * sab * ac * bc + saa * bc**2) / (saa * sbb - sab**2)).ravel()


def _qq_grid(m):
    """The Q-Q evaluation grid at m masks: ``(probabilities, null quantiles)``.

    The null is exact, ``D2 * m / (m-1)^2 ~ Beta(p/2, (m-p-1)/2)`` with p=2; its chi^2_2
    limit is off by more than the measured deviation (0.50 at m=300, 0.75 at m=200).  Its
    mean ``2(m-1)/m`` holds for any data, so matching it is no evidence.  Points are spaced
    evenly in the quantile, not in probability, so the drawn axis is evenly resolved."""
    b, c = (m - 3) / 2.0, (m - 1) ** 2 / m
    q = np.linspace(c * stats.beta.ppf(0.002, 1.0, b),
                    c * stats.beta.ppf(QQ_P_MAX, 1.0, b), QQ_N)
    return stats.beta.cdf(q / c, 1.0, b), q


def _control(y, p=CONTROL_P):
    """A monotone map of the readout: every Spearman correlation is untouched, so whatever a
    check picks up under it is a pure normality effect, the arcsin bridge's failure mode."""
    z = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    return np.sign(z) * np.abs(z) ** p


