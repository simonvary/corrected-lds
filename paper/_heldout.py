"""Held-out seed splits shared by the figures.

Each estimate is scored against a reference (a truth, a fitted law or a forecast anchor)
built from seeds it never touches: half A builds the reference, half B feeds the
estimators; forecast anchors are the last K replicas.  Masks are shared (finite-population
targets).  Split b is the same in every figure, so all agree on which seeds are held out.
"""

import contextlib
import os
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rmse_corrected as base

from corrected_lds.evaluate import _seed_noise_fn

SEED = 0


def split(R, b, n_ref):
    """Replicate b's split ``(A, B)`` of R seeds, each sorted: ``n_ref`` seeds in A (the
    reference), the rest in B; every split size cuts b's one shuffled order."""
    perm = np.random.default_rng([SEED, 7, b]).permutation(R)
    return np.sort(perm[:n_ref]), np.sort(perm[n_ref:])


def halves(R, b):
    """Replicate b's seed halves ``(A, B)``, ``R // 2`` seeds each (an odd seed goes to B)."""
    return split(R, b, R // 2)


@contextlib.contextmanager
def seed_noise(pool):
    """fig1(a)'s estimator (``_rmse_corrected._estimate``) with the pool's own seed-noise
    estimator (stacked, or centred if shared-per-replica), chosen as ``evaluate`` does."""
    with mock.patch.object(base, "seed_noise_variance_stacked", _seed_noise_fn(pool)):
        yield
