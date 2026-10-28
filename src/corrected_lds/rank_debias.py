"""Finite-mask debiasing for ordinary Spearman LDS.

For ``m`` IID continuous pairs with population Spearman ``s`` and Kendall ``tau``,
``E[r_m] = ((m - 2) s + 3 tau) / (m + 1)``; a U-statistic of the observed Spearman and
Kendall tau-a removes this bias exactly, with no Gaussian assumption.  Seed noise is
corrected separately, by :mod:`corrected_lds.estimator`.
"""

import numpy as np

__all__ = [
    "finite_population_unbiased_spearman",
    "unbiased_spearman",
]


def unbiased_spearman(sample_spearman, sample_kendall, masks: int) -> np.ndarray:
    """Unbiased order-three estimator of population Spearman correlation.

    ``((m + 1) r_m - 3 t_m) / (m - 2)`` for sample Spearman ``r_m`` and Kendall tau-a
    ``t_m``; deliberately unclipped, since clipping would destroy unbiasedness.
    """
    if masks < 3:
        raise ValueError("the unbiased Spearman estimator requires at least three masks")
    spearman = np.asarray(sample_spearman, dtype=float)
    kendall = np.asarray(sample_kendall, dtype=float)
    if spearman.shape != kendall.shape:
        raise ValueError("sample Spearman and Kendall arrays must have equal shapes")
    return ((masks + 1.0) * spearman - 3.0 * kendall) / (masks - 2.0)


def finite_population_unbiased_spearman(
    sample_spearman,
    sample_kendall,
    masks: int,
    population_size: int,
) -> np.ndarray:
    """Unbiasedly estimate a finite bank's full-sample Spearman correlation.

    The without-replacement version, for a reference that is the complete retained bank;
    it reduces exactly to sample Spearman when ``masks == population_size``.
    """
    if population_size < masks:
        raise ValueError("population_size must be at least the sampled mask count")
    population = unbiased_spearman(sample_spearman, sample_kendall, masks)
    kendall = np.asarray(sample_kendall, dtype=float)
    return ((population_size - 2.0) * population + 3.0 * kendall) / (
        population_size + 1.0
    )
