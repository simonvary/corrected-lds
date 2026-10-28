"""Mathematical checks for the finite-mask Spearman corrections."""

import itertools

import numpy as np
import pytest
from scipy import stats

from corrected_lds.rank_debias import (
    finite_population_unbiased_spearman,
    unbiased_spearman,
)


def test_unbiased_formula_and_finite_population_endpoint():
    spearman = np.array([1.0, -1.0, 0.2])
    kendall = np.array([1.0, -1.0, 0.1])
    np.testing.assert_allclose(
        unbiased_spearman(spearman, kendall, 4), [1.0, -1.0, 0.35]
    )
    np.testing.assert_allclose(
        finite_population_unbiased_spearman(spearman, kendall, 4, 4), spearman
    )
    with pytest.raises(ValueError, match="at least three"):
        unbiased_spearman([0.0], [0.0], 2)


def test_finite_population_estimator_is_exact_over_all_panels():
    predicted = np.array([0.2, 1.1, -0.4, 0.7, 2.0])
    observed = np.array([0.5, -0.2, -1.0, 1.4, 0.9])
    target = stats.spearmanr(predicted, observed).statistic
    estimates = []
    for panel in itertools.combinations(range(len(predicted)), 3):
        panel = np.asarray(panel)
        spearman = stats.spearmanr(predicted[panel], observed[panel]).statistic
        kendall = stats.kendalltau(predicted[panel], observed[panel]).statistic
        estimates.append(
            finite_population_unbiased_spearman(spearman, kendall, 3, len(predicted))
        )
    assert np.mean(estimates) == pytest.approx(target, abs=2e-15)
