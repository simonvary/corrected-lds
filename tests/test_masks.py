"""Pinned masks: the sha256 values pin every released bank's masks; if one moves, the sampler no
longer reproduces the released banks, a release-breaking bug, never a test to update.
"""

import hashlib
import zlib

import numpy as np
import pytest

from corrected_lds.banks.masks import sample_subsets


def _sha16(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def test_rng_stream_smoke():
    """The (crc32(b'subsets'), seed) RNG key produces the pinned stream."""
    r = np.random.default_rng((zlib.crc32(b"subsets"), 777))
    np.testing.assert_allclose(
        r.random(3), [0.47686738, 0.60164512, 0.53040839], atol=1e-8)


# (M, n, sha256/16) of each released bank's masks (alpha=0.5, seed 777), as stamped in its pool.npz
RELEASED_MASKS = {
    "cifar10": (300, 50000, "5a80022843ba3099"),
    "mnist": (300, 6144, "b996c8930aea1dc7"),
    "qnli": (400, 50000, "e1af8c170d91a981"),
    "wikitext2_gpt2_bae": (400, 4656, "818d4635b31143a6"),
    "imagenet": (100, 1281167, "00f4f4d30bb3540b"),
}


@pytest.mark.parametrize("bank", sorted(RELEASED_MASKS))
def test_released_bank_masks(bank):
    M, n, digest = RELEASED_MASKS[bank]
    m = sample_subsets(M, n, 0.5, 777)
    assert m.dtype == np.int8 and m.shape == (M, n)
    assert _sha16(m) == digest
    # fixed-size subsets: every row has exactly k = floor(0.5 * n) ones
    assert np.all(m.sum(axis=1) == n // 2)


def test_masks_are_prefix_stable_in_M():
    """Growing M only appends rows (qnli's 400 start with cifar10's 300), so M-extension is valid."""
    m400 = sample_subsets(400, 50000, 0.5, 777)
    np.testing.assert_array_equal(m400[:300], sample_subsets(300, 50000, 0.5, 777))
    np.testing.assert_array_equal(m400[:100], sample_subsets(100, 50000, 0.5, 777))


def test_row_sums_at_10k_rows():
    m = sample_subsets(20, 10000, 0.5, 777)
    assert np.all(m.sum(axis=1) == 5000)


def test_alpha_edges_and_validation():
    assert not sample_subsets(3, 10, 0.0, 1).any()
    assert sample_subsets(3, 10, 1.0, 1).all()
    with pytest.raises(ValueError):
        sample_subsets(3, 10, -0.2, 1)
