"""Anchors measured on the sha-verified released banks, on which the paper's numbers rest.
Set ``CORRECTED_LDS_BANKS`` to a ``corrected-lds fetch-banks`` root (holding ``cifar10/``,
``qnli/``, ...), else skipped.  Floats are pinned to rel 1e-9: any CPU or BLAS agrees that far.
"""

import os

import pytest

from corrected_lds.banks.artifacts import load_pool, load_pool_tda, load_tauq
from corrected_lds.banks.compare import consistency
from corrected_lds.banks.hub import validate_bank
from corrected_lds.evaluate import evaluate_pool, evaluate_pool_tda
from corrected_lds.tda_noise import battery_c, member_noise_ratio

ROOT = os.environ.get("CORRECTED_LDS_BANKS")
pytestmark = pytest.mark.skipif(not ROOT, reason="set CORRECTED_LDS_BANKS to a fetch-banks root")


def _path(bank, name):
    return os.path.join(ROOT, bank, name)


def _approx(x):
    return pytest.approx(x, rel=1e-9)


@pytest.mark.parametrize("bank", ["cifar10", "mnist", "qnli", "wikitext2_gpt2_bae", "imagenet"])
def test_released_bank_validates(bank):
    assert "pool.npz" in validate_bank(os.path.join(ROOT, bank))


@pytest.mark.parametrize("bank,pairs", [("cifar10", 4), ("qnli", 6), ("wikitext2_gpt2_bae", 6)])
def test_plain_mean_tauqs_are_their_pool_tda_means(bank, pairs):
    # every pair agreed to <= 3e-8 rel; mnist ships no pool_tda, imagenet only its pool
    rows = consistency(os.path.join(ROOT, bank))
    assert len(rows) == pairs and all(r["ok"] for r in rows), rows


def test_cifar10_ekfac_lds():
    pool = load_pool(_path("cifar10", "pool.npz"))
    res = evaluate_pool_tda(pool, load_pool_tda(_path("cifar10", "pool_tda_ekfac.npz")))["results"]
    assert res["debiased_noise_corrected_spearman"] == _approx(0.2352109562988105)
    assert res["lambda_median"] == _approx(3.4301459729446515)


@pytest.mark.parametrize("bank,lam", [("qnli", 0.18777943860856183),
                                      ("wikitext2_gpt2_bae", 0.38931467336615855)])
def test_seed_noise_lambda(bank, lam):
    # WikiText-2's pool is shared-per-replica, so it also exercises the centred seed noise.
    pool = load_pool(_path(bank, "pool.npz"))
    res = evaluate_pool_tda(pool, load_pool_tda(_path(bank, "pool_tda_ekfac.npz")))["results"]
    assert res["lambda_median"] == _approx(lam)


def test_trak1_member_noise():
    g = load_pool_tda(_path("cifar10", "pool_tda_trak1.npz")).g
    c, diag = member_noise_ratio(g[:2])
    assert battery_c(c) == _approx(108.06347996506472)
    assert diag["clipped_frac"] == 0.4345
    # Figure 1(b)'s TRAK-1 forecast anchor: the last five replicas
    c5, _ = member_noise_ratio(g[95:100])
    assert battery_c(c5) == _approx(123.12909872512631)


@pytest.mark.parametrize("bank,tauq,corrected", [
    ("cifar10", "tauq_trak10_e10", 0.5826281155825361),
    ("wikitext2_gpt2_bae", "tauq_ekfac_e10", 0.49311716137404854),
    ("wikitext2_gpt2_bae", "tauq_trak5_e10", 0.23799436401053867)])
def test_headline_corrected_lds(bank, tauq, corrected):
    pool = load_pool(_path(bank, "pool.npz"))
    res = evaluate_pool(pool, load_tauq(_path(bank, f"{tauq}.npz")))["results"]
    assert res["corrected_rho_inf"] == _approx(corrected)
    if tauq == "tauq_trak10_e10":
        assert res["naive_rho1"] == _approx(0.2858955984168465)
