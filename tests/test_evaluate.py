"""The evaluate result is a versioned, deterministic, JSON-able contract, and the CLI writes it."""

import json
import os

import numpy as np
import pytest
from scipy import stats

from corrected_lds.banks.artifacts import load_pool, load_tauq, predictions
from corrected_lds.banks.presets import Resnet9Recipe, bank_hashes
from corrected_lds.cli import main as cli_main
from corrected_lds.evaluate import RESULT_SCHEMA, evaluate_pool
from corrected_lds.pools.runner import PoolTask, build_pool
from corrected_lds.rank_debias import unbiased_spearman

N, Q = 40, 6
TRAIN_HASH, QUERY_HASH = bank_hashes("cifar10")


def _bank(tmp_path, M=8, R=3):
    recipe = Resnet9Recipe(epochs=1)
    rng = np.random.default_rng(4)
    noise = {}

    def train_cell(data, sub, seed):
        return (int(seed), np.asarray(sub))

    def readout(model, data):
        seed, sub = model
        if seed not in noise:
            noise[seed] = np.random.default_rng(seed).normal(0, 0.3, Q)
        base = np.sin(np.arange(Q) + sub.sum() * 1e-3)         # mask-dependent signal
        return base + noise[seed]

    task = PoolTask(recipe=recipe, train_cell=train_cell, readout=readout)
    data = {"meta": {"n_train": N, "n_targets": Q, "dataset": "cifar10",
                     "train_idx": list(range(N)), "query_idx": list(range(Q)),
                     "train_hash": TRAIN_HASH, "query_hash": QUERY_HASH}}
    d = str(tmp_path / "bank")
    build_pool(d, task, data, M=M, R=R, verbose=False)
    tauq = rng.normal(size=(N, Q)).astype(np.float32)
    np.savez(os.path.join(d, "tauq_trak_e4"), tauq=tauq, dataset="cifar10", n_train=N,
             n_targets=Q, train_idx=np.arange(N), query_idx=np.arange(Q),
             train_hash=TRAIN_HASH, query_hash=QUERY_HASH,
             n_models=4, proj_dim=512, provenance=json.dumps({"method": "trak"}))
    return d


def test_evaluate_pool_contract(tmp_path):
    d = _bank(tmp_path)
    pool = load_pool(os.path.join(d, "pool.npz"))
    tau = load_tauq(os.path.join(d, "tauq_trak_e4.npz"))
    res = evaluate_pool(pool, tau)
    assert res["schema"] == RESULT_SCHEMA
    r = res["results"]
    for key in ("naive_rho1", "corrected_rho_inf", "corrected_se_jackknife",
                "bridged_rho_inf", "reconstructed_rho_R", "sigma2_mean",
                "lambda_median", "floored_frac", "spearman", "debiased_spearman",
                "debiased_noise_corrected_spearman"):
        assert key in r, key
    assert set(res["params"]) == {"clip", "level", "soft_threshold_k"}
    assert set(r["reconstructed_rho_R"]) == {2, 5, 10}
    assert -1 <= r["naive_rho1"] <= 1 and -1 <= r["corrected_rho_inf"] <= 1

    g = predictions(pool, tau)
    ybar = pool.margins.mean(axis=0)
    sample_s = np.array([stats.spearmanr(g[:, q], ybar[:, q]).statistic for q in range(Q)])
    left, right = np.triu_indices(pool.M, 1)
    tau_a = np.mean(
        np.sign(g[left] - g[right]) * np.sign(ybar[left] - ybar[right]), axis=0)
    expected_u_q = unbiased_spearman(sample_s, tau_a, pool.M)
    expected_u = float(np.mean(expected_u_q))
    assert r["spearman"] == pytest.approx(sample_s.mean())
    assert r["debiased_spearman"] == pytest.approx(expected_u)

    raw_p = np.mean([stats.pearsonr(g[:, q], ybar[:, q]).statistic for q in range(Q)])
    noise_factor = r["corrected_rho_inf"] / raw_p
    latent_p = noise_factor * 2.0 * np.sin(np.pi * np.clip(expected_u, -1, 1) / 6.0)
    expected_combined = 6.0 / np.pi * np.arcsin(np.clip(latent_p, -1, 1) / 2.0)
    assert r["debiased_noise_corrected_spearman"] == pytest.approx(expected_combined)

    assert res["diagnostics"]["rank_noise_factor"] == pytest.approx(noise_factor)
    json.dumps(res)                                           # JSON-able end to end
    assert evaluate_pool(pool, tau) == res                    # deterministic


def test_soft_threshold_fixed_k():
    """Eq. 16 with the fixed-k rule: at most k nonzeros per query, survivors shrunk by
    that query's lambda; k >= n is the identity; k < 1 refused."""
    from corrected_lds.evaluate import soft_threshold_fixed_k
    tauq = np.random.default_rng(0).normal(size=(50, 3))
    out = soft_threshold_fixed_k(tauq, 5)
    nz = (out != 0).sum(axis=0)
    assert out.shape == tauq.shape and (nz <= 5).all() and (nz > 0).all()
    for q in range(3):                          # survivors = top-|tau|, shrunk by lam_q
        order = np.argsort(-np.abs(tauq[:, q]))
        lam = np.abs(tauq[order[5], q])
        np.testing.assert_allclose(
            out[order[:5], q],
            np.sign(tauq[order[:5], q]) * (np.abs(tauq[order[:5], q]) - lam))
        assert (out[order[5:], q] == 0).all()
    np.testing.assert_array_equal(soft_threshold_fixed_k(tauq, 50), tauq)
    with pytest.raises(ValueError, match="k >= 1"):
        soft_threshold_fixed_k(tauq, 0)


def test_evaluate_pool_applies_and_stamps_the_threshold(tmp_path):
    """soft_threshold_k reaches every number and is stamped, even as None, so results reproduce."""
    d = _bank(tmp_path)
    pool = load_pool(os.path.join(d, "pool.npz"))
    tau = load_tauq(os.path.join(d, "tauq_trak_e4.npz"))
    base = evaluate_pool(pool, tau)
    res = evaluate_pool(pool, tau, soft_threshold_k=5)
    assert base["params"]["soft_threshold_k"] is None
    assert res["params"]["soft_threshold_k"] == 5
    assert res["results"]["naive_rho1"] != base["results"]["naive_rho1"]
    assert evaluate_pool(pool, tau, soft_threshold_k=5) == res  # deterministic
    json.dumps(res)


def test_evaluate_cli_writes_json(tmp_path):
    d = _bank(tmp_path)
    out = str(tmp_path / "result.json")
    assert cli_main(["evaluate", d, "--tauq", "trak_e4", "--json", out]) == 0
    with open(out) as fh:
        res = json.load(fh)
    assert res["schema"] == RESULT_SCHEMA
    assert "debiased_noise_corrected_spearman" in res["results"]
