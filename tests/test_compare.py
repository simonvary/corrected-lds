"""``corrected-lds compare`` and ``--consistency`` on a tiny synthetic bank."""

import numpy as np

from corrected_lds.banks.artifacts import load_pool, write_pool_tda, write_tauq
from corrected_lds.banks.presets import Resnet9Recipe, bank_hashes
from corrected_lds.cli import main
from corrected_lds.pools.runner import PoolTask, build_pool


def _pool(tmp_path, name, M, R, query_idx=(5, 1, 9)):
    th, qh = bank_hashes("cifar10")
    meta = {"n_train": 40, "n_targets": 3, "dataset": "cifar10", "train_idx": list(range(40)),
            "query_idx": list(query_idx), "train_hash": th, "query_hash": qh}
    task = PoolTask(recipe=Resnet9Recipe(epochs=1),
                    train_cell=lambda d, sub, seed: (int(seed), np.asarray(sub)),
                    readout=lambda m, d: np.sin(np.arange(3) + m[0]) + m[1].sum() * 1e-3)
    return build_pool(str(tmp_path / name), task, {"meta": meta}, M=M, R=R, verbose=False)


def test_compare_and_consistency(tmp_path):
    bank = _pool(tmp_path, "bank", M=6, R=3)
    assert main(["compare", _pool(tmp_path, "prefix", M=4, R=2), bank]) == 0    # bitwise
    assert main(["compare", _pool(tmp_path, "q", M=6, R=3, query_idx=(5, 1, 8)), bank]) == 1
    pool = load_pool(bank)
    parts = [np.random.default_rng(k).integers(-5, 6, (40, 3)).astype(float) for k in range(3)]
    write_pool_tda(str(tmp_path / "bank" / "pool_tda_ekfac.npz"),
                   np.stack([pool.subsets @ p for p in parts]), pool, subsets=pool.subsets,
                   method="ekfac", inner="single-member", e_features=1, seed_base=0,
                   provenance={})
    tauq = str(tmp_path / "bank" / "tauq_ekfac_e2.npz")
    write_tauq(tauq, (parts[0] + parts[1]) / 2, pool, provenance={})
    assert main(["compare", str(tmp_path / "bank"), "--consistency"]) == 0
    write_tauq(tauq, parts[2], pool, provenance={})                            # tampered
    assert main(["compare", str(tmp_path / "bank"), "--consistency"]) == 1
