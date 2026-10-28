"""pool_tda CLI wiring (needs torch; traker/kronfluence stubbed): ``compute-tau``'s Eq. 15 unit
slices are the separated average over exactly their seed block, its plain-mean member slices
``subsets @ (s_m * q_m)``, and each tauq is named by its replicas, not its models.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import corrected_lds.tda.trak as trakmod
from corrected_lds import cli
from corrected_lds.banks.artifacts import load_pool, load_pool_tda, predictions_tda
from corrected_lds.banks.presets import PRESETS, bank_hashes
from corrected_lds.pools.runner import PoolTask, build_pool

N, Q, M_POOL, R_POOL = 10_000, 8, 3, 2


def _meta(p):
    th, qh = bank_hashes(p["dataset"])
    return {"dataset": p["dataset"], "n_train": N, "n_targets": Q,
            "n_classes": 2, "in_ch": 3,
            "train_idx": list(range(N)), "query_idx": list(range(Q)),
            "train_hash": th, "query_hash": qh}


def _build_pool(tag_dir, meta):
    from corrected_lds.banks.presets import Resnet9Recipe
    task = PoolTask(recipe=Resnet9Recipe(epochs=1),
                    train_cell=lambda d, sub, seed: (int(seed), np.asarray(sub)),
                    readout=lambda m, d: np.arange(Q, dtype=float) + m[0] * 1e-3)
    build_pool(str(tag_dir), task, {"meta": dict(meta)}, M=M_POOL, R=R_POOL,
               verbose=False)
    return load_pool(os.path.join(tag_dir, "pool.npz"))


def _fake_train_ensemble(ckpt_dir, data_, recipe_, n_models, train_fn, verbose=True,
                         members=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    paths = [os.path.join(ckpt_dir, f"member_{m:03d}.pt") for m in range(n_models)]
    for path in paths:
        torch.save({"w": torch.zeros(1)}, path)
    return paths


def _member_pieces(mid):
    rng = np.random.default_rng(100 + mid)
    return rng.normal(size=(N, Q)), rng.uniform(0.3, 0.9, size=N)


# -- trak: Eq. 15 unit slices + plain-mean member slices from one pass ------ #
def test_cli_trak_unit_and_member_slices(tmp_path, monkeypatch):
    p = PRESETS["pilot"]                                     # e_features = 5
    meta = _meta(p)
    data = {"meta": meta}
    monkeypatch.setattr(cli, "_load_preset_data",
                        lambda preset_, device: (p, p["recipe"], data))
    monkeypatch.setattr(trakmod, "train_ensemble", _fake_train_ensemble)
    monkeypatch.setattr(trakmod, "_single_model_pieces",
                        lambda sd, data_, mctor, *, model_id, **kw: _member_pieces(model_id))
    pool = _build_pool(tmp_path / "t", meta)

    rc = cli.main(["compute-tau", "--preset", "pilot", "--tag", "t", "--root",
                   str(tmp_path), "--device", "cpu", "--n-models", "10",
                   "--aggregation", "both", "--yes"])
    assert rc == 0
    S64 = pool.subsets.astype(np.float64)
    mpr = p["trak"]["e_features"]
    pieces = [_member_pieces(m) for m in range(10)]

    ptda = load_pool_tda(tmp_path / "t" / f"pool_tda_trak{mpr}.npz")
    assert ptda.R_tda == 2 and ptda.e_features == mpr
    assert ptda.inner == "separated-avg-eq15" and ptda.method == f"trak{mpr}"
    for k in range(2):                     # slice k == Eq. 15 over exactly its block
        blk = pieces[k * mpr:(k + 1) * mpr]
        s_bar = np.mean([s for s, _ in blk], axis=0)
        q_bar = np.mean([q for _, q in blk], axis=0)
        np.testing.assert_allclose(ptda.g[k], S64 @ (s_bar * q_bar[:, None]),
                                   rtol=1e-10)
    predictions_tda(pool, ptda)                              # pairing + mask bind pass

    pm = load_pool_tda(tmp_path / "t" / "pool_tda_trak1.npz")
    assert pm.R_tda == 10 and pm.e_features == 1 and pm.inner == "single-member"
    for m in (0, 7):
        s, q = pieces[m]
        np.testing.assert_allclose(pm.g[m], S64 @ (s * q[:, None]), rtol=1e-10)

    # the eq15 tauq itself is untouched by slice tracking: pooled separated average
    from corrected_lds.banks.artifacts import load_tauq
    tau = load_tauq(tmp_path / "t" / f"tauq_trak{mpr}_e{10 // mpr}.npz")
    s_all = np.mean([s for s, _ in pieces], axis=0)
    q_all = np.mean([q for _, q in pieces], axis=0)
    np.testing.assert_allclose(tau.tauq, (s_all * q_all[:, None]).astype(np.float32),
                               rtol=1e-5)


def test_tauq_filename_encodes_replicas_not_models(tmp_path, monkeypatch):
    """``tauq_trak<e_features>_e<E>``: E counts replicas, so the name states its own cost.
    Five models at e_features=5 are one Eq. 15 unit (_e1); the plain mean over the same five
    is five units of one (_e5); both carry n_models=5."""
    p = dict(PRESETS["pilot"])
    p["trak"] = {**p["trak"], "ensemble_sizes": [5]}
    meta = _meta(p)
    data = {"meta": meta}
    monkeypatch.setattr(cli, "_load_preset_data",
                        lambda preset_, device: (p, p["recipe"], data))
    monkeypatch.setattr(trakmod, "train_ensemble", _fake_train_ensemble)
    monkeypatch.setattr(trakmod, "_single_model_pieces",
                        lambda sd, data_, mctor, *, model_id, **kw: _member_pieces(model_id))
    _build_pool(tmp_path / "t", meta)
    assert cli.main(["compute-tau", "--preset", "pilot", "--tag", "t", "--root",
                     str(tmp_path), "--device", "cpu", "--n-models", "5",
                     "--aggregation", "both", "--yes"]) == 0
    ef = p["trak"]["e_features"]
    from corrected_lds.banks.artifacts import load_tauq
    eq15 = tmp_path / "t" / f"tauq_trak{ef}_e1.npz"        # 5 models / 5 = 1 replica
    pm = tmp_path / "t" / "tauq_trak1_e5.npz"              # 5 models / 1 = 5 replicas
    assert eq15.exists() and pm.exists()
    assert not (tmp_path / "t" / f"tauq_trak{ef}_e5.npz").exists()   # not named by models
    assert load_tauq(eq15).n_models == load_tauq(pm).n_models == 5   # same cost
    assert load_tauq(eq15).provenance["e_features"] == ef
    assert load_tauq(pm).provenance["e_features"] == 1
