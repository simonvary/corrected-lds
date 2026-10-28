"""Loaders (complete banks only, mask re-derivation, schema-3 keys, hash pins), pairing-enforced
predictions and manifest-verified fetch, on real pools the runner writes for a dummy CPU task.
"""

import dataclasses
import hashlib
import json
import os

import numpy as np
import pytest

from corrected_lds.banks.artifacts import load_pool, load_tauq, predictions, write_tauq
from corrected_lds.banks.presets import Resnet9Recipe, bank_hashes
from corrected_lds.cli import main as cli_main
from corrected_lds.pools.runner import PoolTask, build_pool

N, Q = 40, 3
TRAIN_HASH, QUERY_HASH = bank_hashes("cifar10")     # loaders pin-check known datasets


def _task(recipe=None, boom_after=None):
    recipe = recipe or Resnet9Recipe(epochs=1)
    calls = {"n": 0}

    def train_cell(data, sub_idx, seed):
        calls["n"] += 1
        if boom_after is not None and calls["n"] > boom_after:
            raise RuntimeError("boom")
        return (int(seed), np.asarray(sub_idx))

    def readout(model, data):
        seed, sub_idx = model
        return np.arange(Q, dtype=float) + seed * 1e-3 + float(sub_idx.sum()) * 1e-7

    return PoolTask(recipe=recipe, train_cell=train_cell, readout=readout)


def _data():
    return {"meta": {"n_train": N, "n_targets": Q, "dataset": "cifar10",
                     "train_idx": list(range(N)), "query_idx": [5, 1, 9],
                     "train_hash": TRAIN_HASH, "query_hash": QUERY_HASH}}


def _bank(tmp_path, M=5, R=3, **kw):
    d = str(tmp_path / "pool")
    path = build_pool(d, _task(**kw), _data(), M=M, R=R, verbose=False)
    return path


def _resave(path, mutate):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    mutate(d)
    np.savez(path[:-4], **d)                              # np.savez appends .npz


def _tauq(tmp_path, **over):
    vals = {"tauq": np.random.default_rng(3).normal(size=(N, Q)).astype(np.float32),
            "dataset": "cifar10", "n_train": N, "n_targets": Q,
            "train_idx": np.arange(N), "query_idx": np.array([5, 1, 9]),
            "train_hash": TRAIN_HASH, "query_hash": QUERY_HASH,
            "n_models": 10, "proj_dim": 512,
            "provenance": json.dumps({"code": "test"})}
    vals.update(over)
    os.makedirs(str(tmp_path), exist_ok=True)
    p = str(tmp_path / "tauq.npz")
    np.savez(p[:-4], **vals)
    return p


# -- load_pool -------------------------------------------------------------- #
def test_load_pool_roundtrip(tmp_path):
    path = _bank(tmp_path)
    b = load_pool(path)
    assert (b.R, b.M, b.Q, b.n) == (3, 5, Q, N)
    assert b.margins.dtype == np.float64 and np.isfinite(b.margins).all()
    assert b.recipe["epochs"] == 1 and b.dataset == "cifar10" and b.alpha == 0.5
    np.testing.assert_array_equal(b.train_idx, np.arange(N))
    np.testing.assert_array_equal(b.query_idx, [5, 1, 9])
    assert (b.train_hash, b.query_hash) == (TRAIN_HASH, QUERY_HASH)
    assert "corrected-lds" in b.code_version


def test_load_pool_refuses_incomplete(tmp_path):
    d = str(tmp_path / "pool")
    with pytest.raises(RuntimeError, match="boom"):
        build_pool(d, _task(boom_after=4), _data(), M=5, R=3, save_every=1, verbose=False)
    with pytest.raises(ValueError, match="COMPLETE"):
        load_pool(os.path.join(d, "pool.npz"))


def test_load_pool_refuses_tampering_and_missing_keys(tmp_path):
    path = _bank(tmp_path)

    def flip_mask(d):
        d["subsets"] = d["subsets"].copy()
        d["subsets"][0, 0] = 1 - d["subsets"][0, 0]

    _resave(path, flip_mask)
    with pytest.raises(ValueError, match="mask identity"):
        load_pool(path)

    path2 = _bank(tmp_path / "b")

    def inf_cell(d):
        d["margins"] = d["margins"].copy()
        d["margins"][0, 0, 0] = np.inf

    _resave(path2, inf_cell)
    with pytest.raises(ValueError, match="COMPLETE"):
        load_pool(path2)

    path3 = _bank(tmp_path / "c")

    def drop_hash(d):
        del d["train_hash"]

    _resave(path3, drop_hash)
    with pytest.raises(ValueError, match="missing required key"):
        load_pool(path3)

    path4 = _bank(tmp_path / "d")                        # a tampered hash string must not

    def fake_hash(d):                                    # pass just because the key exists
        d["train_hash"] = "0" * 16

    _resave(path4, fake_hash)
    with pytest.raises(ValueError, match="pinned hashes"):
        load_pool(path4)

    path5 = _bank(tmp_path / "e")

    def dup_query(d):                                    # a duplicate index double-counts
        d["query_idx"] = np.array([5, 1, 5])

    _resave(path5, dup_query)
    with pytest.raises(ValueError, match="duplicate-free"):
        load_pool(path5)

    path6 = _bank(tmp_path / "f")

    def drift_alpha(d):                                  # scalar alpha vs recipe alpha
        d["alpha"] = 0.4

    _resave(path6, drift_alpha)
    with pytest.raises(ValueError, match="internally inconsistent"):
        load_pool(path6)


# -- load_tauq + predictions pairing ---------------------------------------- #
def test_predictions_pairing_and_matmul(tmp_path):
    pool = load_pool(_bank(tmp_path))
    tau = load_tauq(_tauq(tmp_path))
    g = predictions(pool, tau)
    assert g.shape == (pool.M, Q) and g.dtype == np.float64
    np.testing.assert_array_equal(
        g, pool.subsets.astype(np.float64) @ tau.tauq.astype(np.float64))

    # load_tauq's pin check refuses tampered hashes, so build mismatches on the loaded object
    for over, msg in [({"train_hash": "OTHER"}, "train_hash"),
                      ({"query_hash": "OTHER"}, "query_hash"),
                      ({"query_idx": np.array([5, 9, 1])}, "query_idx"),
                      ({"train_idx": np.arange(1, N + 1)}, "train_idx"),
                      ({"dataset": "other"}, "dataset"),
                      ({"tauq": np.zeros((N + 1, Q)), "n_train": N + 1,
                        "train_idx": np.arange(N + 1)}, "n_train")]:
        bad = dataclasses.replace(tau, **over)
        with pytest.raises(ValueError, match=msg):
            predictions(pool, bad)


def test_write_tauq_bring_your_own_scores(tmp_path):
    """The BYO contract: identity copied from a loaded Pool -> pairable by construction,
    torch-free end to end; wrong-shape scores refused before anything is written."""
    pool = load_pool(_bank(tmp_path))
    scores = np.random.default_rng(9).normal(size=(pool.n, pool.Q))
    out = write_tauq(str(tmp_path / "tauq_mymethod.npz"), scores, pool,
                     provenance={"method": "mymethod"})
    tau = load_tauq(out)
    assert predictions(pool, tau).shape == (pool.M, pool.Q)   # pairing accepted
    assert tau.provenance["method"] == "mymethod"
    assert (tau.n_models, tau.proj_dim) == (1, 0)             # N/A sentinels
    with pytest.raises(ValueError, match="shape"):
        write_tauq(str(tmp_path / "bad.npz"), scores[:-1], pool, provenance={})
    assert not os.path.exists(str(tmp_path / "bad.npz"))


def test_load_tauq_refuses_bad_shapes_and_nonfinite(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        load_tauq(_tauq(tmp_path / "a", tauq=np.zeros((N, Q + 1), np.float32)))
    bad = np.zeros((N, Q), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        load_tauq(_tauq(tmp_path / "b", tauq=bad))
    p = _tauq(tmp_path / "c")

    def drop(d):
        del d["provenance"]

    _resave(p, drop)
    with pytest.raises(ValueError, match="missing required key"):
        load_tauq(p)
    with pytest.raises(ValueError, match="pinned hashes"):        # pin check at load, too
        load_tauq(_tauq(tmp_path / "d", train_hash="OTHER"))
    with pytest.raises(ValueError, match="duplicate-free"):
        load_tauq(_tauq(tmp_path / "e", train_idx=np.zeros(N, np.int64)))


# -- fetch-banks ------------------------------------------------------------ #
def _write_manifest(tmp_path, url, banks):
    m = {"format": 2, "url": url, "banks": banks}
    p = str(tmp_path / "manifest.json")
    with open(p, "w") as fh:
        json.dump(m, fh)
    return p


def _host(tmp_path, data):
    """``data`` hosted as ``<base>/demo/pool.npz``; returns the base url."""
    os.makedirs(tmp_path / "hosted" / "demo")
    (tmp_path / "hosted" / "demo" / "pool.npz").write_bytes(data)
    return (tmp_path / "hosted").as_uri()


def test_fetch_banks_downloads_verifies_and_skips(tmp_path):
    url = _host(tmp_path, b"pool bytes")
    sha = hashlib.sha256(b"pool bytes").hexdigest()
    root = str(tmp_path / "banks")
    manifest = _write_manifest(tmp_path, url, {"demo": {"pool.npz": sha}})
    assert cli_main(["fetch-banks", "--manifest", manifest, "--root", root]) == 0
    dest = os.path.join(root, "demo", "pool.npz")
    with open(dest, "rb") as fh:
        assert fh.read() == b"pool bytes"
    # second run skips (URL now dead, but the verified file short-circuits the download)
    manifest2 = _write_manifest(tmp_path, (tmp_path / "GONE").as_uri(),
                                {"demo": {"pool.npz": sha}})
    assert cli_main(["fetch-banks", "--manifest", manifest2, "--root", root]) == 0


def test_fetch_banks_refuses_hash_mismatch(tmp_path):
    url = _host(tmp_path, b"tampered")
    root = str(tmp_path / "banks")
    manifest = _write_manifest(tmp_path, url, {"demo": {"pool.npz": "0" * 64}})
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        cli_main(["fetch-banks", "--manifest", manifest, "--root", root])
    assert not os.path.exists(os.path.join(root, "demo", "pool.npz"))


def test_fetch_banks_empty_manifest_reports(tmp_path):
    manifest = _write_manifest(tmp_path, "https://unused", {})
    assert cli_main(["fetch-banks", "--manifest", manifest,
                     "--root", str(tmp_path / "banks")]) == 1
