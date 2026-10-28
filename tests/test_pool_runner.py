"""Pool runner: seed/shard pins and the pool lifecycle on a dummy CPU task (no torch, data or
GPU).  The refusals matter most: a resumable GPU-week artifact must not be overwritten,
shrunk, mixed across identities, or merged with gaps, duplicates or stale shards.
"""

import json
import os

import numpy as np
import pytest

from corrected_lds.banks.masks import sample_subsets
from corrected_lds.banks.presets import Resnet9Recipe
from corrected_lds.pools.runner import (
    PoolTask,
    build_pool,
    cell_owner,
    cell_seed,
    merge_shards,
    pool_meta,
    shard_cells,
)


# -- pinned constants ------------------------------------------------------- #
def test_cell_seed_pins():
    assert cell_seed(0, 0) == 50_000
    assert cell_seed(2, 37) == 70_037
    with pytest.raises(ValueError, match="10000"):
        cell_seed(0, 10_000)
    with pytest.raises(ValueError):
        cell_seed(0, -1)


def test_cell_owner_pins():
    assert cell_owner(0, 0, 4) == 2
    assert cell_owner(1, 7, 4) == 1
    assert cell_owner(19, 299, 4) == 2
    assert [cell_owner(0, j, 3) for j in range(6)] == [1, 0, 2, 1, 2, 2]
    # dimension-independent: ownership of (r, j) never depends on M or R
    assert shard_cells(6, 1, 0, 3) == [(0, j) for j in range(6)
                                       if cell_owner(0, j, 3) == 0]


# -- dummy CPU task: deterministic closed-form "training" ------------------- #
def _task(recipe=None, log=None, boom_after=None, readout_override=None):
    """readout(cell) is a deterministic function of (seed, subset), so resumed/extended
    pools must be bitwise-equal to fresh ones."""
    recipe = recipe or Resnet9Recipe(epochs=1)
    calls = {"n": 0}

    def train_cell(data, sub_idx, seed):
        calls["n"] += 1
        if boom_after is not None and calls["n"] > boom_after:
            raise RuntimeError("boom")
        if log is not None:
            log.append(int(seed))
        return (int(seed), np.asarray(sub_idx))

    def readout(model, data):
        if readout_override is not None:
            return readout_override(model, data)
        seed, sub_idx = model
        Q = data["meta"]["n_targets"]
        return np.arange(Q, dtype=float) + seed * 1e-3 + float(sub_idx.sum()) * 1e-7

    return PoolTask(recipe=recipe, train_cell=train_cell, readout=readout)


def _data(n=40, Q=3, dataset="cifar10", query_idx=None, train_hash="th"):
    return {"meta": {"n_train": n, "n_targets": Q, "dataset": dataset,
                     "train_idx": list(range(n)),
                     "query_idx": list(range(Q)) if query_idx is None else query_idx,
                     "train_hash": train_hash, "query_hash": "qh"}}


def _load(pool_dir, filename="pool.npz"):
    return np.load(os.path.join(pool_dir, filename))


def test_fresh_build_completes_with_schema(tmp_path):
    d = str(tmp_path / "p")
    path = build_pool(d, _task(), _data(), M=5, R=2, verbose=False)
    z = np.load(path)
    assert z["margins"].shape == (2, 5, 3) and np.isfinite(z["margins"]).all()
    assert int(z["completed_cells"]) == 10 and int(z["done_rows"]) == 2
    assert int(z["schema"]) == 3 and str(z["seed_mode"]) == "nested"
    assert str(z["subsets_source"]) == "sample_subsets:777"
    assert str(z["dataset"]) == "cifar10" and float(z["alpha"]) == 0.5
    np.testing.assert_array_equal(z["subsets"], sample_subsets(5, 40, 0.5, 777))
    np.testing.assert_array_equal(z["train_idx"], np.arange(40))
    np.testing.assert_array_equal(z["query_idx"], np.arange(3))
    assert str(z["train_hash"]) == "th" and "corrected-lds" in str(z["code_version"])
    assert json.loads(str(z["recipe"]))["epochs"] == 1


def test_interrupt_resume_is_bitwise_identical(tmp_path):
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    with pytest.raises(RuntimeError, match="boom"):
        build_pool(d1, _task(boom_after=4), _data(), M=4, R=2, save_every=1, verbose=False)
    partial = _load(d1)
    assert 0 < int(partial["completed_cells"]) < 8       # crash persisted partial progress
    log = []
    build_pool(d1, _task(log=log), _data(), M=4, R=2, verbose=False)          # resume
    assert len(log) == 8 - int(partial["completed_cells"])                    # only missing cells
    fresh = build_pool(d2, _task(), _data(), M=4, R=2, verbose=False)         # one-shot reference
    np.testing.assert_array_equal(_load(d1)["margins"], np.load(fresh)["margins"])


@pytest.mark.parametrize("grow", ["M", "R"])
def test_extension_trains_only_new_cells_and_preserves_old(tmp_path, grow):
    d = str(tmp_path / "p")
    build_pool(d, _task(), _data(), M=3, R=2, verbose=False)
    before = _load(d)["margins"].copy()
    M2, R2 = (5, 2) if grow == "M" else (3, 4)
    log = []
    build_pool(d, _task(log=log), _data(), M=M2, R=R2, verbose=False)
    after = _load(d)["margins"]
    assert len(log) == R2 * M2 - 6                        # only the new cells trained
    np.testing.assert_array_equal(after[:2, :3], before)  # old cells bitwise-untouched
    assert np.isfinite(after).all()


def test_refusals(tmp_path):
    d = str(tmp_path / "p")
    build_pool(d, _task(), _data(), M=4, R=2, verbose=False)
    with pytest.raises(RuntimeError, match="shrink"):                 # R may only grow
        build_pool(d, _task(), _data(), M=4, R=1, verbose=False)
    with pytest.raises(RuntimeError, match="does NOT match"):         # M-shrink fails the
        build_pool(d, _task(), _data(), M=3, R=2, verbose=False)      # mask-prefix guard
    with pytest.raises(RuntimeError, match="does NOT match"):
        build_pool(d, _task(recipe=Resnet9Recipe(epochs=2)), _data(), M=4, R=2, verbose=False)
    with pytest.raises(RuntimeError, match="does NOT match"):
        build_pool(d, _task(), _data(dataset="mnist"), M=4, R=2, verbose=False)
    with pytest.raises(RuntimeError, match="does NOT match"):
        build_pool(d, _task(), _data(Q=5), M=4, R=2, verbose=False)


def test_resume_refuses_a_different_data_identity(tmp_path):
    """A pool trained under one train/query draw must never be resumed, extended, or
    relabeled under another — the saver would overwrite the stored identity."""
    d = str(tmp_path / "p")
    build_pool(d, _task(), _data(), M=4, R=2, verbose=False)
    with pytest.raises(RuntimeError, match="DATA IDENTITY"):        # different query draw
        build_pool(d, _task(), _data(query_idx=[2, 0, 1]), M=4, R=2, verbose=False)
    with pytest.raises(RuntimeError, match="DATA IDENTITY"):        # different data content
        build_pool(d, _task(), _data(train_hash="OTHER"), M=4, R=2, verbose=False)
    bare = _data()
    del bare["meta"]["query_idx"]                                   # identity missing on one
    with pytest.raises(RuntimeError, match="DATA IDENTITY"):        # side is also a mismatch
        build_pool(d, _task(), bare, M=4, R=2, verbose=False)


def test_nonfinite_or_misshapen_readout_refused(tmp_path):
    def bad_inf(model, data):
        return np.array([1.0, np.inf, 0.0])

    def bad_shape(model, data):
        return np.zeros(5)

    with pytest.raises(ValueError, match="non-finite"):
        build_pool(str(tmp_path / "a"), _task(readout_override=bad_inf), _data(), M=2, R=1,
                   verbose=False)
    with pytest.raises(ValueError, match="shape"):
        build_pool(str(tmp_path / "b"), _task(readout_override=bad_shape), _data(), M=2, R=1,
                   verbose=False)


# -- shards ----------------------------------------------------------------- #
def _resave_shard(pool_dir, i, mutate):
    sp = os.path.join(pool_dir, f"pool_shard_{i}.npz")
    z = np.load(sp)
    d = {k: z[k] for k in z.files}
    mutate(d)
    np.savez_compressed(sp + ".tmp", **d)
    os.replace(sp + ".tmp.npz", sp)


def test_shard_build_and_merge_matches_serial(tmp_path):
    ds, dm = str(tmp_path / "serial"), str(tmp_path / "sharded")
    serial = build_pool(ds, _task(), _data(), M=4, R=3, verbose=False)
    for i in range(2):
        build_pool(dm, _task(), _data(), M=4, R=3, shard=(i, 2), verbose=False)
    merged = merge_shards(dm, _task().recipe, _data(), M=4, R=3, N=2)
    np.testing.assert_array_equal(np.load(merged)["margins"], np.load(serial)["margins"])
    assert pool_meta(np.load(merged)["margins"])[1] == 3


def test_merge_refuses_missing_gap_dup_and_stale(tmp_path):
    d = str(tmp_path / "p")
    task, data = _task(), _data()
    build_pool(d, task, data, M=4, R=3, shard=(0, 2), verbose=False)
    with pytest.raises(RuntimeError, match="missing"):                 # shard 1 never built
        merge_shards(d, task.recipe, data, M=4, R=3, N=2)
    build_pool(d, task, data, M=4, R=3, shard=(1, 2), verbose=False)

    # stale shard: a different recipe identity must not be publishable
    with pytest.raises(RuntimeError, match="stale shard"):
        merge_shards(d, Resnet9Recipe(epochs=2), data, M=4, R=3, N=2)

    # data-identity drift: shards carry their query draw and must not merge under another
    with pytest.raises(RuntimeError, match="DATA IDENTITY"):
        merge_shards(d, task.recipe, _data(query_idx=[2, 0, 1]), M=4, R=3, N=2)

    # gap: erase one owned cell inside shard 0 -> "written by NO shard"
    r0, j0 = shard_cells(4, 3, 0, 2)[0]

    def _gap(z):
        z["margins"][r0, j0] = np.nan
        z["completed_cells"] = int(z["completed_cells"]) - 1

    _resave_shard(d, 0, _gap)
    with pytest.raises(RuntimeError, match="NO shard"):
        merge_shards(d, task.recipe, data, M=4, R=3, N=2)

    # duplicate/ownership violation: shard 0 also fills a cell shard 1 owns
    r1, j1 = shard_cells(4, 3, 1, 2)[0]

    def _dup(z):
        z["margins"][r0, j0] = 0.0                            # restore the gap
        z["margins"][r1, j1] = 0.0                            # steal shard 1's cell
        z["completed_cells"] = int(z["completed_cells"]) + 2

    _resave_shard(d, 0, _dup)
    with pytest.raises(RuntimeError, match="does not own"):
        merge_shards(d, task.recipe, data, M=4, R=3, N=2)


def test_merge_refuses_a_seed_that_did_not_take(tmp_path):
    """Two seeds with identical margins on a mask mean a cell trained without its own seed."""
    d = str(tmp_path / "p")
    for i in range(2):
        build_pool(d, _task(), _data(), M=4, R=2, shard=(i, 2), verbose=False)
    row = _load(d, f"pool_shard_{cell_owner(0, 1, 2)}.npz")["margins"][0, 1].copy()

    def _same(z):
        z["margins"][1, 1] = row

    _resave_shard(d, cell_owner(1, 1, 2), _same)
    with pytest.raises(RuntimeError, match=r"seeds \[0, 1\] gave identical margins on mask 1"):
        merge_shards(d, _task().recipe, _data(), M=4, R=2, N=2)


# -- code-version stamps ---------------------------------------------------- #


def test_merge_records_code_versions_and_refuses_mixed_code(tmp_path, monkeypatch):
    """Shards of different code merge only when allowed; the pool then records every version
    behind its cells (the extended pool's too) and is stamped by the merging process."""
    from corrected_lds.pools import runner
    d = str(tmp_path / "p")
    monkeypatch.setattr(runner, "_code_version", lambda: "corrected-lds-e95b405")
    build_pool(d, _task(), _data(), M=3, R=2, verbose=False)              # the base pool
    for i, stamp in enumerate(("corrected-lds-7a1a04f", "corrected-lds-7135506")):
        monkeypatch.setattr(runner, "_code_version", lambda s=stamp: s)
        build_pool(d, _task(), _data(), M=5, R=4, shard=(i, 2), verbose=False)
    with pytest.raises(RuntimeError, match="different code"):
        merge_shards(d, _task().recipe, _data(), M=5, R=4, N=2)
    monkeypatch.setattr(runner, "_code_version", lambda: "corrected-lds-798d4cf")
    z = np.load(merge_shards(d, _task().recipe, _data(), M=5, R=4, N=2,
                             allow_mixed_code=True))
    assert str(z["code_version"]) == "corrected-lds-798d4cf"
    assert json.loads(str(z["code_versions"])) == [
        "corrected-lds-7135506", "corrected-lds-7a1a04f", "corrected-lds-e95b405"]
