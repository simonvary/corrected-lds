"""Torch-free tests of the multi-seed SOURCE driver: configs against the run that produced the
released ``tauq_source_L3_e1``, the combine against the plain mean over runs in seed order.
"""
import os

import numpy as np
import pytest

from corrected_lds.banks.masks import sample_subsets
from corrected_lds.banks.presets import PRESETS
from corrected_lds.pools import bergson_source as bs
from corrected_lds.pools.bergson import _save_pool
from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION

P = PRESETS["wikitext2_gpt2_bae"]
N, Q = int(P["n_train"]), int(P["n_targets"])                  # 4656, 481


# -- configs ---------------------------------------------------------------- #
def test_segments_and_spans():
    groups, sizes = bs.source_segments([291, 582, 873, 1164, 1455, 1746], 3)
    assert groups == [[291, 582], [873, 1164], [1455, 1746]] and sizes == [582, 582, 582]
    with pytest.raises(ValueError, match="equal segments"):
        bs.source_segments([1, 2, 3, 4, 5, 6], 4)
    spans = bs.piece_spans(481, 64)
    assert spans[0] == (0, 64) and spans[-1] == (448, 481) and len(spans) == 8
    assert bs._yaml_float(3e-5) == "3.0e-05" and bs._yaml_float(1e-8) == "1.0e-08"
    # 64 queries at L=3 peaked at ~150 GB; 16 fit a 150 GB disk
    assert 150 < bs.piece_peak_gb(64) < 175 and bs.piece_peak_gb(16) < 60


def _released_yaml():
    return bs.render_source_yaml(run_dir="/w/run_00", run_path="/dev/shm/l0", lo=0, hi=64,
                                 overwrite=True, recipe=P["recipe"], n_train=N,
                                 units_per_epoch=2, repo=WIKITEXT2_REPO,
                                 revision=WIKITEXT2_REVISION)


def test_source_yaml_lines_without_a_yaml_parser():
    """The same settings as the typed test below, line by line (CI has no PyYAML)."""
    lines = [ln.strip() for ln in _released_yaml().splitlines()]
    for want in ("run_path: /dev/shm/l0", "model: /w/run_00/exported/checkpoint-1746",
                 "overwrite: true", "precision: fp32", "token_batch_size: 1024",
                 "max_batch_size: 64", 'filter_modules: "lm_head"', "nproc_per_node: 1",
                 "method: kfac", "hessian_dtype: fp32", "ev_correction: true",
                 "segments: 3", "lr_list: [3.0e-05, 3.0e-05, 3.0e-05]",
                 "step_size_list: [582, 582, 582]", "split: validation[0:64]",
                 "query_aggregation: none", "query_batch_size: 64",
                 "use_adam_preconditioner: true", "damping_factor: 1.0e-08"):
        assert want in lines, want
    assert lines.count(f"data_kwargs: revision={WIKITEXT2_REVISION}") == 2
    assert [ln for ln in lines if ln.startswith("- /w/run_00/checkpoints/")] == [
        f"- /w/run_00/checkpoints/step_{s}.ckpt" for s in (291, 582, 873, 1164, 1455, 1746)]


def test_source_train_yaml_keeps_the_adam_state():
    yaml = pytest.importorskip("yaml")
    y = bs.render_source_train_yaml(3, root="/r/runs", recipe=P["recipe"], seed_base=30000,
                                    n_train=N, units_per_epoch=2, repo=WIKITEXT2_REPO,
                                    revision=WIKITEXT2_REVISION)
    t = yaml.safe_load(y)["steps"][0]["train"]
    assert t["seed"] == 30003 and t["run_path"] == "/r/runs/run_03"
    assert t["save_optimizer_state"] == "all" and t["save_interval"] == 291
    # the trajectory YAML writes ``3e-05``, a string to PyYAML that bergson coerces: compare values
    assert t["train_mode"] is True and float(t["lr_schedule"]["lr"]) == 3e-5


def test_bootstrap_pins_bergson_and_checks_the_fix():
    b = bs.render_source_bootstrap()
    assert "bergson.git@0ad2e5b0ce09207d2af0e80383fe2d952dd8e747" in b
    assert f"cherry-pick {bs.SOURCE_FIX_COMMIT}" in b
    assert bs._render_fix_check("/root/bv", "/root/bergson_fix_site") in b


# -- combine ---------------------------------------------------------------- #


def _pool(tmp_path, n=N, q=Q, M=8, R=2):
    masks = sample_subsets(M, n, 0.5, 777)
    y = np.random.default_rng(1).normal(3.0, 0.1, (R, M, q))
    return _save_pool(str(tmp_path / "pool_eval.npz"), y, masks, [1004, 1005], alpha=0.5,
                      recipe={"alpha": 0.5}, Q=q, dataset="wikitext2",
                      code_version="test", readout="eval")


_PROV = {"bergson_commit": "0ad2e5b0 + #511 (08e50eff) cherry-picked as ea4f6941",
         "checkpoints": "steps 291..1746 (6, half-epoch)", "damping": 1e-08,
         "harness": "bergson", "method": "source_L3", "scorer": "bergson approxunrolling",
         "segments": 3, "sign": "negated: bergson SOURCE scores are loss-signed",
         "use_adam_preconditioner": True, "queries": "8 pieces"}


def test_combine_is_the_plain_mean_with_runs_in_seed_order(tmp_path):
    from corrected_lds.banks.artifacts import load_pool, load_pool_tda, load_tauq, write_tauq
    n, q = 60, 6
    pool = load_pool(_pool(tmp_path, n=n, q=q))
    rng = np.random.default_rng(0)
    taus, paths = {}, []
    for seed in (30002, 30000, 30001):                  # out of order on purpose
        taus[seed] = rng.normal(size=(n, q))
        commit = (_PROV["bergson_commit"] if seed == 30000
                  else "0ad2e5b0 + #511 (08e50eff)")
        paths.append(write_tauq(str(tmp_path / f"t{seed}.npz"), taus[seed], pool,
                                provenance=dict(_PROV, train_seed=seed,
                                                bergson_commit=commit)))
    tq, pt = bs.combine_source_runs(paths, pool, str(tmp_path / "out"))
    t = load_tauq(tq)
    assert os.path.basename(tq) == "tauq_source_L3_e3.npz" and t.n_models == 3
    np.testing.assert_allclose(t.tauq, np.mean([taus[s] for s in (30000, 30001, 30002)],
                                               axis=0), atol=1e-6)      # float32 on disk
    assert t.provenance["train_seeds"] == [30000, 30001, 30002]
    ptda = load_pool_tda(pt)
    np.testing.assert_allclose(ptda.g[1], pool.subsets.astype(float) @ taus[30001],
                               atol=1e-4)

    bad = write_tauq(str(tmp_path / "bad.npz"), taus[30000], pool,
                     provenance=dict(_PROV, train_seed=30005, damping=1e-3))
    with pytest.raises(ValueError, match="different settings"):
        bs.combine_source_runs(paths + [bad], pool, str(tmp_path / "out2"))
    with pytest.raises(ValueError, match="repeated train seeds"):
        bs.combine_source_runs(paths + [paths[0]], pool, str(tmp_path / "out3"))
    pinned = write_tauq(str(tmp_path / "pin.npz"), taus[30000], pool,
                        provenance=dict(_PROV, train_seed=30007,
                                        bergson_commit="0ad2e5b0"))
    with pytest.raises(ValueError, match="pin \\+ #511"):
        bs.combine_source_runs(paths + [pinned], pool, str(tmp_path / "out4"))
