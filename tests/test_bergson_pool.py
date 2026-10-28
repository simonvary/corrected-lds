"""Torch-free bergson pool bridge tests on fixture CSVs shaped as bergson's ``validate`` writes."""
import os

import numpy as np
import pytest

from corrected_lds.banks.masks import sample_subsets
from corrected_lds.estimator import (
    seed_noise_variance_centred,
    seed_noise_variance_stacked,
)
from corrected_lds.pools.bergson import (
    collect,
    read_subsets_json,
    write_subsets_json,
)


# -- masks <-> bergson's subsets.json --------------------------------------- #
@pytest.mark.parametrize("alpha,n,M", [(0.5, 4656, 8), (0.9901, 4608, 5), (0.99, 300, 3)])
def test_subsets_json_round_trip_is_exact(tmp_path, alpha, n, M):
    """Our masks (1 = kept) survive bergson's removed-id format byte-for-byte."""
    masks = sample_subsets(M, n, alpha, 777)
    p = tmp_path / "subsets.json"
    write_subsets_json(masks, p)
    assert np.array_equal(masks, read_subsets_json(p, n))


def test_subsets_json_lists_removed_not_kept(tmp_path):
    """Polarity is the easiest thing to get backwards, so pin it explicitly."""
    masks = sample_subsets(4, 200, 0.5, 777)
    p = tmp_path / "subsets.json"
    removed = write_subsets_json(masks, p)
    for j in range(4):
        assert sorted(removed[j]) == sorted(np.nonzero(masks[j] == 0)[0].tolist())
        assert len(removed[j]) == 200 - int(masks[j].sum())


# -- collect(): bergson's CSVs -> raw per-query loss ------------------------ #
def _write_chunk(d, lo, hi, Q, base, loss):
    """Write one chunk in bergson's exact CSV shape (diff = baseline - loss)."""
    os.makedirs(d, exist_ok=True)
    sfx = f"_{lo}_{hi}"
    # plain floats: numpy 2.x reprs a scalar as "np.float64(3.0)", which genfromtxt reads as NaN
    with open(os.path.join(d, f"validation{sfx}.csv"), "w") as f:
        f.write("subset,query,diff,score_sum\n")
        for j in range(lo, hi):
            f.writelines(f"{j},{q},{float(base[q] - loss[j, q]):.17g},0.0\n" for q in range(Q))
    with open(os.path.join(d, f"summary{sfx}.csv"), "w") as f:
        f.write("query,spearman_corr,spearman_p,pearson_corr,pearson_p,N,baseline_loss\n")
        f.writelines(f"{q},0.0,1.0,0.0,1.0,{hi - lo},{float(base[q]):.17g}\n" for q in range(Q))


def test_collect_reconstructs_raw_loss_across_chunks(tmp_path):
    """Chunks stitch, and each chunk's own baseline cancels in baseline - diff: chunking is safe."""
    M, Q = 10, 4
    rng = np.random.default_rng(0)
    loss = rng.normal(3.0, 0.01, (M, Q))
    _write_chunk(tmp_path, 0, 6, Q, base=rng.normal(3.0, 0.01, Q), loss=loss)
    _write_chunk(tmp_path, 6, 10, Q, base=rng.normal(3.0, 0.01, Q), loss=loss)
    got, _, seen = collect(str(tmp_path), M, Q)
    assert seen.all()
    assert np.allclose(got, loss, rtol=0, atol=1e-12)


def test_collect_flags_missing_masks(tmp_path):
    """An incomplete run must be visible, never silently zero-filled."""
    M, Q = 10, 3
    loss = np.full((M, Q), 3.0)
    _write_chunk(tmp_path, 0, 6, Q, base=np.full(Q, 3.0), loss=loss)
    _, _, seen = collect(str(tmp_path), M, Q)
    assert seen[:6].all() and not seen[6:].any()


# -- the shared-seed estimator ---------------------------------------------- #
def test_centred_variance_is_invariant_to_a_per_replica_offset():
    """c(r) shifts a whole replica and must not count toward the attenuating floor."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, (4, 60, 3))
    off = rng.normal(0.0, 50.0, (4, 1, 1))                   # huge per-replica offsets
    assert np.allclose(seed_noise_variance_centred(y), seed_noise_variance_centred(y + off))
    # the uncentred estimator is not invariant: that is the point
    assert seed_noise_variance_stacked(y + off).mean() > 100 * seed_noise_variance_stacked(y).mean()


def test_centred_variance_is_unbiased_for_pure_interaction():
    """With no offset, centred and uncentred must agree up to the m/(m-1) df cost."""
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, (6, 300, 4))
    a = seed_noise_variance_centred(y).mean()
    b = seed_noise_variance_stacked(y).mean()
    assert np.isclose(a, b, rtol=0.05)


# -- schema ----------------------------------------------------------------- #


def test_evaluate_routes_the_seed_noise_estimator_by_seed_mode():
    """The stamped seed_mode picks the floor, so a shared-seed pool is never over-subtracted."""
    from corrected_lds.evaluate import _seed_noise_fn

    class _P:
        def __init__(self, mode):
            self.seed_mode = mode

    assert _seed_noise_fn(_P("nested")) is seed_noise_variance_stacked
    assert _seed_noise_fn(_P("shared-per-replica")) is seed_noise_variance_centred
    with pytest.raises(ValueError, match="seed_mode"):
        _seed_noise_fn(_P("per-epoch"))


def _shared_bank(tmp_path, R=4, M=40, Q=6, n=60):
    """A shared-per-replica pool built from bergson-shaped CSVs, plus a tauq with real signal."""
    from corrected_lds.banks.artifacts import load_pool, load_tauq, write_tauq
    from corrected_lds.banks.presets import bank_hashes
    from corrected_lds.pools.bergson import build_pool

    rng = np.random.default_rng(0)
    masks = sample_subsets(M, n, 0.5, 777)
    tau = rng.normal(size=(n, Q))
    margin = masks @ tau * 0.05 + rng.normal(0, 0.4, (R, M, Q))    # mu(S) + eps(r, j)
    base = np.full(Q, 3.0)
    dirs = []
    for r in range(R):
        d = str(tmp_path / f"r{r}")
        _write_chunk(d, 0, M, Q, base, -margin[r])                 # loss = -margin
        dirs.append(d)
    pool_path = build_pool(str(tmp_path / "pool.npz"), dirs, list(range(R)), masks,
                           alpha=0.5, recipe={"alpha": 0.5}, Q=Q)
    th, qh = bank_hashes("wikitext2")
    meta = {"dataset": "wikitext2", "n_train": n, "n_targets": Q,
            "train_idx": list(range(n)), "query_idx": list(range(Q)),
            "train_hash": th, "query_hash": qh}
    tau_path = write_tauq(str(tmp_path / "tauq_x.npz"), tau, meta,
                          provenance={"method": "ekfac"})
    return load_pool(pool_path), load_tauq(tau_path)


def test_shared_pool_corrected_results_ignore_the_replica_offset(tmp_path):
    """bergson's per-(replica, query) offset c(r) moves no corrected number of a shared-seed pool;
    labelled nested, it inflates the floor, which is why seed_mode routes the estimator."""
    import dataclasses

    from corrected_lds.evaluate import evaluate_pool

    pool, tau = _shared_bank(tmp_path)
    assert pool.seed_mode == "shared-per-replica"
    c = np.random.default_rng(1).normal(0, 50.0, (pool.R, 1, pool.Q))
    shifted = dataclasses.replace(pool, margins=pool.margins + c)
    a = evaluate_pool(pool, tau)
    b = evaluate_pool(shifted, tau)
    assert a["pool"]["seed_mode"] == "shared-per-replica"
    for k in ("naive_rho1", "spearman", "corrected_rho_inf", "corrected_se_jackknife",
              "bridged_rho_inf", "sigma2_mean", "lambda_median",
              "debiased_noise_corrected_spearman"):
        assert a["results"][k] == pytest.approx(b["results"][k], rel=1e-8, abs=1e-10), k
    assert b["diagnostics"]["seed_offset_share_mean"] > 0.99          # c dominates
    assert a["results"]["sigma2_mean"] == pytest.approx(
        float(seed_noise_variance_centred(pool.margins).mean()))
    nested = dataclasses.replace(shifted, seed_mode="nested")
    wrong = evaluate_pool(nested, tau)
    assert wrong["results"]["sigma2_mean"] > 100 * a["results"]["sigma2_mean"]
    assert "seed_mode" not in wrong["pool"]          # nested results stay byte-identical


def test_auto_pool_tda_key_skips_a_tauq_scored_by_bergson():
    """bergson scored tauq_ekfac_bergson.npz, not our pool_tda_ekfac.npz: never auto-pair them."""
    from types import SimpleNamespace

    from corrected_lds import cli

    theirs = SimpleNamespace(provenance={"method": "ekfac", "harness": "bergson"})
    ours = SimpleNamespace(provenance={"method": "ekfac", "protocol": "full-set-final-checkpoint"})
    assert cli._pool_tda_auto_key(theirs) is None
    assert cli._pool_tda_auto_key(ours) == "ekfac"


# -- attributor trajectories trained by bergson ----------------------------- #
def test_bae_trajectory_steps_are_the_six_half_epoch_checkpoints():
    from corrected_lds.banks.presets import PRESETS
    from corrected_lds.pools.bergson import trajectory_steps

    p = PRESETS["wikitext2_gpt2_bae"]
    k = p["tracin"]["units_per_epoch"]
    assert trajectory_steps(p["recipe"], p["n_train"], k) == [291, 582, 873, 1164, 1455, 1746]
    with pytest.raises(ValueError, match="equal units"):
        trajectory_steps(p["recipe"], p["n_train"], 4)              # 582 % 4 != 0
    with pytest.raises(ValueError, match="equal units"):
        trajectory_steps(p["recipe"], 4655, 2)                      # ragged final batch


def test_trajectory_yaml_is_the_bank_recipe(tmp_path):
    """Training fields come off the recipe, so scored runs and retrains cannot drift apart."""
    from corrected_lds.banks.presets import PRESETS
    from corrected_lds.pools.bergson import render_trajectory_yaml
    from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION

    recipe = PRESETS["wikitext2_gpt2_bae"]["recipe"]
    y = render_trajectory_yaml(3, root="/w/traj", recipe=recipe, seed_base=30_000,
                               n_train=4656, units_per_epoch=2, repo=WIKITEXT2_REPO,
                               revision=WIKITEXT2_REVISION)
    for line in ("run_path: /w/traj/run_03", "seed: 30003", "precision: fp32",
                 "optimizer: adamw", "batch_size: 8", "num_epochs: 3", "lr: 3e-05",
                 "weight_decay: 0.01", "train_mode: true", "save_mode: interval",
                 "save_interval: 291", "save_optimizer_state: none",
                 f"data_kwargs: revision={recipe.dataset_revision}"):
        assert line in y, line
    assert WIKITEXT2_REVISION == recipe.dataset_revision         # one pin, two readers


def test_wikitext2_is_loaded_whole():
    """bergson indexes documents by their hosted row, so any subsample would mis-pair
    every tauq with the pool -- refused before torch is even imported."""
    from corrected_lds.pools.datasets import load_wikitext2

    with pytest.raises(ValueError, match="used whole"):
        load_wikitext2(n_train=1000)


def test_bootstrap_pins_bergson_and_its_transformers():
    """bergson does not pin transformers; the venv must get the version the released
    pool was reproduced with, resolved together with bergson's own dependencies."""
    from corrected_lds.pools.bergson import render_bootstrap

    b = render_bootstrap(venv="/v")
    assert '/v/bin/pip install -q bergson "transformers==5.18.0"\n' in b
    assert "bergson.git@0ad2e5b0ce09207d2af0e80383fe2d952dd8e747" in b


# -- the eval-mode replication ---------------------------------------------- #


def test_replication_units_keep_the_bank_layout_and_cut_new_subsets_from_their_start():
    """Subsets 0-99 as the bank cut them (7 x 13 + 9), 100-299 cut from 100 (15 x 13 + 5),
    alike for every seed; seeds 1004-1013 only need 100-299."""
    from corrected_lds.pools.bergson import replication_units

    old, new = range(1004, 1014), range(1014, 1024)
    units = replication_units([(s, [(100, 300)]) for s in old]
                              + [(s, [(0, 100), (100, 300)]) for s in new], 13)
    assert len(units) == 10 * 16 + 10 * 24
    assert sum(hi - lo for _, lo, hi in units) == 10 * 200 + 10 * 300
    assert units[:2] == [(1004, 100, 113), (1004, 113, 126)] and units[15] == (1004, 295, 300)
    s14 = [(lo, hi) for s, lo, hi in units if s == 1014]
    assert s14[:8] == [(0, 13), (13, 26), (26, 39), (39, 52), (52, 65), (65, 78), (78, 91),
                       (91, 100)] and s14[8:] == [(lo, hi) for _, lo, hi in units[:16]]
    assert units[:160] == [u for u in units if u[0] in old]        # the R=10 stage first
    with pytest.raises(ValueError, match="overlap"):
        replication_units([(1004, [(0, 100), (91, 120)])], 13)


def test_eval_pool_assembles_chunks_and_refuses_gaps(tmp_path):
    from corrected_lds.banks.artifacts import load_pool
    from corrected_lds.pools.bergson import build_eval_pool

    R, M, Q = 2, 6, 3
    masks = sample_subsets(M, 40, 0.5, 777)
    y = np.random.default_rng(0).normal(3.0, 0.1, (R, M, Q))
    dirs = []
    for r in range(R):
        d = tmp_path / f"r{r}"
        d.mkdir()
        for lo, hi in ((0, 4), (4, 6)):
            np.savez(d / f"eval_{lo}_{hi}.npz", losses=y[r, lo:hi], lo=lo, hi=hi,
                     readout="eval")
        dirs.append(str(d))
    out = build_eval_pool(str(tmp_path / "pool.npz"), dirs, [1004, 1005], masks,
                          alpha=0.5, recipe={"alpha": 0.5}, Q=Q)
    pool = load_pool(out)
    assert np.array_equal(pool.margins, -y) and pool.seed_mode == "shared-per-replica"
    with np.load(out) as z:
        assert str(z["readout"]) == "eval" and "baselines" not in z.files
    os.remove(os.path.join(dirs[1], "eval_4_6.npz"))
    with pytest.raises(ValueError, match="INCOMPLETE: 2 of 6"):
        build_eval_pool(str(tmp_path / "p2.npz"), dirs, [1004, 1005], masks,
                        alpha=0.5, recipe={"alpha": 0.5}, Q=Q)
    np.savez(os.path.join(dirs[1], "eval_3_6.npz"), losses=y[1, 3:6], lo=3, hi=6)
    with pytest.raises(ValueError, match="subset 3 is in both eval_0_4.npz and eval_3_6"):
        build_eval_pool(str(tmp_path / "p3.npz"), dirs, [1004, 1005], masks,
                        alpha=0.5, recipe={"alpha": 0.5}, Q=Q)


def test_collect_refuses_overlapping_chunk_files(tmp_path):
    """Two layouts in one replica directory: refuse rather than let the later file win."""
    M, Q = 8, 2
    loss = np.full((M, Q), 3.0)
    _write_chunk(tmp_path, 0, 5, Q, base=np.full(Q, 3.1), loss=loss)
    _write_chunk(tmp_path, 4, 8, Q, base=np.full(Q, 3.1), loss=loss)
    with pytest.raises(ValueError, match="subset 4 is in both"):
        collect(str(tmp_path), M, Q)


def test_score_store_round_trip_and_refuses_unwritten(tmp_path):
    """bergson's structured scores.bin + info.json -> (n_train, Q), as stored; built here
    in the exact layout bergson writes (score_i float32 + written_i bool per column)."""
    import json as _json

    from corrected_lds.pools.bergson import read_score_store

    n, Q = 7, 3
    names = [f for q in range(Q) for f in (f"score_{q}", f"written_{q}")]
    formats = [f for _ in range(Q) for f in ("<f4", "|b1")]
    dt = np.dtype({"names": names, "formats": formats})
    offs = [dt.fields[k][1] for k in names]
    want = np.random.default_rng(0).normal(size=(n, Q)).astype(np.float32)
    mm = np.memmap(tmp_path / "scores.bin", dtype=dt, mode="w+", shape=(n,))
    for q in range(Q):
        mm[f"score_{q}"] = want[:, q]
        mm[f"written_{q}"] = True
    mm.flush()
    info = {"num_scores": Q, "dtype": {"names": names, "formats": formats, "offsets": offs,
                                       "itemsize": dt.itemsize}}
    (tmp_path / "info.json").write_text(_json.dumps(info))
    assert np.array_equal(read_score_store(str(tmp_path)), want.astype(np.float64))
    mm[2]["written_1"] = False
    mm.flush()
    with pytest.raises(ValueError, match="1 of 3 score columns"):
        read_score_store(str(tmp_path))
