"""TRAK/EK-FAC tests (torch; library calls stubbed): Eq. 15's separated average (mean(scores) *
mean(Q), not mean(scores*Q)), the proj_dim guard, EK-FAC's text/image batches, tauq pairing,
``compute-tau`` through the real ``compute_tauq`` signature, and the multi-GPU pieces cache
matching one sequential pass bit for bit.
"""


import numpy as np
import pytest

torch = pytest.importorskip("torch")

import corrected_lds.tda.trak as trakmod
from corrected_lds import cli
from corrected_lds.banks.artifacts import load_pool, load_tauq, predictions, write_tauq
from corrected_lds.banks.presets import PRESETS, Resnet9Recipe, bank_hashes
from corrected_lds.pools.runner import PoolTask, build_pool
from corrected_lds.tda.ekfac import EkfacArgs
from corrected_lds.tda.trak import (
    check_proj_dim,
    compute_tauq,
)

N, Q = 40, 3


# -- Eq. 15 aggregation (the math, with library calls stubbed) -------------- #
def _fake_ckpts(tmp_path, M):
    ckpts = []
    for m in range(M):
        p = tmp_path / f"c{m}.pt"
        torch.save({"w": torch.zeros(1)}, p)
        ckpts.append(str(p))
    return ckpts


def test_compute_tauq_is_separated_average_not_mean_of_products(tmp_path, monkeypatch):
    n, q, M = 6, 3, 4
    rng = np.random.default_rng(0)
    scores = [rng.normal(size=(n, q)) for _ in range(M)]
    # each model's Q correlates with its own scores so cov_m(scores, Q) != 0
    qs = [0.5 + 0.2 * m + 0.4 * scores[m].mean(axis=1) for m in range(M)]
    pieces = iter(zip(scores, qs))
    monkeypatch.setattr(trakmod, "_single_model_pieces", lambda *a, **k: next(pieces))

    data = {"meta": {"n_train": n, "n_targets": q}}
    out = compute_tauq(_fake_ckpts(tmp_path, M), data, (lambda: None), proj_dim=8,
                       device="cpu", work_dir=str(tmp_path / "w"),
                       scratch_dir=str(tmp_path / "s"))

    expected = np.mean(scores, axis=0) * np.mean(qs, axis=0)[:, None]      # mean(s)*mean(Q)
    old = np.mean([scores[m] * qs[m][:, None] for m in range(M)], axis=0)  # mean(s*Q)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert not np.allclose(out, old, atol=1e-6), "Eq.15 fix is a no-op on this fixture"


def test_plain_mean_and_both_come_from_one_pass(tmp_path, monkeypatch):
    """plain-mean is the mean of per-member products (the EK-FAC/TracIn-style combine), and
    asking for both estimators featurizes each member once (one pass, three running sums)."""
    n, q, M = 6, 3, 4
    rng = np.random.default_rng(2)
    scores = [rng.normal(size=(n, q)) for _ in range(M)]
    qs = [0.5 + 0.3 * scores[m].mean(axis=1) for m in range(M)]    # correlated on purpose
    calls = {"n": 0}

    def pieces(*a, **k):
        i = calls["n"]
        calls["n"] += 1
        return scores[i], qs[i]

    monkeypatch.setattr(trakmod, "_single_model_pieces", pieces)
    data = {"meta": {"n_train": n, "n_targets": q}}
    ckpts = _fake_ckpts(tmp_path, M)
    res = compute_tauq(ckpts, data, (lambda: None), device="cpu",
                       work_dir=str(tmp_path / "w"),
                       aggregation=("separated-avg-eq15", "plain-mean"))
    assert calls["n"] == M                                         # one pass for both
    pm = np.mean([scores[m] * qs[m][:, None] for m in range(M)], axis=0)
    eq = np.mean(scores, axis=0) * np.mean(qs, axis=0)[:, None]
    np.testing.assert_allclose(res["plain-mean"], pm, atol=1e-12)
    np.testing.assert_allclose(res["separated-avg-eq15"], eq, atol=1e-12)
    assert not np.allclose(pm, eq, atol=1e-6)                      # the cov term is real

    # a plain string returns the bare array (back-compat), from a fresh accumulator
    rerun = iter(zip(scores, qs))
    monkeypatch.setattr(trakmod, "_single_model_pieces", lambda *a, **k: next(rerun))
    out2 = compute_tauq(ckpts, data, (lambda: None), device="cpu",
                        work_dir=str(tmp_path / "w2"), aggregation="plain-mean")
    np.testing.assert_allclose(out2, pm, atol=1e-12)
    # an eq15-only accumulator has no prod_sum, so a later plain-mean request against it fails
    third = iter(zip(scores, qs))
    monkeypatch.setattr(trakmod, "_single_model_pieces", lambda *a, **k: next(third))
    compute_tauq(ckpts[:2], data, (lambda: None), device="cpu",
                 work_dir=str(tmp_path / "w3"))
    with pytest.raises(RuntimeError, match="prod_sum"):
        compute_tauq(ckpts, data, (lambda: None), device="cpu",
                     work_dir=str(tmp_path / "w3"), aggregation="plain-mean")


# -- guards + the tauq pairing contract ------------------------------------- #
def test_proj_dim_guard():
    # every canonical preset must pass without --force (guard at n/2, disaster was k ~ n)
    check_proj_dim(20_480, 50_000)                                 # cifar10
    check_proj_dim(4_096, 50_000)                                  # qnli
    check_proj_dim(2_048, 10_000)                                  # pilot
    check_proj_dim(15_360, 1_281_167)                              # imagenet
    with pytest.raises(ValueError, match="pure-noise"):
        check_proj_dim(20_480, 20_000)                             # the measured k ~ n disaster


def test_ekfac_task_is_modality_agnostic():
    """One Task, two batch shapes: ``(x, y)`` and ``(ids, token_type, mask, y)``.  The mask
    keeps PAD tokens out of the Fisher (kronfluence zeroes them out of the activation and the
    bias ones column, ``module/linear.py:34-44``); images have no mask and must get None."""
    pytest.importorskip("kronfluence")
    from corrected_lds.tda.ekfac import _make_task

    class _Img(torch.nn.Module):
        def forward(self, x):
            return x

    class _Txt(torch.nn.Module):
        # the contract: all three inputs reach forward(), and get_attention_mask returns the mask
        def forward(self, ids, tt, am):
            assert am is not None
            return ids + tt

    task = _make_task(EkfacArgs())
    y = torch.tensor([0, 1])
    logits = torch.randn(2, 3, dtype=torch.float64)
    img = (logits, y)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    txt = (logits, torch.zeros(2, 3, dtype=torch.float64), mask.double(), y)

    assert task.get_attention_mask(img) is None
    assert torch.equal(task.get_attention_mask(txt), mask.double())
    # the same estimator either way: text's extra inputs sum to the image's logits
    assert torch.allclose(task.compute_train_loss(img, _Img()),
                          task.compute_train_loss(txt, _Txt()))
    assert torch.allclose(task.compute_measurement(img, _Img()),
                          task.compute_measurement(txt, _Txt()))


def test_ekfac_datasets_follow_the_bank_modality():
    """``_datasets`` keys on ``"input_ids" in data`` as tracin.py does: one modality per bank."""
    from corrected_lds.tda.ekfac import _datasets
    n, q, L = 5, 3, 7
    img = {"X_train": np.zeros((n, 2)), "y_train": np.zeros(n),
           "X_query": np.zeros((q, 2)), "y_query": np.zeros(q)}
    tr, qy = _datasets(img)
    assert len(tr) == n and len(tr[0]) == 2 and len(qy) == q
    # load_qnli's real key names: text labels are "labels"/"q_labels", not "y_train"/"y_query"
    txt = {"input_ids": np.zeros((n, L)), "token_type_ids": np.zeros((n, L)),
           "attention_mask": np.ones((n, L)), "labels": np.zeros(n),
           "q_input_ids": np.zeros((q, L)), "q_token_type_ids": np.zeros((q, L)),
           "q_attention_mask": np.ones((q, L)), "q_labels": np.zeros(q)}
    tr, qy = _datasets(txt)
    assert len(tr) == n and len(tr[0]) == 4 and len(qy) == q
    assert tr[0][3].dtype == torch.long        # the label, whatever the modality
    # a text dict carrying the image label name must fail loudly, not score the wrong column
    mislabelled = {k: v for k, v in txt.items() if k != "labels"}
    mislabelled["y_train"] = np.zeros(n)
    with pytest.raises(KeyError):
        _datasets(mislabelled)


def _pool(tmp_path):
    recipe = Resnet9Recipe(epochs=1)
    task = PoolTask(recipe=recipe,
                    train_cell=lambda d, sub, seed: (int(seed), np.asarray(sub)),
                    readout=lambda m, d: np.arange(Q, dtype=float) + m[0] * 1e-3)
    th, qh = bank_hashes("cifar10")
    data = {"meta": {"n_train": N, "n_targets": Q, "dataset": "cifar10",
                     "train_idx": list(range(N)), "query_idx": [5, 1, 9],
                     "train_hash": th, "query_hash": qh}}
    return build_pool(str(tmp_path / "p"), task, data, M=4, R=2, verbose=False), data


def test_write_tauq_passes_the_pairing_enforcement(tmp_path):
    pool_path, data = _pool(tmp_path)
    pool = load_pool(pool_path)
    tauq = np.random.default_rng(0).normal(size=(N, Q)).astype(np.float32)
    out = write_tauq(str(tmp_path / "p" / "tauq_trak_e4.npz"), tauq, data["meta"],
                     n_models=4, proj_dim=512, provenance={"method": "trak"})
    tau = load_tauq(out)
    g = predictions(pool, tau)                                     # pairing accepted
    assert g.shape == (4, Q)
    assert tau.provenance["method"] == "trak" and tau.n_models == 4


# -- CLI: compute-tau through the real compute_tauq(_text) signatures ------- #
def _wire_compute_tau(monkeypatch, tmp_path, preset, piece_attr, extra_argv=()):
    """Run ``compute-tau --method trak`` with only the data loader, ensemble trainer and per-model
    traker call (``piece_attr``) stubbed: a kwarg the tau function rejects fails here, not after a
    GPU ensemble trains.  Returns ``(exit_code, kwargs seen by the per-model stub, tauq path)``."""
    import os

    p = PRESETS[preset]
    n, q = 10_000, 8                        # > 2*proj_dim, so the pure-noise guard passes
    th, qh = bank_hashes(p["dataset"])
    data = {"meta": {"dataset": p["dataset"], "n_train": n, "n_targets": q,
                     "n_classes": 2, "in_ch": 3,
                     "train_idx": list(range(n)), "query_idx": list(range(q)),
                     "train_hash": th, "query_hash": qh}}
    monkeypatch.setattr(cli, "_load_preset_data",
                        lambda preset_, device: (p, p["recipe"], data))

    def fake_train_ensemble(ckpt_dir, data_, recipe_, n_members, train_fn, verbose=True,
                            members=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        paths = [os.path.join(ckpt_dir, f"member_{m:03d}.pt") for m in range(n_members)]
        for m, path in enumerate(paths):    # loadable: _ensemble_eq15 torch.loads them
            if members is None or m in members:
                torch.save({"w": torch.zeros(1)}, path)
        return paths

    monkeypatch.setattr(trakmod, "train_ensemble", fake_train_ensemble)

    seen = []

    def fake_pieces(sd, data_, model_ctor, **kw):
        seen.append(kw)
        return np.ones((n, q)), np.full(n, 0.5)

    monkeypatch.setattr(trakmod, piece_attr, fake_pieces)
    # 5 = the pilot/qnli e_features: the smallest budget the unit guard allows
    rc = cli.main(["compute-tau", "--preset", preset, "--tag", "t", "--root",
                   str(tmp_path), "--device", "cpu", "--n-models", "5",
                   *extra_argv, "--yes"])
    # the Eq. 15 tauq is named for the preset's e_features (trak5 here, trak10 on cifar10)
    ef = p["trak"]["e_features"]
    # E counts replicas: 5 models at e_features=5 are one Eq. 15 unit -> _e1
    return rc, seen, tmp_path / "t" / f"tauq_trak{ef}_e{5 // ef}.npz"


def test_cli_compute_tau_wiring_image(tmp_path, monkeypatch):
    rc, seen, out = _wire_compute_tau(monkeypatch, tmp_path, "pilot",
                                      "_single_model_pieces")
    assert rc == 0 and len(seen) == 5
    for kw in seen:                          # the preset knobs must arrive per model
        assert kw["use_half_precision"] is False          # tauq identity (traker default fp16)
        assert kw["require_cuda_projector"] is False
        assert kw["featurize_augment"] is not None        # pilot: featurize-on-augmented on
    tau = load_tauq(out)                     # the written artifact round-trips validation
    assert tau.n_models == 5 and tau.proj_dim == PRESETS["pilot"]["trak"]["proj_dim"]
    assert tau.provenance["use_half_precision"] is False
    assert tau.provenance["aggregation"] == "separated-avg-eq15"
    np.testing.assert_allclose(tau.tauq, 0.5)             # mean(ones) * mean(0.5)


# -- pieces cache + --members: multi-GPU featurization ---------------------- #
def _correlated_pieces(n, q, M, seed=0):
    rng = np.random.default_rng(seed)
    scores = [rng.normal(size=(n, q)) for _ in range(M)]
    qs = [0.5 + 0.2 * m + 0.4 * scores[m].mean(axis=1) for m in range(M)]
    return scores, qs


def test_pieces_cache_reproduces_the_sequential_pass_bit_for_bit(tmp_path, monkeypatch):
    """Two 'GPUs' cache disjoint members; the accumulating pass over the cache must equal one
    sequential pass bit for bit: both estimators, the prefix snapshot and both pool_tda slices."""
    n, q, M = 12, 4, 4
    scores, qs = _correlated_pieces(n, q, M)
    subsets = (np.random.default_rng(1).random((5, n)) < 0.5).astype(np.int8)
    data = {"meta": {"n_train": n, "n_targets": q}}
    ckpts = _fake_ckpts(tmp_path, M)
    kw = {"device": "cpu", "aggregation": ("separated-avg-eq15", "plain-mean"),
          "unit_size": 2, "subsets": subsets, "member_slices": True, "snapshot_at": (2,),
          "provenance": {"method": "trak", "n_models": M}}

    seq = iter(zip(scores, qs))                               # (A) the sequential reference
    monkeypatch.setattr(trakmod, "_single_model_pieces", lambda *a, **k: next(seq))
    ref = compute_tauq(ckpts, data, (lambda: None), work_dir=str(tmp_path / "wA"), **kw)

    calls = []                                                # (B) pieces-only x2, then accumulate

    def by_member(sd, data_, model_ctor, **k):
        calls.append(k["model_id"])
        return scores[k["model_id"]], qs[k["model_id"]]

    monkeypatch.setattr(trakmod, "_single_model_pieces", by_member)
    wB = str(tmp_path / "wB")
    r1 = compute_tauq(ckpts, data, (lambda: None), work_dir=wB, members=[0, 1], **kw)
    r2 = compute_tauq(ckpts, data, (lambda: None), work_dir=wB, members=[3, 2], **kw)
    assert r1 == {"pieces": [0, 1]} and r2 == {"pieces": [2, 3]}
    assert sorted(calls) == [0, 1, 2, 3]
    assert not (tmp_path / "wB" / "tauq_acc.npz").exists()    # pieces mode touches no sums
    assert sorted(p.name for p in (tmp_path / "wB" / "pieces").iterdir()) == \
        [f"m{m:03d}.npz" for m in range(M)]
    # a cached member is skipped, not refeaturized
    assert compute_tauq(ckpts, data, (lambda: None), work_dir=wB, members=[1],
                        **kw) == {"pieces": [1]}
    assert sorted(calls) == [0, 1, 2, 3]

    out = compute_tauq(ckpts, data, (lambda: None), work_dir=wB, **kw)
    assert sorted(calls) == [0, 1, 2, 3]                      # the pass featurized nothing
    for key in ("separated-avg-eq15", "plain-mean", "g_units", "g_members_pm"):
        assert np.array_equal(out[key], ref[key]), key
    for a in ("separated-avg-eq15", "plain-mean"):
        assert np.array_equal(out["snapshots"][2][a], ref["snapshots"][2][a]), a
