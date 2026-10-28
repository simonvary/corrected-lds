"""GPT-2 / WikiText attributors on CPU, with tiny configs: the model our EK-FAC and TracIn
score, the import of bergson-trained checkpoints, the causal-LM gradient paths, and TracIn's
sub-epoch checkpoint units.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

import corrected_lds.tda.tracin as tracinmod
from corrected_lds.banks.presets import (
    PRESETS,
    Gpt2WikitextBaeRecipe,
)
from corrected_lds.pools.bergson import import_trajectory
from corrected_lds.pools.tasks.wikitext_gpt2 import (
    hf_to_ours,
    make_model,
    per_doc_losses,
)

V, T = 31, 12


def _cfg():
    return GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=V, n_positions=32)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return make_model(device="cpu", config=_cfg())


def _loop_grads(model, ids):
    """Per-document gradients the slow way, in ``named_parameters`` order."""
    out = []
    for i in range(ids.shape[0]):
        model.zero_grad()
        per_doc_losses(model(ids[i:i + 1]), ids[i:i + 1])[0].backward()
        out.append(torch.cat([p.grad.reshape(-1) for _, p in model.named_parameters()]))
    return torch.stack(out)


# -- the model and the checkpoint import ------------------------------------ #
def test_conv1d_swap_is_exact():
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(_cfg()).eval()
    ours = make_model(device="cpu", config=_cfg())
    ours.load_state_dict(hf_to_ours(hf.state_dict()), strict=True)
    ids = torch.randint(0, V, (3, T))
    with torch.no_grad():
        assert torch.equal(hf(input_ids=ids).logits, ours(ids))
    assert ours.lm.lm_head.weight is ours.lm.transformer.wte.weight


def test_hf_to_ours_reads_the_layout_instead_of_assuming_it():
    """bergson exports with its transformers; a Linear-layout export must pass through
    untransposed, and a dropped tied lm_head must come back."""
    a = _tiny(0)
    linear_layout = {k[len("lm."):]: v for k, v in a.state_dict().items()
                     if k != "lm.lm_head.weight"}
    b = _tiny(1)
    b.load_state_dict(hf_to_ours(linear_layout), strict=True)
    ids = torch.randint(0, V, (2, T))
    with torch.no_grad():
        assert torch.equal(a(ids), b(ids))


def test_import_trajectory_round_trip(tmp_path):
    """HF save_pretrained dirs -> our checkpoints, strictly loaded and logit-exact."""
    torch.manual_seed(0)
    hfs = {s: GPT2LMHeadModel(_cfg()).eval() for s in (5, 10)}
    for s, hf in hfs.items():
        hf.save_pretrained(str(tmp_path / "exported" / f"checkpoint-{s}"),
                           safe_serialization=True)
    outs = [str(tmp_path / "ours" / f"unit_{u:03d}_of2.pt") for u in (1, 2)]
    import_trajectory(str(tmp_path / "exported"), outs, [5, 10], _tiny())
    ids = torch.randint(0, V, (2, T))
    for s, out in zip((5, 10), outs):
        m = _tiny(9)
        m.load_state_dict(torch.load(out, weights_only=True))
        with torch.no_grad():
            assert torch.equal(m(ids), hfs[s](input_ids=ids).logits)
    with pytest.raises(FileNotFoundError, match="export_checkpoints"):
        import_trajectory(str(tmp_path / "exported"), outs[:1], [15], _tiny())


# -- TracIn: sub-epoch units + the causal-LM gradients ---------------------- #


def test_bae_units_are_six_half_epochs_carrying_the_same_total_weight():
    rec = PRESETS["wikitext2_gpt2_bae"]["recipe"]
    half = tracinmod.TracinArgs(ckpt_every=1, units_per_epoch=2)
    assert half.to_dict()["units_per_epoch"] == 2
    assert tracinmod.resolve_ckpt_epochs(rec.epochs, 1, (), 2) == (1, 2, 3, 4, 5, 6)
    w2 = tracinmod.ckpt_lr_weights(rec, 4_656, half)
    w1 = tracinmod.ckpt_lr_weights(rec, 4_656, tracinmod.TracinArgs(ckpt_every=1))
    assert np.allclose(w2, rec.peak_lr / 2) and len(w2) == 6
    assert w2.sum() == pytest.approx(w1.sum())      # same trajectory, finer sampling
    assert tracinmod.trajectory_paths("T", 1, (6,), 2)[0].endswith("run_01/unit_006_of2.pt")
    with pytest.raises(ValueError, match="equal units"):
        tracinmod.epoch_mean_lrs(rec, 4_656, units_per_epoch=4)


def test_lm_per_sample_grads_match_autograd():
    model = _tiny()
    ids = torch.randint(0, V, (5, T))
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    g = tracinmod._grad_chunk_fn_lm(model, params, buffers, ids)(0, 5)
    assert torch.allclose(g, _loop_grads(model, ids), atol=1e-6)


def test_tracin_lm_tauq_is_the_weighted_gradient_dot(tmp_path):
    """Two runs x two half-epoch units, unprojected, vs brute-force sum_t w_t <grad_i, grad_q>."""
    rec = Gpt2WikitextBaeRecipe(epochs=1, batch=4)          # 8 docs -> 2 steps per epoch
    args = tracinmod.TracinArgs(ckpt_every=1, units_per_epoch=2)
    data = {"lm_ids": torch.randint(0, V, (8, T)), "q_lm_ids": torch.randint(0, V, (3, T)),
            "meta": {"n_train": 8, "n_targets": 3}}
    trajs, want = [], np.zeros((8, 3))
    weights = tracinmod.ckpt_lr_weights(rec, 8, args)
    for r in range(2):
        paths = tracinmod.trajectory_paths(str(tmp_path), r, (1, 2), 2)
        for t, path in enumerate(paths):
            m = _tiny(10 * r + t)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(m.state_dict(), path)
            want += weights[t] * (_loop_grads(m, data["lm_ids"])
                                  @ _loop_grads(m, data["q_lm_ids"]).T).double().numpy()
        trajs.append(paths)
    tau = tracinmod.compute_tauq_tracin(trajs, data, lambda: _tiny(99), recipe=rec,
                                        args=args, proj_dim=None, device="cpu", batch=4,
                                        work_dir=str(tmp_path / "w"))
    assert np.allclose(tau, want / 2, rtol=1e-4, atol=1e-9)


def test_tracin_traces_only_the_tracked_modules(tmp_path):
    """``tracked`` restricts the dot product to those modules' weight + bias (the untracked tied
    wte shapes the forward but adds no coordinates); a name owning no parameter is refused."""
    tracked = ("lm.transformer.h.0.attn.c_attn", "lm.transformer.h.1.mlp.c_fc")
    rec = Gpt2WikitextBaeRecipe(epochs=1, batch=4)
    args = tracinmod.TracinArgs(ckpt_every=1, units_per_epoch=2, tracked=tracked)
    assert args.to_dict()["tracked"] == list(tracked)
    data = {"lm_ids": torch.randint(0, V, (8, T)), "q_lm_ids": torch.randint(0, V, (3, T)),
            "meta": {"n_train": 8, "n_targets": 3}}
    m0 = _tiny(0)
    keep = torch.tensor([k.rsplit(".", 1)[0] in tracked
                         for k, p in m0.named_parameters() for _ in range(p.numel())])
    assert 0 < int(keep.sum()) < keep.numel()
    weights = tracinmod.ckpt_lr_weights(rec, 8, args)
    paths = tracinmod.trajectory_paths(str(tmp_path), 0, (1, 2), 2)
    want = np.zeros((8, 3))
    for t, path in enumerate(paths):
        m = _tiny(t)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(m.state_dict(), path)
        want += weights[t] * (_loop_grads(m, data["lm_ids"])[:, keep]
                              @ _loop_grads(m, data["q_lm_ids"])[:, keep].T).double().numpy()
    tau = tracinmod.compute_tauq_tracin([paths], data, lambda: _tiny(99), recipe=rec,
                                        args=args, proj_dim=None, device="cpu", batch=4,
                                        work_dir=str(tmp_path / "w"))
    assert np.allclose(tau, want, rtol=1e-4, atol=1e-9)
    with pytest.raises(ValueError, match="own no parameters"):
        tracinmod._traced_params(m0, ("lm.transformer.h.0.attn.nope",))


def test_tracked_gradients_carry_no_outer_graph():
    """Untracked parameters must not stay trainable, or every per-sample gradient drags an
    autograd graph through them and pins the chunk's activations (OOM on 32 GB at batch 8)."""
    model = _tiny()
    params = tracinmod._traced_params(model, ("lm.transformer.h.0.attn.c_attn",))
    assert not any(p.requires_grad for p in model.parameters())
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    g = tracinmod._grad_chunk_fn_lm(model, params, buffers, torch.randint(0, V, (3, T)))(0, 3)
    assert g.grad_fn is None and not g.requires_grad


@pytest.mark.parametrize("per_ckpt", [False, True])
def test_projection_matrix_is_per_run_or_per_checkpoint(tmp_path, monkeypatch, per_ckpt):
    """One projection seed per run, or a fresh one per (run, checkpoint): run r's
    checkpoint t projects under model_id r*T + t, train and query sides alike."""
    seen = []

    class _Stub:
        def project(self, g, model_id):
            seen.append(int(model_id))
            return g[:, :4]

    monkeypatch.setattr(tracinmod, "_make_projector", lambda *a, **k: _Stub())
    rec = Gpt2WikitextBaeRecipe(epochs=1, batch=4)
    args = tracinmod.TracinArgs(ckpt_every=1, units_per_epoch=2,
                                projection_per_checkpoint=per_ckpt)
    assert ("projection_per_checkpoint" in args.to_dict()) is per_ckpt
    data = {"lm_ids": torch.randint(0, V, (8, T)), "q_lm_ids": torch.randint(0, V, (3, T)),
            "meta": {"n_train": 8, "n_targets": 3}}
    trajs = []
    for r in range(2):
        paths = tracinmod.trajectory_paths(str(tmp_path), r, (1, 2), 2)
        for t, path in enumerate(paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(_tiny(10 * r + t).state_dict(), path)
        trajs.append(paths)
    tracinmod.compute_tauq_tracin(trajs, data, lambda: _tiny(99), recipe=rec, args=args,
                                  proj_dim=512, device="cpu", batch=8,
                                  work_dir=str(tmp_path / "w"))
    # per checkpoint: one train chunk + one query chunk, both under the same id
    want = [0, 1, 2, 3] if per_ckpt else [0, 0, 1, 1]
    assert seen == [i for i in want for _ in range(2)]


# -- EK-FAC on a causal LM -------------------------------------------------- #


def test_read_retrained_is_the_eval_mode_readout(tmp_path):
    """validate --save_models dirs -> eval-mode query losses, model by model."""
    from corrected_lds.pools.bergson import read_retrained
    from corrected_lds.pools.tasks.wikitext_gpt2 import query_losses

    torch.manual_seed(0)
    data = {"q_lm_ids": torch.randint(0, V, (5, T))}
    want = []
    for i in (3, 4):
        hf = GPT2LMHeadModel(_cfg()).eval()
        hf.save_pretrained(str(tmp_path / f"subset_{i}"), safe_serialization=True)
        ours = _tiny(9)
        ours.load_state_dict(hf_to_ours(hf.state_dict()))
        want.append(query_losses(ours, data, "cpu"))
    got = read_retrained(str(tmp_path), 3, 5, _tiny(7), data, device="cpu")
    assert got.shape == (2, 5) and np.allclose(got, np.stack(want))
    with pytest.raises(FileNotFoundError, match="save_models"):
        read_retrained(str(tmp_path), 5, 6, _tiny(7), data, device="cpu")
