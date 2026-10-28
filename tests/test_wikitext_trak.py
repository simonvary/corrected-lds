"""TRAK on the GPT-2 / WikiText bank, on CPU with tiny configs: the causal-LM model output and
its out-to-loss term, the tracked-only gradient computer, and the members bergson trains on
their own halves.
"""
import dataclasses
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import GPT2Config

import corrected_lds.tda.trak as trakmod
from corrected_lds.banks.presets import PRESETS
from corrected_lds.pools import bergson
from corrected_lds.pools.tasks.wikitext_gpt2 import make_model, per_doc_losses

V, T = 31, 12
TINY_TRACKED = tuple(f"lm.transformer.h.{i}.{m}" for i in range(2)
                     for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"))
BAE = PRESETS["wikitext2_gpt2_bae"]


def _cfg():
    return GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=V, n_positions=32)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return make_model(device="cpu", config=_cfg())


def _read(path):
    with open(path) as fh:
        return fh.read()


# -- the model output ------------------------------------------------------- #
def test_lm_margins_sum_the_per_token_log_odds():
    torch.manual_seed(0)
    logits, ids = torch.randn(3, T, V), torch.randint(0, V, (3, T))
    p = logits[:, :-1].softmax(-1).gather(-1, ids[:, 1:, None])[..., 0].double()
    want = torch.log(p / (1 - p)).sum(-1)
    assert torch.allclose(trakmod.lm_margins(logits, ids).double(), want, atol=1e-4)
    assert torch.allclose(trakmod.lm_out_to_loss(logits, ids).double(),
                          (1 - p).mean(-1), atol=1e-6)


def test_loss_gradient_is_the_out_to_loss_weighted_output_gradient():
    """Per token the TRAK decomposition grad(-log p_t) = -(1 - p_t) grad f_t is exact, so a
    document's mean-loss gradient is -(1/(T-1)) sum_t (1 - p_t) grad f_t; lm_out_to_loss's
    single Q = mean_t (1 - p_t) equates the two when 1 - p_t and grad f_t do not covary."""
    model = _tiny()
    ids = torch.randint(0, V, (1, T))
    params = [p for p in model.parameters()]

    def flat_grad(scalar):
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(scalar, params)])

    g_loss = flat_grad(per_doc_losses(model(ids), ids)[0])
    acc = torch.zeros_like(g_loss)
    for t in range(T - 1):
        logits = model(ids)[:, t:t + 2]                   # predicts ids[t + 1] at slot 0
        f_t = trakmod.lm_margins(logits, ids[:, t:t + 2])[0]
        p_t = logits[0, 0].softmax(-1)[ids[0, t + 1]].detach()
        acc += (1 - p_t) * flat_grad(f_t)
    assert torch.allclose(g_loss, -acc / (T - 1), atol=1e-6)


# -- traker: tracked-only gradients and the end-to-end pieces --------------- #
def test_tracked_gradient_computer_differentiates_only_the_tracked_modules():
    pytest.importorskip("trak")
    from corrected_lds.tda.tracin import _traced_params

    model = _tiny()
    names = list(_traced_params(model, TINY_TRACKED))
    gc = trakmod._tracked_gradient_computer(TINY_TRACKED)(
        model=model, task=trakmod._lm_model_output(), grad_dim=1, dtype=torch.float32,
        device="cpu", grad_wrt=names)
    ids = torch.randint(0, V, (3, T))
    grads = gc.compute_per_sample_grad((ids,))
    assert list(grads) == names                          # never wte / lm_head / ln
    ref = make_model(device="cpu", config=_cfg())
    ref.load_state_dict(model.state_dict())
    tracked = dict(ref.named_parameters())
    for i in range(3):
        f = trakmod.lm_margins(ref(ids[i:i + 1]), ids[i:i + 1])[0]
        auto = torch.autograd.grad(f, [tracked[k] for k in names])
        for k, a in zip(names, auto):
            assert torch.allclose(grads[k][i], a, atol=1e-5), k


def test_compute_tauq_lm_carries_our_q_and_both_estimators(tmp_path):
    pytest.importorskip("trak")
    n, Q = 24, 5
    torch.manual_seed(0)
    ids = torch.randint(0, V, (n, T))
    data = {"lm_ids": ids, "q_lm_ids": ids[:Q].clone(),
            "meta": {"n_train": n, "n_targets": Q}}
    paths = []
    for m in range(2):
        p = str(tmp_path / f"member_{m:03d}.pt")
        torch.save(_tiny(100 + m).state_dict(), p)
        paths.append(p)
    ctor = lambda: make_model(device="cpu", config=_cfg())
    kw = {"tracked": TINY_TRACKED, "proj_dim": 8, "device": "cpu", "batch": 4,
          "require_cuda_projector": False}
    out = trakmod.compute_tauq_lm(paths, data, ctor, work_dir=str(tmp_path / "w"),
                                  aggregation=("separated-avg-eq15", "plain-mean"), **kw)
    for a in ("separated-avg-eq15", "plain-mean"):
        assert out[a].shape == (n, Q) and np.isfinite(out[a]).all()
    sd = torch.load(paths[0], weights_only=True)
    s0, q0 = trakmod._single_model_pieces_lm(
        sd, data, ctor, save_dir=str(tmp_path / "m0"), exp_name="x", model_id=0,
        use_half_precision=False, **{k: v for k, v in kw.items() if k != "tracked"},
        tracked=TINY_TRACKED)
    model = ctor()
    model.load_state_dict(sd)
    with torch.no_grad():
        assert np.array_equal(q0, trakmod.lm_out_to_loss(model(ids), ids).double().numpy())
    assert s0.shape == (n, Q)
    s_all, q_all = trakmod._single_model_pieces_lm(              # every parameter, tied wte
        sd, data, ctor, save_dir=str(tmp_path / "m0all"), exp_name="x", model_id=0,
        use_half_precision=False, proj_dim=8, device="cpu", batch=4,
        require_cuda_projector=False, tracked=())
    assert s_all.shape == (n, Q) and np.isfinite(s_all).all() and np.array_equal(q_all, q0)


# -- members: rows and rendered configs ------------------------------------- #
def test_member_subset_is_the_draw_train_ensemble_uses(tmp_path):
    seen = {}

    def train_fn(data, sub, seed):
        seen[seed - trakmod.ENS_SEED_BASE] = np.array(sub)
        return torch.nn.Linear(1, 1)

    recipe = dataclasses.replace(PRESETS["qnli"]["recipe"])
    data = {"meta": {"n_train": 50, "dataset": "qnli"}}
    trakmod.train_ensemble(str(tmp_path), data, recipe, 3, train_fn, verbose=False)
    for m in range(3):
        assert np.array_equal(seen[m], trakmod.member_subset(m, 50, recipe.alpha))
        assert bergson.member_rows(m, 50, recipe.alpha) == sorted(seen[m].tolist())


def test_member_scripts_render_the_bank_recipe_on_the_members_half(tmp_path):
    from corrected_lds.pools import datasets

    recipe = BAE["recipe"]
    out = str(tmp_path / "members")
    sh = bergson.write_member_scripts(
        out, 3, recipe=recipe, seed_base=trakmod.ENS_SEED_BASE, n_train=4656,
        repo=datasets.WIKITEXT2_REPO, revision=datasets.WIKITEXT2_REVISION,
        venv_bergson="/bv", venv_ours="/venv", preset="wikitext2_gpt2_bae",
        tag="wikitext2_gpt2_bae", pools_root="/pools")
    traj = bergson.render_trajectory_yaml(
        0, root="/t", recipe=recipe, seed_base=30_000, n_train=4656, units_per_epoch=2,
        repo=datasets.WIKITEXT2_REPO, revision=datasets.WIKITEXT2_REVISION)
    fields = bergson._recipe_train_fields(recipe)
    assert fields in traj
    for m in range(3):
        y = _read(os.path.join(out, "yaml", f"member_{m:03d}.yaml"))
        assert f"seed: {trakmod.ENS_SEED_BASE + m}" in y and fields in y
        assert f"dataset: {out}/data/member_{m:03d}" in y and "save_mode: final" in y
        spec = json.loads(_read(os.path.join(out, "subsets", f"member_{m:03d}.json")))
        assert spec["rows"] == bergson.member_rows(m, 4656, recipe.alpha)
    s = _read(sh)
    assert "steps=[873]" in s and "import-bergson-member" in s and "--data-meta" in s
    py = _read(os.path.join(out, "make_member_data.py"))
    assert datasets.WIKITEXT2_REPO in py and datasets.WIKITEXT2_REVISION in py
    compile(py, "make_member_data.py", "exec")
