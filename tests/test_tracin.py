"""TracIn tests (torch, no traker: identity projector): the ``on_epoch_end`` hook is RNG-neutral,
the Eq. 1 weights read exactly the LR the trainers stepped, the estimator equals brute-force
per-sample autograd, and per-run partials are identity-checked down to checkpoint content.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import corrected_lds.tda.tracin as tracinmod
from corrected_lds.banks.presets import (
    BertRecipe,
    Resnet9Recipe,
    Resnet18Recipe,
    bank_hashes,
)
from corrected_lds.pools.train import linear_decay_lr, run_subset_sgd, triangular_lr
from corrected_lds.tda.tracin import (
    TracinArgs,
    ckpt_epochs,
    ckpt_lr_weights,
    combine_tracin_partials,
    compute_tauq_tracin,
    epoch_mean_lrs,
    run_partial_path,
    trajectory_paths,
)


# -- the epoch hook: RNG-neutral by construction ---------------------------- #
def _train_tiny(seed, hook=None):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    X = torch.randn(8, 3, generator=torch.Generator().manual_seed(9))
    y = torch.tensor([0, 1] * 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def batch_loss(b, gen, epoch):
        # draw from gen like the image path draws augmentation — the interleaving probe
        noise = torch.rand(1, device=b.device, generator=gen)
        return torch.nn.functional.cross_entropy(model(X[b]), y[b]) + 0.0 * noise.sum()

    run_subset_sgd(model, torch.arange(8), 3, 4, seed, opt, sched, "cpu",
                   batch_loss, lambda loss: (loss.backward(), opt.step())[0],
                   on_epoch_end=hook)
    return model


def test_on_epoch_end_is_rng_neutral():
    seen = []
    m_hooked = _train_tiny(7, hook=lambda e, m: seen.append(int(e)))
    m_bare = _train_tiny(7, hook=None)
    assert seen == [0, 1, 2]                                   # every epoch, in order
    for a, b in zip(m_hooked.parameters(), m_bare.parameters(), strict=True):
        assert torch.equal(a, b), "the hook changed training — RNG contract broken"


# -- checkpoint schedule + Eq. 1 weights vs the actually-stepped LRs -------- #
def test_ckpt_epochs():
    assert ckpt_epochs(24, 1) == tuple(range(1, 25))
    assert ckpt_epochs(24, 4) == (4, 8, 12, 16, 20, 24)        # final epoch always kept
    assert ckpt_epochs(100, 5)[-1] == 100
    with pytest.raises(ValueError, match="multiple"):
        ckpt_epochs(24, 5)
    with pytest.raises(ValueError, match="multiple"):
        ckpt_epochs(0, 1)


def _stepped_lrs(lam, peak_lr, total_steps):
    """The LR each optimizer step actually runs at, stepping LambdaLR exactly like
    ``run_subset_sgd`` (opt.step() then sched.step())."""
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(m.parameters(), lr=peak_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return np.array(lrs)


def test_epoch_mean_lrs_match_the_stepped_schedule():
    # image: triangular, epochs=4, peak at 1, n=16 @ batch 8 -> 2 steps/epoch
    r = Resnet9Recipe(epochs=4, peak_epoch=1, batch=8)
    stepped = _stepped_lrs(triangular_lr(4, 1, 2), r.peak_lr, 8).reshape(4, 2).mean(1)
    np.testing.assert_allclose(epoch_mean_lrs(r, 16), stepped, rtol=1e-12)
    # text: linear decay, epochs=2, n=8 @ batch 4 -> 2 steps/epoch
    b = BertRecipe(epochs=2, batch=4)
    stepped = _stepped_lrs(linear_decay_lr(2, 2), b.peak_lr, 4).reshape(2, 2).mean(1)
    np.testing.assert_allclose(epoch_mean_lrs(b, 8), stepped, rtol=1e-12)
    with pytest.raises(ValueError, match="no pinned LR schedule"):
        epoch_mean_lrs(Resnet18Recipe(), 32)                   # imagenet: not wired


def test_mnist_ckpt_lr_weights_follow_the_pinned_six():
    """The released bank scores six checkpoints, so Eq. 1 must produce six weights that
    sum the epochs each one covers -- 3, 3, 4, 3, 3 and 4 epochs at a constant LR."""
    from corrected_lds.banks.presets import MNIST_CKPT_EPOCHS, MnistRecipe
    r = MnistRecipe()
    w = ckpt_lr_weights(r, 6_144, TracinArgs(ckpt_every=1,
                                             checkpoint_epochs=MNIST_CKPT_EPOCHS))
    covered = np.diff((0,) + MNIST_CKPT_EPOCHS)
    assert list(covered) == [3, 3, 4, 3, 3, 4]
    np.testing.assert_allclose(w, covered * r.peak_lr, rtol=1e-12)


def test_ckpt_lr_weights_sum_thinned_intervals():
    r = Resnet9Recipe(epochs=4, peak_epoch=1, batch=8)
    means = epoch_mean_lrs(r, 16)
    w1 = ckpt_lr_weights(r, 16, TracinArgs(ckpt_every=1))
    np.testing.assert_allclose(w1, means, rtol=1e-12)          # c=1: the per-epoch etas
    w2 = ckpt_lr_weights(r, 16, TracinArgs(ckpt_every=2))
    np.testing.assert_allclose(w2, [means[:2].sum(), means[2:].sum()], rtol=1e-12)
    np.testing.assert_array_equal(
        ckpt_lr_weights(r, 16, TracinArgs(ckpt_every=2, lr_weighting="equal")), [1.0, 1.0])


# -- the estimator vs brute-force per-sample autograd ----------------------- #
N, Q = 6, 3


def _tiny_data():
    g = torch.Generator().manual_seed(3)
    X = torch.randn(N, 4, generator=g)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    th, qh = bank_hashes("cifar10")
    return {"X_train": X, "y_train": y, "X_query": X[:Q].clone(), "y_query": y[:Q].clone(),
            "meta": {"n_train": N, "n_targets": Q, "dataset": "cifar10",
                     "train_idx": list(range(N)), "query_idx": list(range(Q)),
                     "train_hash": th, "query_hash": qh}}


def _tiny_runs(tmp_path, n_runs, n_ckpts):
    """Fake trajectories: seeded random Linear(4,3) state_dicts on disk."""
    runs = []
    for r in range(n_runs):
        paths = trajectory_paths(str(tmp_path / "traj"), r, range(1, n_ckpts + 1))
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        for t, p in enumerate(paths):
            torch.manual_seed(100 * r + t)
            torch.save(torch.nn.Linear(4, 3).state_dict(), p)
        runs.append(paths)
    return runs


def _brute_tauq(runs, weights, data, ls=0.0):
    import torch.nn.functional as F
    acc = np.zeros((N, Q))
    def grads(model, X, y):
        rows = []
        for i in range(len(y)):
            li = F.cross_entropy(model(X[i:i + 1]), y[i:i + 1], label_smoothing=ls)
            gi = torch.autograd.grad(li, list(model.parameters()))
            rows.append(torch.cat([t.reshape(-1) for t in gi]).double())
        return torch.stack(rows).numpy()

    for paths in runs:
        for p, w in zip(paths, weights, strict=True):
            model = torch.nn.Linear(4, 3)
            model.load_state_dict(torch.load(p, weights_only=True))
            model.eval()
            acc += float(w) * grads(model, data["X_train"], data["y_train"]) @ \
                grads(model, data["X_query"], data["y_query"]).T
    return acc / len(runs)


def test_compute_tauq_tracin_matches_bruteforce(tmp_path):
    # epochs=2 @ ckpt_every=1 -> 2 checkpoints; schedule weights from the recipe
    recipe = Resnet9Recipe(epochs=2, peak_epoch=1, batch=4)
    args = TracinArgs()
    data = _tiny_data()
    runs = _tiny_runs(tmp_path, n_runs=2, n_ckpts=2)
    out = compute_tauq_tracin(runs, data, lambda: torch.nn.Linear(4, 3), recipe=recipe,
                              args=args, proj_dim=None, device="cpu", batch=4,
                              work_dir=str(tmp_path / "w"))
    weights = ckpt_lr_weights(recipe, N, args)
    np.testing.assert_allclose(out, _brute_tauq(runs, weights, data), rtol=1e-5, atol=1e-8)
    # queries are train rows 0..2: the self-influence diagonal is strictly positive
    assert (np.diag(out[:Q]) > 0).all()

    # full reuse: a second call touches no run (the per-run stub would raise)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError("partials should have been reused")

    orig, tracinmod._single_run_tauq = tracinmod._single_run_tauq, boom
    try:
        again = compute_tauq_tracin(runs, data, lambda: torch.nn.Linear(4, 3),
                                    recipe=recipe, args=args, proj_dim=None,
                                    device="cpu", batch=4, work_dir=str(tmp_path / "w"))
    finally:
        tracinmod._single_run_tauq = orig
    np.testing.assert_allclose(again, out, atol=1e-12)
    assert calls == []


def test_partial_identity_recompute_and_combine_errors(tmp_path):
    recipe = Resnet9Recipe(epochs=2, peak_epoch=1, batch=4)
    data = _tiny_data()
    runs = _tiny_runs(tmp_path, n_runs=2, n_ckpts=2)
    kw = {"recipe": recipe, "proj_dim": None, "device": "cpu", "batch": 4,
          "work_dir": str(tmp_path / "w")}
    compute_tauq_tracin(runs, data, lambda: torch.nn.Linear(4, 3), args=TracinArgs(), **kw)

    computed = []
    orig = tracinmod._single_run_tauq

    def spy(paths, weights, *a, mid, **k):
        computed.append(int(mid))
        return orig(paths, weights, *a, mid=mid, **k)

    tracinmod._single_run_tauq = spy
    try:
        # a drifted checkpoint invalidates only its run's partial (content md5)
        torch.manual_seed(999)
        torch.save(torch.nn.Linear(4, 3).state_dict(), runs[1][0])
        compute_tauq_tracin(runs, data, lambda: torch.nn.Linear(4, 3),
                            args=TracinArgs(), **kw)
        assert computed == [1]
        # changed args invalidate every partial
        computed.clear()
        compute_tauq_tracin(runs, data, lambda: torch.nn.Linear(4, 3),
                            args=TracinArgs(label_smoothing=0.1), **kw)
        assert computed == [0, 1]
    finally:
        tracinmod._single_run_tauq = orig

    with pytest.raises(ValueError, match="empty"):
        combine_tracin_partials(str(tmp_path / "w"), [])
    with pytest.raises(ValueError, match="duplicate"):
        combine_tracin_partials(str(tmp_path / "w"), [0, 0])
    with pytest.raises(FileNotFoundError):
        combine_tracin_partials(str(tmp_path / "w"), [5])
    assert os.path.exists(run_partial_path(str(tmp_path / "w"), 0))


def test_text_grad_chunk_fn_matches_bruteforce():
    import torch.nn.functional as F

    class TinyText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(7, 5)
            self.lin = torch.nn.Linear(5, 2)

        def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
            h = (self.emb(input_ids) * attention_mask.unsqueeze(-1).float()).mean(1)
            return self.lin(h + 0.0 * token_type_ids.float().mean(1, keepdim=True))

    torch.manual_seed(4)
    model = TinyText().eval()
    B, L = 5, 3
    ids = torch.randint(0, 7, (B, L))
    tt = torch.zeros(B, L, dtype=torch.long)
    am = torch.ones(B, L, dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0])
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    chunk = tracinmod._grad_chunk_fn_text(model, params, buffers, ids, tt, am, y, 0.0)
    got = chunk(0, B)
    for i in range(B):
        model.zero_grad()
        li = F.cross_entropy(model(input_ids=ids[i:i + 1], token_type_ids=tt[i:i + 1],
                                   attention_mask=am[i:i + 1]), y[i:i + 1])
        gi = torch.cat([t.reshape(-1) for t in
                        torch.autograd.grad(li, list(model.parameters()))])
        np.testing.assert_allclose(got[i].numpy(), gi.numpy(), rtol=1e-5, atol=1e-7)


def test_source_driver_scores_the_tracin_seeds():
    """The SOURCE driver's run r is TracIn/EK-FAC's run r: seed TRACIN_SEED_BASE + r
    (pools/bergson_source.py keeps its own copy to stay torch-free)."""
    from corrected_lds.pools.bergson_source import SourceConfig
    assert SourceConfig.__dataclass_fields__["seed_base"].default == tracinmod.TRACIN_SEED_BASE
