"""SOURCE unit tests; need torch but not kronfluence (unlike ``test_source_l1_gate.py``).  The
two eigenvalue maps are the whole difference between SOURCE and our EK-FAC influence
functions, so their limits and zero-curvature behaviour are pinned exactly.
"""


import pytest

torch = pytest.importorskip("torch")

from corrected_lds.tda.source import f_backward, f_segment

LR_K = 3.7          # eta*K; nothing depends on the particular value


def _grid(dtype=torch.float64):
    """Eigenvalues spanning the regimes: exact zero, denormal-ish, and dominant."""
    return torch.tensor([0.0, 1e-12, 1e-6, 1e-3, 1.0, 1e3, 1e8], dtype=dtype)


# -- f_segment: the paper's Figure 3 limits --------------------------------- #
def test_f_segment_limit_at_zero_curvature_is_lr_k():
    """F_r(0) = lr_k exactly -- not NaN. Zero eigenvalues are ordinary in EK-FAC."""
    out = f_segment(torch.zeros(4, dtype=torch.float64), LR_K)
    assert torch.allclose(out, torch.full((4,), LR_K, dtype=torch.float64), atol=0, rtol=0)


def test_f_segment_tends_to_inverse_sigma_at_high_curvature():
    """F_r(s) -> 1/s once lr_k*s >> 1: the influence-function regime."""
    s = torch.tensor([1e3, 1e5, 1e8], dtype=torch.float64)
    assert torch.allclose(f_segment(s, LR_K), 1.0 / s, rtol=1e-12)


def test_f_segment_beats_the_naive_formula_near_zero():
    """Why expm1: (1 - exp(-x))/x cancels catastrophically as x -> 0, losing ~4 digits at
    s = 1e-12, where the true lr_k * (1 - x/2 + O(x^2)), x = lr_k*s, is lr_k in float64."""
    s = torch.tensor([1e-12], dtype=torch.float64)
    x = LR_K * s
    exact = LR_K * (1.0 - x / 2.0)
    naive = (1.0 - torch.exp(-x)) / s
    ours = f_segment(s, LR_K)
    err_ours = (ours - exact).abs().item() / LR_K
    err_naive = (naive - exact).abs().item() / LR_K
    assert err_ours < 1e-15, err_ours
    assert err_naive > 1e3 * max(err_ours, 1e-17), (err_ours, err_naive)


# -- f_backward: the segment Jacobian --------------------------------------- #
def test_f_backward_limits_and_range():
    s = _grid()
    out = f_backward(s, LR_K)
    assert torch.isfinite(out).all()
    assert out[0].item() == 1.0                 # zero curvature -> perfectly remembered
    assert (out >= 0).all() and (out <= 1.0).all()
    assert (out.diff() <= 0).all()
    assert out[-1].item() == 0.0                # high curvature -> forgotten


def test_f_backward_underflows_to_zero_not_nan():
    """exp() underflows past lr_k*sigma ~ 745 in float64, which real EK-FAC spectra reach: exactly 0
    is correct (those directions are forgotten), but NaN/Inf would break the sequential walk."""
    for dtype in (torch.float32, torch.float64):
        out = f_backward(torch.tensor([1e4, 1e30], dtype=dtype), LR_K)
        assert torch.isfinite(out).all(), out
        assert (out == 0).all(), out


def test_f_backward_contracts_so_the_segment_product_cannot_blow_up():
    """Valued in (0, 1], so applying L segments sequentially can only shrink the query
    gradient -- which is why the walk never needs to form prod(S_bar) explicitly."""
    s = torch.rand(500, dtype=torch.float64) * 10
    out = torch.ones_like(s)
    for _ in range(8):
        out = out * f_backward(s, LR_K)
    assert torch.isfinite(out).all() and (out <= 1.0).all()


# -- per-segment lr*K, against the recipe's own stepped schedule ------------ #
from corrected_lds.banks.presets import Resnet9Recipe
from corrected_lds.pools.train import triangular_lr
from corrected_lds.tda.source import (
    SourceArgs,
    checkpoint_segments,
    segment_epochs,
    segment_lr_times_steps,
)

N_TRAIN = 50_000


def _stepped_lrs(recipe, n_train):
    """The trainer's per-step LRs from the pinned schedule, independent of epoch_mean_lrs."""
    import numpy as np
    spe = max(1, int(np.ceil(n_train / recipe.batch)))
    lam = triangular_lr(recipe.epochs, recipe.peak_epoch, spe)
    return np.array([recipe.peak_lr * lam(k) for k in range(recipe.epochs * spe)]), spe


def test_segment_lr_times_steps_matches_the_stepped_schedule():
    """lr_k[l] = (mean per-step LR over segment l) * (steps in segment l) / (1-beta)."""
    import numpy as np
    recipe, args = Resnet9Recipe(), SourceArgs(segments=3, ckpt_every=4)
    lrs, spe = _stepped_lrs(recipe, N_TRAIN)
    per = recipe.epochs // args.segments                    # 8 epochs per segment
    want = []
    for l in range(args.segments):
        block = lrs[l * per * spe:(l + 1) * per * spe]      # this segment's steps
        want.append(block.mean() * block.size / (1.0 - recipe.momentum))
    got = segment_lr_times_steps(recipe, N_TRAIN, args)
    assert got.shape == (3,) and got.dtype == np.float64
    assert np.allclose(got, want, rtol=1e-12), (got, want)


def test_momentum_is_a_factor_of_ten_not_a_rounding_detail():
    """App. D.2's 1/(1-beta): at the recipe's momentum 0.9 this scales lr_k by 10.
    Omitting it would be a silent order-of-magnitude error in both maps."""
    import numpy as np
    recipe = Resnet9Recipe()
    assert recipe.momentum == 0.9
    with_m = segment_lr_times_steps(recipe, N_TRAIN, SourceArgs())
    without = segment_lr_times_steps(recipe, N_TRAIN, SourceArgs(momentum=0.0))
    assert np.allclose(with_m, without * 10.0, rtol=1e-12)


# -- the released banks' lr*K, hard-coded ----------------------------------- #
def test_released_cifar10_lr_k_is_unchanged():
    """The ``lr_times_steps`` stamped in the released ``cifar10/tauq_source_L3_e30.npz``, which
    pin reading spans off the checkpoints (not ``E // L``); if they move, that bank and its
    SOURCE figures stop reproducing.  Hard-coded so the test runs on a clean clone."""
    import numpy as np
    got = segment_lr_times_steps(Resnet9Recipe(), N_TRAIN,
                                 SourceArgs(segments=3, ckpt_every=4))
    released = [2576.8421052631584, 2476.842105263159, 826.3157894736844]
    assert np.allclose(got, released, rtol=0, atol=1e-9), (list(got), released)


def test_mnist_preset_matches_the_released_bank():
    """The mnist preset must describe the released bank's run, not merely be self-consistent, or
    `compute-tau` refuses to reuse its artifacts; values are hard-coded from its ``tauq_source_L3``
    provenance (segments=5/ckpt_every=4 is valid too, but a different run)."""
    import numpy as np

    from corrected_lds.banks.presets import PRESETS
    from corrected_lds.tda.tracin import resolve_ckpt_epochs
    p = PRESETS["mnist"]
    recipe, cfg = p["recipe"], p["source"]
    args = SourceArgs(segments=cfg["segments"], ckpt_every=cfg["ckpt_every"],
                      checkpoint_epochs=cfg["checkpoint_epochs"],
                      label_smoothing=recipe.label_smoothing,
                      grad_dtype=cfg["grad_dtype"])

    assert args.to_dict() == {                    # the released source_args, verbatim
        "segments": 3, "ckpt_every": 1, "momentum": None, "use_empirical_fisher": False,
        "factor_seed": 0, "label_smoothing": 0.0, "tracked": None,
        "grad_dtype": "float32", "checkpoint_epochs": [3, 6, 10, 13, 16, 20]}
    assert (resolve_ckpt_epochs(recipe.epochs, args.ckpt_every, args.checkpoint_epochs)
            == (3, 6, 10, 13, 16, 20))
    # ragged by necessity: six checkpoints do not divide 20 epochs evenly
    assert segment_epochs(recipe.epochs, args.ckpt_every, args.segments,
                          args.checkpoint_epochs) == ((1, 6), (7, 13), (14, 20))
    got = segment_lr_times_steps(recipe, p["n_train"], args)
    assert np.allclose(got, [172.8, 201.6, 201.6], rtol=0, atol=1e-9), list(got)

    # the tracin block scores the same six checkpoints: SOURCE reuses its trajectory
    assert tuple(p["tracin"]["checkpoint_epochs"]) == (3, 6, 10, 13, 16, 20)
    assert p["source"]["ckpt_every"] % p["tracin"]["ckpt_every"] == 0   # shareable


# -- partials and presets (no kronfluence needed) --------------------------- #

import numpy as np

from corrected_lds.tda import source as srcmod


def _write_partial(work, r, tauq, *, ckpts_md5="m", prov="{}", args_json="{}",
                   lr_k=(1.0,), version=None):
    from corrected_lds.tda.source import KRONFLUENCE_VERSION
    p = srcmod.run_partial_path(work, r)
    np.savez(p, tauq=np.asarray(tauq, dtype=np.float32), run=r, ckpts_md5=ckpts_md5,
             lr_k=np.asarray(lr_k), provenance=prov, source_args=args_json,
             kronfluence_version=KRONFLUENCE_VERSION if version is None else version)
    return p


@pytest.mark.parametrize("field", ["run", "ckpts_md5", "provenance", "source_args",
                                   "lr_k", "kronfluence_version"])
def test_partial_identity_rejects_every_field(tmp_path, field):
    """A partial is reusable only if all six fields match; lr_k comes from the recipe, so it is
    checked apart from source_args, past which a changed schedule or momentum would slip."""
    w = str(tmp_path)
    base = {"ckpts_md5": "abc", "prov": '{"a":1}', "args_json": '{"b":2}', "lr_k": (2.0, 3.0)}
    _write_partial(w, 0, np.ones((2, 2)), **base)
    path = srcmod.run_partial_path(w, 0)
    ok = {"run": 0, "ckpts_md5": "abc", "prov_json": '{"a":1}', "args_json": '{"b":2}',
              "lr_k": (2.0, 3.0)}
    assert srcmod._partial_valid(path, **ok)

    bad = dict(ok)
    if field == "run":
        bad["run"] = 1
    elif field == "ckpts_md5":
        bad["ckpts_md5"] = "zzz"
    elif field == "provenance":
        bad["prov_json"] = '{"a":2}'
    elif field == "source_args":
        bad["args_json"] = '{"b":3}'
    elif field == "lr_k":
        bad["lr_k"] = (2.0, 3.5)
    else:
        _write_partial(w, 0, np.ones((2, 2)), version="0.0.0-not-ours", **base)
    assert not srcmod._partial_valid(path, **bad)


def test_every_source_preset_divides_and_shares_the_tracin_bank():
    """C % L == 0 for each shipped block, and source.ckpt_every is a multiple of tracin.ckpt_every
    (else train_trajectories wipes the shared trajectories instead of thinning them); C counts
    scored checkpoints, via ``resolve_ckpt_epochs``, since a pinned schedule (mnist) changes it."""
    from corrected_lds.banks.presets import PRESETS
    from corrected_lds.tda.tracin import resolve_ckpt_epochs
    seen = 0
    for name, p in PRESETS.items():
        cfg = p.get("source")
        if cfg is None:
            continue
        seen += 1
        pinned = cfg.get("checkpoint_epochs", ())
        a = SourceArgs(segments=cfg["segments"], ckpt_every=cfg["ckpt_every"],
                       checkpoint_epochs=pinned)
        cps = resolve_ckpt_epochs(p["recipe"].epochs, a.ckpt_every, a.checkpoint_epochs)
        checkpoint_segments(len(cps), a.segments)                 # raises if ragged
        segment_lr_times_steps(p["recipe"], p["n_train"], a)      # raises if inconsistent
        tracin_every = p["tracin"]["ckpt_every"]
        assert a.ckpt_every % tracin_every == 0, (name, a.ckpt_every, tracin_every)
        # a pinned SOURCE schedule must also be saved by the tracin block they share
        if pinned:
            shared = resolve_ckpt_epochs(p["recipe"].epochs, tracin_every, ())
            assert set(a.checkpoint_epochs) <= set(shared), (name, pinned, shared)
    assert seen >= 2
