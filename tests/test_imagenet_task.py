"""ImageNet task tests on CPU (need torch and torchvision).  With bitwise fidelity to ffcv
unavailable by design, each piece is checked against ffcv-imagenet, torchvision or golden
values; then same seed -> bitwise-identical margins, and ``build_pool`` end to end.
"""
# Quotes ffcv-imagenet (Apache-2.0): THIRD_PARTY_NOTICES.md


import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import corrected_lds.pools.tasks.imagenet_resnet18 as task
from corrected_lds.banks.presets import Resnet18Recipe
from corrected_lds.pools.runner import PoolTask, build_pool
from corrected_lds.pools.train import run_subset_sgd


# -- reference equivalence -------------------------------------------------- #
def test_run_subset_sgd_passes_the_epoch_index():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    seen = []

    def batch_loss(b, gen, epoch):
        seen.append(epoch)
        return model(torch.zeros(len(b), 2)).sum()

    def backward_step(loss):
        loss.backward()
        opt.step()

    run_subset_sgd(model, torch.arange(4), epochs=3, batch=2, seed=1, opt=opt,
                   sched=sched, device="cpu", batch_loss=batch_loss,
                   backward_step=backward_step)
    assert seen == [0, 0, 1, 1, 2, 2]


def test_cyclic_lr_matches_the_reference_composition():
    """imagenet_cyclic_lr(step) == get_cyclic_lr + train_loop's per-iteration interp,
    both quoted verbatim from ffcv-imagenet train_imagenet.py, at every step."""
    def reference(step, epochs, peak_epoch, spe, lr):
        def get_cyclic(epoch):                       # train_imagenet.py:110
            return np.interp([epoch], [0, peak_epoch, epochs], [1e-4 * lr, lr, 0])[0]
        e, ix = divmod(step, spe)                    # train_loop: train_imagenet.py:356
        return np.interp(np.arange(spe), [0, spe], [get_cyclic(e), get_cyclic(e + 1)])[ix]

    for epochs, peak, spe, lr in [(15, 2, 626, 5.2), (16, 2, 40, 0.5), (2, 1, 1, 3.0)]:
        f = task.imagenet_cyclic_lr(epochs, peak, spe, lr, 1e-4)
        got = np.array([f(s) for s in range(epochs * spe)])
        ref = np.array([reference(s, epochs, peak, spe, lr) for s in range(epochs * spe)])
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
        assert f(0) == 1e-4 * lr                     # starts at 1e-4*lr, not 0
        assert f(epochs * spe) == 0.0                # beyond the last epoch


def test_rrc_params_exactly_match_torchvision():
    """_rrc_params replays torchvision v1 get_params' draws exactly (ffcv's get_random_crop,
    fast_crop.py:17, is the same with numpy RNG); extreme ratios hit the central-crop fallback."""
    from torchvision.transforms import RandomResizedCrop
    for h, w in [(500, 375), (375, 500), (64, 64), (20, 500), (500, 20), (7, 9)]:
        for seed in range(25):
            torch.manual_seed(seed)
            ref = RandomResizedCrop.get_params(torch.zeros(3, h, w),
                                               scale=[0.08, 1.0], ratio=[3 / 4, 4 / 3])
            gen = torch.Generator().manual_seed(seed)
            assert task._rrc_params(h, w, gen) == tuple(ref), (h, w, seed)


def test_blurpool_wraps_the_golden_module_set():
    """The stride>1 & in_ch>=16 rule wraps exactly torchvision resnet18's six layer{2,3,4} strided
    convs (incl. downsample 1x1s); the 3-channel stem stays bare."""
    m = task.make_model(Resnet18Recipe(), num_classes=10, seed=0, device="cpu")
    wrapped = {n for n, mod in m.named_modules() if isinstance(mod, task.BlurPoolConv2d)}
    assert wrapped == {"layer2.0.conv1", "layer2.0.downsample.0",
                       "layer3.0.conv1", "layer3.0.downsample.0",
                       "layer4.0.conv1", "layer4.0.downsample.0"}
    assert isinstance(m.conv1, torch.nn.Conv2d)          # stem: in_ch 3 < 16
    torch.testing.assert_close(m.layer2[0].conv1.blur_filter[0, 0] * 16,
                               torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]))
    m2 = task.make_model(Resnet18Recipe(use_blurpool=False), 10, seed=0, device="cpu")
    assert not any(isinstance(mod, task.BlurPoolConv2d) for mod in m2.modules())


def test_bn_split_replicates_the_downsample_quirk():
    """Weight decay splits on the name substring 'bn', so torchvision's downsample BNs
    (``downsample.1``) are decayed: a quirk replicated on purpose."""
    m = task.make_model(Resnet18Recipe(), num_classes=10, seed=0, device="cpu")
    groups = task._bn_split_param_groups(m, 4e-5)
    id2name = {id(p): n for n, p in m.named_parameters()}
    g0 = {id2name[id(p)] for p in groups[0]["params"]}
    g1 = {id2name[id(p)] for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.0 and groups[1]["weight_decay"] == 4e-5
    assert g0 and all("bn" in n for n in g0)
    assert any("downsample.1" in n for n in g1)          # BatchNorms that do decay
    assert g0 | g1 == set(id2name.values()) and not g0 & g1


def test_train_resolution_golden_schedules():
    r = Resnet18Recipe()                                 # the paper defaults, 15 epochs
    assert [task._train_resolution(e, r) for e in range(15)] == [160] * 12 + [192] * 3
    # a ramp with interior epochs, against hand-computed /32 half-to-even values:
    # interp(3)=170.7->160, interp(4)=181.3->192, 5:192, 6:202.7->192, 7:213.3->224
    r2 = Resnet18Recipe(min_res=160, max_res=224, start_ramp=2, end_ramp=8)
    assert [task._train_resolution(e, r2) for e in range(10)] == \
        [160, 160, 160, 160, 192, 192, 192, 224, 224, 224]


# -- the reproducibility contract + runner integration ---------------------- #
def _fake_data(n=10, q=4, n_classes=2, size=(48, 36)):
    """In-memory imagenet data dict: a real JPEG blob (encoded noise) plus the loader-contract
    keys, with no cache files or pins, so the task is tested apart from the data layer."""
    from torchvision.io import encode_jpeg
    g = torch.Generator().manual_seed(0)
    blobs, offsets = [], [0]
    for _ in range(n):
        img = torch.randint(0, 256, (3, *size), generator=g, dtype=torch.uint8)
        raw = bytes(encode_jpeg(img, quality=90).numpy())
        blobs.append(raw)
        offsets.append(offsets[-1] + len(raw))
    y = np.arange(n, dtype=np.int64) % n_classes
    return {
        "jpegs": np.frombuffer(b"".join(blobs), dtype=np.uint8),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "train_rows": np.arange(n, dtype=np.int64),
        "y_train": torch.as_tensor(y, dtype=torch.long),
        "X_query": torch.randn(q, 3, 32, 32, generator=g),
        "y_query": torch.randint(0, n_classes, (q,), generator=g),
        "meta": {"n_train": n, "n_targets": q, "n_classes": n_classes,
                 "dataset": "imagenet-synthetic"},
    }


# peak_epoch < epochs (validated); a tame peak_lr, since the equivalence test covers the
# schedule and a smoke must not risk divergence at lr 5.2
_SMOKE_RECIPE = Resnet18Recipe(epochs=2, peak_epoch=1, peak_lr=0.05, batch=4,
                               min_res=32, max_res=32, start_ramp=0, end_ramp=0,
                               val_res=32)


def test_same_seed_twice_is_bitwise_identical_on_cpu():
    data = _fake_data()
    sub = np.arange(6)
    a = task.margins(task.train_imagenet_model(data, sub, seed=123,
                                               recipe=_SMOKE_RECIPE, device="cpu"),
                     data, "cpu")
    b = task.margins(task.train_imagenet_model(data, sub, seed=123,
                                               recipe=_SMOKE_RECIPE, device="cpu"),
                     data, "cpu")
    assert a.dtype == np.float64 and a.shape == (4,)
    np.testing.assert_array_equal(a, b)                  # bitwise
    c = task.margins(task.train_imagenet_model(data, sub, seed=124,
                                               recipe=_SMOKE_RECIPE, device="cpu"),
                     data, "cpu")
    assert not np.array_equal(a, c)                      # real seed noise exists


def test_build_pool_end_to_end_on_cpu(tmp_path):
    """The real imagenet task through the runner: 2 masks x 1 seed, CPU.
    n=12 -> alpha-0.5 subsets of 6 -> batches of 4+2 (no size-1 final batch)."""
    data = _fake_data(n=12)
    t = PoolTask(recipe=_SMOKE_RECIPE,
                 train_cell=lambda d, sub, seed: task.train_imagenet_model(
                     d, sub, seed, _SMOKE_RECIPE, "cpu"),
                 readout=lambda m, d: task.margins(m, d, "cpu"))
    path = build_pool(str(tmp_path / "pool"), t, data, M=2, R=1, verbose=False)
    z = np.load(path)
    assert z["margins"].shape == (1, 2, 4) and np.isfinite(z["margins"]).all()
    assert int(z["done_rows"]) == 1
