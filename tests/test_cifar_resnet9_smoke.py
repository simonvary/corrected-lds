"""CPU micro-smoke of the image trainer: the same seed twice gives bitwise-identical margins
within one environment.  Needs torch (e.g. the [gpu] extra), so it skips in core-only CI."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from corrected_lds.banks.presets import Resnet9Recipe
from corrected_lds.pools.runner import PoolTask, build_pool
from corrected_lds.pools.tasks.cifar_resnet9 import margins, train_paper_model
from corrected_lds.pools.train import per_query_margin


def _fake_data(n=64, Q=8):
    g = torch.Generator().manual_seed(0)
    return {
        "X_train": torch.randn(n, 3, 32, 32, generator=g),
        "y_train": torch.randint(0, 10, (n,), generator=g),
        "X_query": torch.randn(Q, 3, 32, 32, generator=g),
        "y_query": torch.randint(0, 10, (Q,), generator=g),
        "meta": {"n_train": n, "n_targets": Q, "n_classes": 10, "in_ch": 3, "dataset": "cifar10"},
    }


def test_per_query_margin_formula():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    y = torch.tensor([0, 1])
    got = per_query_margin(logits, y)
    exp = torch.tensor([2.0 - float(np.logaddexp(1.0, 0.0)),
                        3.0 - float(np.logaddexp(0.0, 1.0))], dtype=got.dtype)
    torch.testing.assert_close(got, exp)


def test_same_seed_twice_is_bitwise_identical_on_cpu():
    data = _fake_data()
    recipe = Resnet9Recipe(epochs=1, batch=32)
    sub = np.arange(48)
    a = margins(train_paper_model(data, sub, seed=123, recipe=recipe, device="cpu"),
                     data, "cpu")
    b = margins(train_paper_model(data, sub, seed=123, recipe=recipe, device="cpu"),
                     data, "cpu")
    assert a.dtype == np.float64 and a.shape == (8,)
    np.testing.assert_array_equal(a, b)                   # bitwise
    # a different seed moves the readout (real seed noise exists)
    c = margins(train_paper_model(data, sub, seed=124, recipe=recipe, device="cpu"),
                     data, "cpu")
    assert not np.array_equal(a, c)


def test_build_pool_end_to_end_on_cpu(tmp_path):
    """The real image task through the runner: 2 masks x 1 seed, 1 epoch, CPU."""
    data = _fake_data()
    recipe = Resnet9Recipe(epochs=1, batch=32)
    task = PoolTask(
        recipe=recipe,
        train_cell=lambda d, sub, seed: train_paper_model(d, sub, seed, recipe, "cpu"),
        readout=lambda m, d: margins(m, d, "cpu"))
    path = build_pool(str(tmp_path / "pool"), task, data, M=2, R=1, verbose=False)
    z = np.load(path)
    assert z["margins"].shape == (1, 2, 8) and np.isfinite(z["margins"]).all()
    assert int(z["done_rows"]) == 1
