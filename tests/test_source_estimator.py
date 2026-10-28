"""End-to-end SOURCE estimator tests.  The analytic gate matters most: at ``lr_k * sigma << 1``,
``f_backward -> 1`` and ``f_segment -> lr_k``, so SOURCE collapses to a gradient dot product
scaled by ``sum_l lr_k[l] / N``, computed without the walk, eigenbasis or factors; agreement pins
orientation, sign, Eq. 20's 1/N, the per-segment mean and sum, and module coverage.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kronfluence")

import torch.nn.functional as F

from corrected_lds.pools.tasks.cifar_resnet9 import make_model
from corrected_lds.pools.train import per_query_margin
from corrected_lds.tda._kronfluence import per_sample_grads, tracked_modules
from corrected_lds.tda.source import SourceArgs, compute_run_tauq

N_TRAIN, N_QUERY, N_CLS, WIDTH = 13, 5, 10, 0.125     # unequal: a transpose cannot hide


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ctor(device):
    return lambda: make_model(num_classes=N_CLS, in_ch=3, width=WIDTH, seed=0,
                              device=device).double().eval()


def _fixture(tmp_path, n_ckpts, *, identical=True):
    """A trajectory of checkpoints plus the data dict the builders take."""
    device = _device()
    torch.manual_seed(0)
    ctor = _ctor(device)
    base = ctor()
    paths = []
    for c in range(n_ckpts):
        sd = {k: v.clone() for k, v in base.state_dict().items()}
        if not identical:                      # a mild perturbation per checkpoint
            for k, v in sd.items():
                if v.is_floating_point():
                    sd[k] = v + 0.01 * c * torch.randn_like(v)
        p = str(tmp_path / f"ep_{c:03d}.pt")
        torch.save(sd, p)
        paths.append(p)
    data = {
        "X_train": torch.randn(N_TRAIN, 3, 32, 32, dtype=torch.float64),
        "y_train": torch.randint(0, N_CLS, (N_TRAIN,)),
        "X_query": torch.randn(N_QUERY, 3, 32, 32, dtype=torch.float64),
        "y_query": torch.randint(0, N_CLS, (N_QUERY,)),
    }
    return ctor, paths, data, device


def _raw_dot(ctor, ckpt, data, device):
    """(n_train, n_targets) of <measurement grad at q, train-loss grad at b> -- the whole
    estimator with every SOURCE-specific operator removed."""
    m = ctor()
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    mods = tracked_modules(m)
    g_tr = per_sample_grads(m, mods, data["X_train"].to(device),
                            data["y_train"].to(device),
                            lambda lo, y: F.cross_entropy(lo, y, reduction="sum"))
    g_q = per_sample_grads(m, mods, data["X_query"].to(device),
                           data["y_query"].to(device),
                           lambda lo, y: -per_query_margin(lo, y).sum())
    out = None
    for n in mods:
        part = torch.einsum("qoi,boi->bq", g_q[n], g_tr[n])
        out = part if out is None else out + part
    return out.double().cpu().numpy()


# -- it runs, and produces the contract ------------------------------------- #
def test_tauq_shape_dtype_and_finiteness(tmp_path):
    ctor, paths, data, device = _fixture(tmp_path, 2, identical=False)
    args = SourceArgs(segments=1, use_empirical_fisher=True, grad_batch=8, query_chunk=3)
    tauq = compute_run_tauq(paths, data, ctor, args=args, lr_k=np.array([1.0]),
                            device=device, work_dir=str(tmp_path / "w"))
    assert tauq.shape == (N_TRAIN, N_QUERY)          # not the transpose
    assert tauq.dtype == np.float64
    assert np.isfinite(tauq).all()
    assert np.abs(tauq).max() > 0


def test_query_chunking_does_not_change_the_answer(tmp_path):
    """query_chunk is a memory lever only: the walk is redone per chunk, so a bug there
    would show up as chunk-dependent scores."""
    ctor, paths, data, device = _fixture(tmp_path, 2, identical=False)
    kw = {"use_empirical_fisher": True, "grad_batch": 16}
    a = compute_run_tauq(paths, data, ctor,
                         args=SourceArgs(segments=1, query_chunk=N_QUERY, **kw),
                         lr_k=np.array([1.0]), device=device,
                         work_dir=str(tmp_path / "w1"))
    b = compute_run_tauq(paths, data, ctor,
                         args=SourceArgs(segments=1, query_chunk=2, **kw),
                         lr_k=np.array([1.0]), device=device,
                         work_dir=str(tmp_path / "w2"))
    assert np.allclose(a, b, rtol=1e-10, atol=0)


def test_grad_batching_does_not_change_the_answer(tmp_path):
    ctor, paths, data, device = _fixture(tmp_path, 2, identical=False)
    kw = {"use_empirical_fisher": True, "query_chunk": N_QUERY}
    a = compute_run_tauq(paths, data, ctor,
                         args=SourceArgs(segments=1, grad_batch=N_TRAIN, **kw),
                         lr_k=np.array([1.0]), device=device,
                         work_dir=str(tmp_path / "w1"))
    b = compute_run_tauq(paths, data, ctor,
                         args=SourceArgs(segments=1, grad_batch=4, **kw),
                         lr_k=np.array([1.0]), device=device,
                         work_dir=str(tmp_path / "w2"))
    assert np.allclose(a, b, rtol=1e-10, atol=0)


# -- the analytic gate ------------------------------------------------------ #
@pytest.mark.parametrize("segments,n_ckpts", [(1, 2), (2, 4)])
def test_analytic_gate_collapses_to_scaled_gradient_dot_products(tmp_path, segments,
                                                                 n_ckpts):
    """At lr_k*sigma << 1, SOURCE = (sum_l lr_k[l] / N) * <grad_q, grad_b>; identical checkpoints
    make the segment mean exact, and distinct per-segment lr_k catch a shared or misrouted one."""
    ctor, paths, data, device = _fixture(tmp_path, n_ckpts, identical=True)
    lr_k = np.array([1e-8 * (l + 1) for l in range(segments)])   # deep in the small limit
    args = SourceArgs(segments=segments, use_empirical_fisher=True, grad_batch=8,
                      query_chunk=N_QUERY)
    tauq = compute_run_tauq(paths, data, ctor, args=args, lr_k=lr_k, device=device,
                            work_dir=str(tmp_path / "w"))
    want = _raw_dot(ctor, paths[0], data, device) * (lr_k.sum() / N_TRAIN)
    rel = np.abs(tauq - want).max() / np.abs(want).max()
    assert rel < 1e-6, f"relative {rel:.3e}"
    assert np.corrcoef(tauq.ravel(), want.ravel())[0, 1] > 1 - 1e-12


def test_sign_is_not_flipped(tmp_path):
    """A positive tauq must mean proponent: with m = -margin and F_r >= 0 the negations cancel,
    so tauq correlates positively with the raw dot product (ekfac.py note 2, checked here)."""
    ctor, paths, data, device = _fixture(tmp_path, 2, identical=True)
    args = SourceArgs(segments=1, use_empirical_fisher=True, grad_batch=8,
                      query_chunk=N_QUERY)
    tauq = compute_run_tauq(paths, data, ctor, args=args, lr_k=np.array([1e-8]),
                            device=device, work_dir=str(tmp_path / "w"))
    raw = _raw_dot(ctor, paths[0], data, device)
    assert np.corrcoef(tauq.ravel(), raw.ravel())[0, 1] > 0.99
