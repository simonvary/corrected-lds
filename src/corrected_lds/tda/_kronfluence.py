"""The kronfluence layer: factor fitting, per-sample gradients, eigenbasis transforms.

The only module that touches ``kronfluence`` (1.0.1, used only to fit EK-FAC factors and
read them through its public loaders), so the L=1 gate ``tests/test_source_l1_gate.py``
tests exactly this layer.  SOURCE cannot use kronfluence's scoring: ``compute_pairwise_scores``
derives query gradients from the ``Task``, but SOURCE's are the previous segment's output.

Notes: six conventions read from kronfluence 1.0.1 and verified by the L=1 gate (all 9
ResNet-9 modules agree to 1.5e-15 with a float64 model; a float32 model agrees only to ~1e-5,
as cuDNN picks different algorithms per batch size, with Spearman/Pearson 1.0000000000).

1. Per-sample gradients are ``(batch, O, I)`` for both module types,
   ``einsum("bso,bsi->boi", output_grad, activation)`` over the "sequence" axis
   (``module/linear.py:69-73``, ``module/conv2d.py:163-172``; their ``i``/``o`` are swapped).
2. Conv activations are ``F.unfold`` order (channel-major ``C_in * k1 * k2``), transposed to
   ``(b, O1*O2, I)`` (``module/conv2d.py:55-63`` ``extract_patches``), whose ``groups``
   mean-reduction is a no-op at ``groups=1``; we refuse anything else.
3. A bias appends a ones column to the activation (``linear.py:59-64``, ``conv2d.py:147-154``).
4. ``sigma = lambda_matrix / num_lambda_processed``: lambda is stored as the raw sum, divided
   and damped only in memory at score time (``factor/config.py:331-338``).
5. The transform is ``Q_G.T @ g @ Q_A`` -> elementwise scale on the (O, I) eigenvalue grid
   -> ``Q_G @ . @ Q_A.T`` (``factor/config.py:349-356``), so any matrix function of the
   Hessian is this sandwich with a different scale.
6. ``model.eval()`` is mandatory: per-sample gradients need BatchNorm's running stats
   (kronfluence forces it in ``prepare_model``, ``analyzer.py:37``; we assert it).
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "eigen_transform",
    "fit_covariances",
    "fit_lambda",
    "load_factors",
    "make_analyzer",
    "per_sample_grads",
    "perform_eigendecomposition",
    "sum_factor_dirs",
    "tracked_modules",
]

_SUPPORTED = (nn.Conv2d, nn.Linear)


def tracked_modules(model) -> dict:
    """``{name: module}`` for every module kronfluence tracks, in its naming.

    Only Linear/Conv2d, so BatchNorm affine parameters are excluded from the influence
    (``ekfac.py`` note 3), and SOURCE also unrolls only this subspace.
    """
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, _SUPPORTED):
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                raise ValueError(f"[source] module {name!r} has groups={m.groups}; "
                                 f"extract_patches averages over groups (note 2) and that "
                                 f"path is untested here.")
            out[name] = m
    if not out:
        raise ValueError("[source] no Linear/Conv2d modules found to track")
    return out


def _activation(mod, x):
    """``(b, S, I)`` exactly as kronfluence flattens it (notes 1-2)."""
    if isinstance(mod, nn.Conv2d):
        activation = F.unfold(x, mod.kernel_size, dilation=mod.dilation,
                              padding=mod.padding, stride=mod.stride).transpose(1, 2)
    else:
        activation = x.unsqueeze(1) if x.dim() == 2 else x
    if mod.bias is not None:
        activation = torch.cat((activation, torch.ones_like(activation[..., :1])), dim=-1)
    return activation


def _output_grad(mod, g):
    """``(b, S, O)`` exactly as kronfluence flattens it (note 1)."""
    if isinstance(mod, nn.Conv2d):
        return g.flatten(2).transpose(1, 2)
    return g.unsqueeze(1) if g.dim() == 2 else g


def per_sample_grads(model, mods, x, y, loss_fn, *, dtype=torch.float64) -> dict:
    """``{name: (b, O, I)}`` — the kronfluence per-sample gradient, computed by hand.

    ``loss_fn(logits, y)`` must reduce with **sum**, never mean: each example's gradient is
    read off its output gradient, so a ``1/b`` would rescale every module.
    """
    if model.training:
        raise RuntimeError("[source] model must be in eval() mode — per-sample gradients "
                           "are ill-defined while BatchNorm uses batch statistics (note 6)")
    acts, grads, handles = {}, {}, []
    try:
        for name, m in mods.items():
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: acts.__setitem__(n, inp[0].detach())))
            handles.append(m.register_full_backward_hook(
                lambda mod, gin, gout, n=name: grads.__setitem__(n, gout[0].detach())))
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
    finally:
        for h in handles:
            h.remove()
    model.zero_grad(set_to_none=True)
    return {n: torch.einsum("bso,bsi->boi",
                            _output_grad(m, grads[n]).to(dtype),
                            _activation(m, acts[n]).to(dtype))
            for n, m in mods.items()}


def load_factors(factors_dir, *, device="cpu", dtype=torch.float64) -> dict:
    """``{name: (Q_A, Q_G, sigma)}`` from a kronfluence factors directory (notes 4-5).

    ``sigma`` is the raw ``(O, I)`` EK-FAC eigenvalue grid, undamped and not inverted.
    """
    from pathlib import Path

    from kronfluence.factor.eigen import load_eigendecomposition, load_lambda_matrices
    from kronfluence.utils.constants import (
        ACTIVATION_EIGENVECTORS_NAME,
        GRADIENT_EIGENVECTORS_NAME,
        LAMBDA_MATRIX_NAME,
        NUM_LAMBDA_PROCESSED,
    )
    factors_dir = Path(factors_dir)     # kronfluence's path helpers need a Path, not str
    eig = load_eigendecomposition(output_dir=factors_dir)
    lam = load_lambda_matrices(output_dir=factors_dir)
    out = {}
    for name in lam[LAMBDA_MATRIX_NAME]:
        out[name] = (
            eig[ACTIVATION_EIGENVECTORS_NAME][name].to(device=device, dtype=dtype),
            eig[GRADIENT_EIGENVECTORS_NAME][name].to(device=device, dtype=dtype),
            (lam[LAMBDA_MATRIX_NAME][name].to(device=device, dtype=dtype)
             / lam[NUM_LAMBDA_PROCESSED][name].to(device=device, dtype=dtype)),
        )
    return out


def eigen_transform(g, q_a, q_g, scale):
    """Apply a matrix function of the EK-FAC Hessian to ``g`` (note 5).

    ``g`` is ``(..., O, I)``; ``scale`` is the function on the eigenvalue grid, broadcast over
    the leading axes: ``1/(sigma+lambda)`` for the damped inverse, ``exp(-lrK*sigma)`` for
    SOURCE's ``S_bar`` and ``(1-exp(-lrK*sigma))/sigma`` for its ``r_bar``.
    """
    return q_g @ ((q_g.transpose(-2, -1) @ g @ q_a) * scale) @ q_a.transpose(-2, -1)


# -- staged factor fitting: the paper's segment-averaged Hessian (SOURCE 3.4) --- #
# `fit_all_factors` runs all three stages on one model state; SOURCE 3.4 averages H_bar_l over
# a segment's checkpoints, so the stages run apart through kronfluence's public API.

def make_analyzer(model, task, out_dir, *, name="src", device="cpu"):
    """``prepare_model`` + ``Analyzer``.  Destructive: load the checkpoint into ``model``
    before calling, and never reuse a prepared model (``ekfac.py`` note 4)."""
    import os

    from kronfluence import Analyzer, prepare_model
    os.makedirs(out_dir, exist_ok=True)
    return Analyzer(name, model=prepare_model(model, task), task=task,
                    cpu=(torch.device(device).type == "cpu"), output_dir=str(out_dir),
                    disable_tqdm=True, log_level=40, disable_model_save=True)


def fit_covariances(analyzer, factors_name, dataset, *, factor_args, batch):
    """Stage 1: the raw K-FAC covariance accumulators at one checkpoint."""
    analyzer.fit_covariance_matrices(factors_name, dataset=dataset,
                                     per_device_batch_size=batch,
                                     factor_args=factor_args, overwrite_output_dir=True)


def perform_eigendecomposition(analyzer, factors_name, *, factor_args):
    """Stage 2: eigendecompose the covariances already in ``factors_name`` (the segment sum
    from :func:`sum_factor_dirs`, hence no ``load_from_factors_name``)."""
    analyzer.perform_eigendecomposition(factors_name, factor_args=factor_args,
                                        overwrite_output_dir=True)


def fit_lambda(analyzer, factors_name, dataset, *, from_factors, factor_args, batch):
    """Stage 3: the eigenvalue correction at one checkpoint, in ``from_factors``' basis, so
    a segment's checkpoints share one eigenbasis and their lambdas can be summed."""
    analyzer.fit_lambda_matrices(factors_name, dataset=dataset,
                                 per_device_batch_size=batch, factor_args=factor_args,
                                 load_from_factors_name=from_factors,
                                 overwrite_output_dir=True)


def sum_factor_dirs(src_dirs, dst_dir, *, kind):
    """Sum a ``"covariance"`` or ``"lambda"`` factor family (``kind``) across checkpoint dirs.

    Summing accumulators and counts is kronfluence's own partition aggregation
    (``_aggregate_factors``, ``computer/factor_computer.py:90-100``) applied across
    checkpoints.  Every consumer divides by the count, so this is SOURCE 3.4's mean of the
    per-checkpoint Hessians, even if the checkpoints saw different numbers of examples.
    """
    from kronfluence.factor.covariance import (
        COVARIANCE_FACTOR_NAMES,
        load_covariance_matrices,
        save_covariance_matrices,
    )
    from kronfluence.factor.eigen import (
        LAMBDA_FACTOR_NAMES,
        load_lambda_matrices,
        save_lambda_matrices,
    )
    load, save, names = {
        "covariance": (load_covariance_matrices, save_covariance_matrices,
                       COVARIANCE_FACTOR_NAMES),
        "lambda": (load_lambda_matrices, save_lambda_matrices, LAMBDA_FACTOR_NAMES),
    }[kind]

    # kronfluence's path helpers need a Path (a str would TypeError deep inside the library)
    from pathlib import Path
    src_dirs = [Path(d) for d in src_dirs]
    dst_dir = Path(dst_dir)
    if not src_dirs:
        raise ValueError(f"[source] sum_factor_dirs({kind}): no source directories")
    total = None
    for d in src_dirs:
        loaded = load(output_dir=d)
        if total is None:
            total = {f: {m: t.clone() for m, t in loaded[f].items()} for f in names}
            continue
        for f in names:
            if set(loaded[f]) != set(total[f]):
                raise ValueError(f"[source] {kind} factors in {d} track different modules "
                                 f"than {src_dirs[0]}; refusing to sum mismatched sets.")
            for m, t in loaded[f].items():
                total[f][m].add_(t)
    dst_dir.mkdir(parents=True, exist_ok=True)
    save(output_dir=dst_dir, factors=total)
    return dst_dir
