"""The L=1 gate: our hand-rolled EK-FAC scoring must reproduce kronfluence's exactly.  SOURCE is
influence functions with ``1/(sigma+lambda)`` replaced by Bae et al.'s ``F_backward``/``F_segment``
and a walk over L segments, so matching kronfluence's pairwise scores at L=1 proves every
convention in :mod:`corrected_lds.tda._kronfluence`.  Two models cover kronfluence's two
activation conventions, ``bias=False`` (``resnet9``) and ``bias=True`` (``biased_mlp``, note 3's
ones-column path, as in the MNIST MLP).  Needs kronfluence; ~60 s on CPU, also passes on CUDA.
float64 on purpose: in float32, cuDNN's per-batch-size convolution algorithms differ by ~1e-5
relative, which would make a machine-precision assertion flaky.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kronfluence")

import torch.nn.functional as F
from torch import nn

from corrected_lds.pools.tasks.cifar_resnet9 import make_model
from corrected_lds.pools.train import per_query_margin
from corrected_lds.tda._kronfluence import (
    eigen_transform,
    fit_covariances,
    fit_lambda,
    load_factors,
    make_analyzer,
    per_sample_grads,
    perform_eigendecomposition,
    sum_factor_dirs,
    tracked_modules,
)

N_TRAIN, N_QUERY, N_CLS, WIDTH = 37, 11, 10, 0.125   # unequal: a transpose cannot hide
MLP_IN, MLP_H1, MLP_H2 = (1, 4, 6), 16, 12          # 24 inputs -> 25 with the ones column


def _build_resnet9(device):
    return make_model(num_classes=N_CLS, in_ch=3, width=WIDTH, seed=0,
                      device=device).double().eval()


def _build_biased_mlp(device):
    """The MNIST MLP's shape in miniature (Flatten -> biased Linears -> classifier), with widths
    under N_TRAIN=37 so activation covariances reach full rank: no nullspaces get compared."""
    d_in = MLP_IN[0] * MLP_IN[1] * MLP_IN[2]
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(d_in, MLP_H1, bias=True),
        nn.ReLU(),
        nn.Linear(MLP_H1, MLP_H2, bias=True),
        nn.ReLU(),
        nn.Linear(MLP_H2, N_CLS, bias=True),
    ).to(device=device).double().eval()


# (builder, input shape): a conv stack that never widens I, a Linear stack that always does
CASES = {"resnet9": (_build_resnet9, (3, 32, 32)),
         "biased_mlp": (_build_biased_mlp, MLP_IN)}


def _train_loss(logits, y):
    return F.cross_entropy(logits, y, reduction="sum")


def _measurement(logits, y):
    return -per_query_margin(logits, y).sum()


@pytest.mark.parametrize("case", sorted(CASES))
def test_l1_gate_reproduces_kronfluence_ekfac(case, tmp_path):
    from kronfluence import Analyzer, FactorArguments, ScoreArguments, prepare_model
    from kronfluence.factor.config import HEURISTIC_DAMPING_SCALE
    from kronfluence.task import Task

    build, shape = CASES[case]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    sd = {k: v.clone() for k, v in build(device).state_dict().items()}
    X_tr = torch.randn(N_TRAIN, *shape, device=device, dtype=torch.float64)
    y_tr = torch.randint(0, N_CLS, (N_TRAIN,), device=device)
    X_q = torch.randn(N_QUERY, *shape, device=device, dtype=torch.float64)
    y_q = torch.randint(0, N_CLS, (N_QUERY,), device=device)

    class MarginTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            x, y = batch
            return _train_loss(model(x), y)

        def compute_measurement(self, batch, model):
            x, y = batch
            return _measurement(model(x), y)

        def get_influence_tracked_modules(self):
            return None

    task = MarginTask()
    base = build(device)
    base.load_state_dict(sd)                 # before prepare_model (ekfac.py note 4)
    an = Analyzer("l1gate", model=prepare_model(base, task), task=task,
                  cpu=(device == "cpu"), output_dir=str(tmp_path),
                  disable_tqdm=True, log_level=40, disable_model_save=True)
    tr_ds = torch.utils.data.TensorDataset(X_tr.cpu(), y_tr.cpu())
    q_ds = torch.utils.data.TensorDataset(X_q.cpu(), y_q.cpu())
    f64 = {"per_sample_gradient_dtype": torch.float64, "lambda_dtype": torch.float64}
    an.fit_all_factors("f", dataset=tr_ds, per_device_batch_size=N_TRAIN,
                       factor_args=FactorArguments(
                           strategy="ekfac", use_empirical_fisher=True,
                           activation_covariance_dtype=torch.float64,
                           gradient_covariance_dtype=torch.float64,
                           eigendecomposition_dtype=torch.float64, **f64),
                       overwrite_output_dir=True)
    an.compute_pairwise_scores("s", factors_name="f", query_dataset=q_ds,
                               train_dataset=tr_ds,
                               per_device_query_batch_size=N_QUERY,
                               per_device_train_batch_size=N_TRAIN,
                               score_args=ScoreArguments(
                                   damping_factor=None, precondition_dtype=torch.float64,
                                   score_dtype=torch.float64,
                                   compute_per_module_scores=True,
                                   per_sample_gradient_dtype=torch.float64),
                               overwrite_output_dir=True)
    kf = an.load_pairwise_scores("s")                     # {module: (Q, n_train)}

    # ---- our path: the same numbers, from the factors alone ----
    ours = build(device)
    ours.load_state_dict(sd)
    mods = tracked_modules(ours)
    factors = load_factors(an.factors_output_dir("f"), device=device)
    assert sorted(mods) == sorted(factors), (sorted(mods), sorted(factors))

    g_tr = per_sample_grads(ours, mods, X_tr, y_tr, _train_loss)
    g_q = per_sample_grads(ours, mods, X_q, y_q, _measurement)

    total_kf = torch.zeros(N_QUERY, N_TRAIN, dtype=torch.float64)
    total_ours = torch.zeros(N_QUERY, N_TRAIN, dtype=torch.float64)
    for name, (q_a, q_g, sigma) in factors.items():
        precond = 1.0 / (sigma + HEURISTIC_DAMPING_SCALE * sigma.mean())
        pg = eigen_transform(g_q[name], q_a, q_g, precond)
        mine = torch.einsum("qoi,boi->qb", pg, g_tr[name]).cpu()
        theirs = kf[name].double().cpu()
        rel = (theirs - mine).abs().max() / theirs.abs().max()
        assert rel < 1e-10, f"module {name} disagrees: relative {rel:.3e}"
        total_kf += theirs
        total_ours += mine

    rel = ((total_kf - total_ours).abs().max() / total_kf.abs().max()).item()
    assert rel < 1e-10, f"total scores disagree: relative {rel:.3e}"
    assert np.corrcoef(total_kf.flatten(), total_ours.flatten())[0, 1] > 1 - 1e-12


def test_eval_mode_is_required():
    """Note 6: per-sample gradients are ill-defined while BatchNorm uses batch stats."""
    m = nn.Sequential(nn.Conv2d(3, 2, 3, bias=False), nn.BatchNorm2d(2)).double().train()
    with pytest.raises(RuntimeError, match="eval"):
        per_sample_grads(m, tracked_modules(m), torch.randn(2, 3, 5, 5, dtype=torch.float64),
                         torch.zeros(2, dtype=torch.long), lambda o, y: o.sum())


# -- the staged factor path (SOURCE 3.4's segment-averaged Hessian) --------- #
def _factor_args():
    from kronfluence import FactorArguments
    return FactorArguments(strategy="ekfac", use_empirical_fisher=True,
                           activation_covariance_dtype=torch.float64,
                           gradient_covariance_dtype=torch.float64,
                           eigendecomposition_dtype=torch.float64,
                           per_sample_gradient_dtype=torch.float64,
                           lambda_dtype=torch.float64)


class _MarginTask:
    """Built lazily so the module imports without kronfluence."""

    def __new__(cls):
        from kronfluence.task import Task

        class T(Task):
            def compute_train_loss(self, batch, model, sample=False):
                x, y = batch
                return _train_loss(model(x), y)

            def compute_measurement(self, batch, model):
                x, y = batch
                return _measurement(model(x), y)

            def get_influence_tracked_modules(self):
                return None
        return T()


def _fixture(tmp_path, device):
    torch.manual_seed(0)
    sd = {k: v.clone() for k, v in _build_resnet9(device).state_dict().items()}
    X = torch.randn(N_TRAIN, 3, 32, 32, device=device, dtype=torch.float64)
    y = torch.randint(0, N_CLS, (N_TRAIN,), device=device)
    return sd, torch.utils.data.TensorDataset(X.cpu(), y.cpu())


def _precond_action(factors, g, damping_scale):
    """Basis-invariant fingerprint of a factor set: what it does to a gradient.
    Comparing Q_A/Q_G directly would trip over eigenvector sign conventions."""
    out = {}
    for name, (q_a, q_g, sigma) in factors.items():
        precond = 1.0 / (sigma + damping_scale * sigma.mean())
        out[name] = eigen_transform(g[name], q_a, q_g, precond)
    return out


def test_staged_factors_reproduce_fit_all_factors(tmp_path):
    """One checkpoint per segment must give exactly what the bundled call gives: SOURCE runs
    fit_all_factors' covariance -> eigendecomposition -> lambda stages apart to sum covariances
    over a segment first, which is only sound if C_seg = 1 collapses back to the bundled path."""
    from kronfluence.factor.config import HEURISTIC_DAMPING_SCALE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd, ds = _fixture(tmp_path, device)
    fa = _factor_args()

    m = _build_resnet9(device); m.load_state_dict(sd)
    bundled = make_analyzer(m, _MarginTask(), tmp_path / "a", name="bundled", device=device)
    bundled.fit_all_factors("f", dataset=ds, per_device_batch_size=N_TRAIN,
                            factor_args=fa, overwrite_output_dir=True)

    m2 = _build_resnet9(device); m2.load_state_dict(sd)
    st = make_analyzer(m2, _MarginTask(), tmp_path / "b", name="staged", device=device)
    fit_covariances(st, "ck0", ds, factor_args=fa, batch=N_TRAIN)
    sum_factor_dirs([st.factors_output_dir("ck0")], st.factors_output_dir("seg0"),
                    kind="covariance")
    perform_eigendecomposition(st, "seg0", factor_args=fa)
    fit_lambda(st, "lam0", ds, from_factors="seg0", factor_args=fa, batch=N_TRAIN)
    sum_factor_dirs([st.factors_output_dir("lam0")], st.factors_output_dir("seg0"),
                    kind="lambda")

    f_bundled = load_factors(bundled.factors_output_dir("f"), device=device)
    f_staged = load_factors(st.factors_output_dir("seg0"), device=device)
    assert sorted(f_bundled) == sorted(f_staged)

    ours = _build_resnet9(device); ours.load_state_dict(sd)
    mods = tracked_modules(ours)
    X, y = ds.tensors
    g = per_sample_grads(ours, mods, X.to(device), y.to(device), _train_loss)
    a = _precond_action(f_bundled, g, HEURISTIC_DAMPING_SCALE)
    b = _precond_action(f_staged, g, HEURISTIC_DAMPING_SCALE)
    for name in a:
        rel = (a[name] - b[name]).abs().max() / a[name].abs().max()
        assert rel < 1e-10, f"module {name}: staged != bundled, relative {rel:.3e}"


def test_summing_a_checkpoint_with_itself_leaves_sigma_unchanged(tmp_path):
    """The invariant that makes sum-over-sum an average: sigma = lambda_matrix /
    num_lambda_processed and sum_factor_dirs adds both, so summing a segment's raw accumulators
    gives the mean per-checkpoint Hessian SOURCE 3.4 asks for, not their sum."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd, ds = _fixture(tmp_path, device)
    fa = _factor_args()
    m = _build_resnet9(device); m.load_state_dict(sd)
    an = make_analyzer(m, _MarginTask(), tmp_path / "c", name="dup", device=device)

    fit_covariances(an, "ck0", ds, factor_args=fa, batch=N_TRAIN)
    one = [an.factors_output_dir("ck0")]
    sum_factor_dirs(one, an.factors_output_dir("s1"), kind="covariance")
    sum_factor_dirs(one * 2, an.factors_output_dir("s2"), kind="covariance")
    for nm in ("s1", "s2"):
        perform_eigendecomposition(an, nm, factor_args=fa)
        fit_lambda(an, f"l_{nm}", ds, from_factors=nm, factor_args=fa, batch=N_TRAIN)
    sum_factor_dirs([an.factors_output_dir("l_s1")], an.factors_output_dir("s1"),
                    kind="lambda")
    sum_factor_dirs([an.factors_output_dir("l_s2")] * 2, an.factors_output_dir("s2"),
                    kind="lambda")

    f1 = load_factors(an.factors_output_dir("s1"), device=device)
    f2 = load_factors(an.factors_output_dir("s2"), device=device)
    for name in f1:
        s1, s2 = f1[name][2], f2[name][2]
        rel = (s1 - s2).abs().max() / s1.abs().max()
        assert rel < 1e-10, f"module {name}: sigma moved under duplication, {rel:.3e}"
