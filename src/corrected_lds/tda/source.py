"""SOURCE (Bae et al. 2024, arXiv:2405.12186) as the fourth TDA method.

Influence functions with the damped inverse replaced by two matrix functions of the segment
Hessian, walked over ``L`` training segments.  For segment ell with ``K`` iterations at mean
learning rate ``eta``, writing ``lr_k = eta*K``:

    S_bar = exp(-lr_k * H)                      the segment Jacobian      (Eq. 16)
    r_bar = (1/N) * F_r(H) * g,   F_r(s) = (1 - exp(-lr_k*s)) / s         (Eq. 20-21)
    tau(q, m) = grad_f(q)^T  sum_ell ( prod_{l'=L..ell+1} S_bar_l' )  r_bar_ell   (Eq. 24)

Each operator is :func:`~corrected_lds.tda._kronfluence.eigen_transform` with its own scale
on the EK-FAC eigenvalues; kronfluence only fits the factors (its scoring cannot express the
walk).  L=1 is influence functions (paper Fig. 3), matched to kronfluence's EK-FAC scores to
1.5e-15 by ``tests/test_source_l1_gate.py``.  Runs are combined by a plain mean, the paper's
multiple-training-run estimator (§3.3, App. C), so the 1/E member-noise law holds exactly.

Also as in the paper: §3.4's segment averages of the learning rate, gradient and Hessian;
App. D.2's heavy-ball ``1/(1-momentum)``; the true Fisher (``use_empirical_fisher=False``),
App. D's Gauss-Newton Hessian for cross-entropy; and tracking only Linear/Conv2d (App. C
excludes normalization and embedding layers).  Not implemented: Fast-SOURCE (App. E.2) and
the Adam preconditioner (we train with SGD).
"""

import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
import torch

from corrected_lds.tda.ekfac import KRONFLUENCE_VERSION

# shared with TracIn: one hash of the ordered checkpoint list (weights pair by position)
from corrected_lds.tda.tracin import (
    _files_md5,
    epoch_mean_lrs,
    resolve_ckpt_epochs,
)

__all__ = [
    "SourceArgs",
    "checkpoint_segments",
    "combine_source_partials",
    "compute_run_tauq",
    "compute_source_partials",
    "compute_tauq_source",
    "f_backward",
    "f_segment",
    "run_partial_path",
    "segment_epochs",
    "segment_factors",
    "segment_lr_times_steps",
]


@dataclass(frozen=True)
class SourceArgs:
    """Everything that changes ``tauq`` on the SOURCE side; ``to_dict()`` is the provenance
    payload, so it must stay JSON-native and round-trip stable.  ``segments`` also sets the
    scale of ``tauq`` (segment terms are summed while each ``lr_k`` shrinks as L grows), so
    pre-register it like ``ekfac.py``'s damping: choosing it by the LDS is fitting the metric.
    """

    segments: int = 3
    ckpt_every: int = 4                      # thins the shared tracin_ckpts/ trajectory
    # () -> the ckpt_every grid; a tuple pins the scored epochs (tracin.resolve_ckpt_epochs)
    checkpoint_epochs: tuple = ()
    momentum: float | None = None            # None -> recipe.momentum; never left implicit
    use_empirical_fisher: bool = False       # False -> sampled-label (true) Fisher = GNH
    factor_seed: int = 0
    label_smoothing: float = 0.0
    tracked: tuple[str, ...] | None = None   # None -> all supported Linear/Conv2d
    grad_dtype: str = "float64"              # gradients + walk; see torch_grad_dtype
    # --- pure chunking, not provenance identity ---
    factor_batch: int = 64
    grad_batch: int = 64
    query_chunk: int = 500

    def __post_init__(self):
        if int(self.segments) < 1:
            raise ValueError(f"segments must be >= 1, got {self.segments}")
        if int(self.ckpt_every) < 1:
            raise ValueError(f"ckpt_every must be >= 1, got {self.ckpt_every}")
        if self.momentum is not None and not 0.0 <= float(self.momentum) < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if str(self.grad_dtype) not in ("float64", "float32", "bfloat16", "float16"):
            raise ValueError(f"grad_dtype must be float64|float32|bfloat16|float16, "
                             f"got {self.grad_dtype!r}")
        # lists (presets, JSON) -> tuple, so the dataclass stays hashable and round-trip equal
        object.__setattr__(self, "checkpoint_epochs",
                           tuple(int(e) for e in (self.checkpoint_epochs or ())))

    def to_dict(self) -> dict:
        """JSON-native provenance payload, compared after a json round trip, so ``tracked`` is
        a list.  Batch sizes and ``query_chunk`` are pure chunking and excluded.
        ``checkpoint_epochs`` appears only when pinned (as in :meth:`TracinArgs.to_dict`): the
        dict must match released tauqs verbatim, or the CLI refuses to reuse them."""
        d = {
            "segments": int(self.segments),
            "ckpt_every": int(self.ckpt_every),
            "momentum": None if self.momentum is None else float(self.momentum),
            "use_empirical_fisher": bool(self.use_empirical_fisher),
            "factor_seed": int(self.factor_seed),
            "label_smoothing": float(self.label_smoothing),
            "tracked": None if self.tracked is None else [str(t) for t in self.tracked],
            "grad_dtype": str(self.grad_dtype),
        }
        if self.checkpoint_epochs:
            d["checkpoint_epochs"] = [int(e) for e in self.checkpoint_epochs]
        return d

    @property
    def torch_grad_dtype(self):
        """The dtype the per-sample gradients and the walk are carried in.

        Provenance identity (it changes tauq's numerics) and the dominant memory term: two
        ``(query_chunk, O, I)`` query-gradient copies, 9.09 GB each at chunk 500 on ResNet-9
        in float64, so halving its width doubles ``query_chunk`` and halves the train passes.
        float64 is the default because the gates assert identity with kronfluence; lower
        precision must be measured.  The eigenvalue maps always run in float32 or better,
        since bfloat16 would discard ``f_segment``'s expm1 accuracy near zero.
        """
        return {"float64": torch.float64, "float32": torch.float32,
                "bfloat16": torch.bfloat16, "float16": torch.float16}[str(self.grad_dtype)]


# -- segments: how the shared trajectory is partitioned --------------------- #
def checkpoint_segments(n_ckpts, segments) -> tuple:
    """Segment index of each checkpoint — consecutive equal blocks, ``l = c // (C // L)``.

    Refuses ``C % L != 0``: the walk applies one factor per block (epoch spans may differ).
    """
    c, ell = int(n_ckpts), int(segments)
    if c < 1 or ell < 1 or c % ell:
        raise ValueError(f"[source] {c} checkpoints do not divide into {ell} segments; "
                         f"pick a ckpt_every/segments pair that divides evenly.")
    per = c // ell
    return tuple(i // per for i in range(c))


def segment_epochs(epochs, ckpt_every, segments, checkpoint_epochs=()) -> tuple:
    """The 1-based inclusive ``(first, last)`` epoch range of each segment.

    Read off the checkpoints: block ``l`` covers checkpoints ``[l*C/L, (l+1)*C/L)``, so its
    epochs run from just after block ``l-1``'s last checkpoint through block ``l``'s.  On the
    uniform grid this is ``(l*E/L, (l+1)*E/L]`` (cifar10's released ``lr_k`` is pinned by
    ``test_released_cifar10_lr_k_is_unchanged``).  A pinned ``checkpoint_epochs`` can make
    spans ragged: App. B.1's six checkpoints over 20 epochs give 6, 7 and 7 epochs, where a
    uniform ``E // L`` would cover only 18 of MNIST's 20 epochs and silently drop the rest.
    """
    cps = resolve_ckpt_epochs(epochs, ckpt_every, checkpoint_epochs)
    checkpoint_segments(len(cps), segments)         # refuses C % L != 0
    ell = int(segments)
    per = len(cps) // ell
    ends = [0] + [int(cps[(l + 1) * per - 1]) for l in range(ell)]
    return tuple((ends[l] + 1, ends[l + 1]) for l in range(ell))


def segment_lr_times_steps(recipe, n_train, args: SourceArgs) -> np.ndarray:
    """``lr_k[l] = eta_bar_l * K_l / (1 - momentum)``, shape ``(L,)`` float64.

    * ``eta_bar_l``: the paper's segment-averaged learning rate (§3.4), read off the recipe's
      schedule via :func:`corrected_lds.tda.tracin.epoch_mean_lrs` so SOURCE and TracIn agree;
      epochs have equal step counts, so the mean of epoch means is the mean over steps.
    * ``K_l = (epochs in segment l) * ceil(n_train / batch)``, per segment (spans can be ragged).
    * ``1/(1 - momentum)``: the heavy-ball correction (App. D.2), a factor of 10 at the
      recipe's SGD momentum 0.9, which is why ``momentum`` is resolved explicitly here.
    """
    spans = segment_epochs(recipe.epochs, args.ckpt_every, args.segments,
                           args.checkpoint_epochs)
    momentum = float(recipe.momentum if args.momentum is None else args.momentum)
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"[source] momentum must be in [0, 1), got {momentum}")

    steps_per_epoch = max(1, int(np.ceil(int(n_train) / recipe.batch)))
    per_epoch = epoch_mean_lrs(recipe, n_train)          # (E,) float64
    # spans are 1-based inclusive; per_epoch is 0-based, so (lo, hi) -> [lo-1 : hi]
    eta = np.array([per_epoch[lo - 1:hi].mean() for lo, hi in spans], dtype=np.float64)
    k_l = np.array([(hi - lo + 1) * steps_per_epoch for lo, hi in spans], dtype=np.float64)
    return eta * k_l / (1.0 - momentum)


def f_backward(sigma, lr_k):
    """``exp(-lr_k * sigma)`` — the segment Jacobian ``S_bar`` on the eigenvalues (Eq. 16).

    In ``(0, 1]`` for ``sigma >= 0``: near 1 at vanishing curvature (remembered), near 0 at
    high curvature (forgotten).  In float64 it underflows to exactly 0 once ``lr_k*sigma``
    exceeds ~745 (common on real EK-FAC spectra), which is correct and never NaN/Inf, so the
    walk stays finite (pinned by ``tests/test_source.py``).
    """
    return torch.exp(-lr_k * sigma)


def f_segment(sigma, lr_k):
    """``(1 - exp(-lr_k * sigma)) / sigma`` — the response ``F_r`` (Eq. 21).

    Tends to ``1/sigma`` at high curvature and ``lr_k`` as it vanishes (paper Figure 3), like
    ``1/(sigma + lambda)`` with ``lambda = 1/lr_k`` (the damping heuristic bergson cites this
    paper for).  Computed as ``lr_k * (-expm1(-x)/x)``, ``x = lr_k*sigma``, for precision as
    ``x -> 0``; ``x == 0`` (ordinary in EK-FAC) gets the analytic limit instead of NaN.
    """
    x = lr_k * sigma
    zero = x == 0
    safe = x.masked_fill(zero, 1.0)
    ratio = -torch.expm1(-safe) / safe
    return lr_k * ratio.masked_fill(zero, 1.0)


# -- the estimator ---------------------------------------------------------- #
def _make_task(args: SourceArgs):
    """The ``kronfluence.Task`` used only to fit factors; scoring is done by hand.

    ``compute_measurement`` (the paper's readout) only satisfies the interface.  Both hooks
    reduce with sum because kronfluence sums over the batch itself.
    """
    import torch.nn.functional as F
    from kronfluence.task import Task

    from corrected_lds.pools.train import per_query_margin

    ls, tracked = float(args.label_smoothing), (
        None if args.tracked is None else [str(t) for t in args.tracked])

    class _MarginTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            x, y = batch
            logits = model(x)
            if not sample:
                return F.cross_entropy(logits, y, reduction="sum", label_smoothing=ls)
            with torch.no_grad():          # true Fisher = GNH for CE (paper App. D)
                y = torch.multinomial(torch.softmax(logits.detach(), dim=-1), 1).flatten()
            return F.cross_entropy(logits, y, reduction="sum", label_smoothing=ls)

        def compute_measurement(self, batch, model):
            x, y = batch
            return -per_query_margin(model(x), y).sum()

        def get_influence_tracked_modules(self):
            return tracked

    return _MarginTask()


def _train_loss(args: SourceArgs):
    import torch.nn.functional as F
    ls = float(args.label_smoothing)
    return lambda logits, y: F.cross_entropy(logits, y, reduction="sum",
                                             label_smoothing=ls)


def _measurement(logits, y):
    from corrected_lds.pools.train import per_query_margin
    return -per_query_margin(logits, y).sum()


def _load_into(model_ctor, ckpt, device):
    m = model_ctor()
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.eval()


def segment_factors(ckpt_paths, data, model_ctor, *, args: SourceArgs, device, work_dir):
    """``[{module: (Q_A, Q_G, sigma)}]``, one dict per segment: SOURCE §3.4's averaged Hessian.

    Covariances are summed over the segment's checkpoints before the eigendecomposition, then
    lambda is fitted per checkpoint in that shared eigenbasis and summed; with their counts
    summed too, these sums are averages (:func:`~corrected_lds.tda._kronfluence.sum_factor_dirs`).
    """
    from kronfluence import FactorArguments

    from corrected_lds.tda import _kronfluence as kf

    seg_of = checkpoint_segments(len(ckpt_paths), args.segments)
    task = _make_task(args)
    fargs = FactorArguments(strategy="ekfac",
                            use_empirical_fisher=bool(args.use_empirical_fisher))
    ds = _tensor_dataset(data["X_train"], data["y_train"])

    # stage 1: raw covariance accumulators at every checkpoint
    for c, ckpt in enumerate(ckpt_paths):
        an = kf.make_analyzer(_load_into(model_ctor, ckpt, device), task,
                              os.path.join(work_dir, f"ck{c:03d}"), device=device)
        torch.manual_seed(int(args.factor_seed) + c)      # the sampled-label Fisher draw
        kf.fit_covariances(an, "cov", ds, factor_args=fargs, batch=args.factor_batch)

    def _dir(c, name):
        return os.path.join(work_dir, f"ck{c:03d}", "src", f"factors_{name}")

    out = []
    for l in range(int(args.segments)):
        members = [c for c in range(len(ckpt_paths)) if seg_of[c] == l]
        seg_dir = os.path.join(work_dir, f"seg{l:02d}", "src", "factors_seg")
        # stage 2: sum the segment's covariances, then eigendecompose the average
        kf.sum_factor_dirs([_dir(c, "cov") for c in members], seg_dir, kind="covariance")
        an = kf.make_analyzer(_load_into(model_ctor, ckpt_paths[members[-1]], device),
                              task, os.path.join(work_dir, f"seg{l:02d}"), device=device)
        kf.perform_eigendecomposition(an, "seg", factor_args=fargs)
        # stage 3: lambda at each checkpoint, in this segment's shared eigenbasis
        lam_dirs = []
        for c in members:
            a2 = kf.make_analyzer(_load_into(model_ctor, ckpt_paths[c], device), task,
                                  os.path.join(work_dir, f"seg{l:02d}"), device=device)
            torch.manual_seed(int(args.factor_seed) + 1000 + c)
            kf.fit_lambda(a2, f"lam{c:03d}", ds, from_factors="seg", factor_args=fargs,
                          batch=args.factor_batch)
            lam_dirs.append(os.path.join(work_dir, f"seg{l:02d}", "src",
                                         f"factors_lam{c:03d}"))
        kf.sum_factor_dirs(lam_dirs, seg_dir, kind="lambda")
        out.append(kf.load_factors(seg_dir, device=device))
    return out


def _tensor_dataset(X, y):
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.as_tensor(X).detach().cpu(),
                         torch.as_tensor(y).detach().cpu().long())


def compute_run_tauq(ckpt_paths, data, model_ctor, *, args: SourceArgs, lr_k, device,
                     work_dir, verbose=False) -> np.ndarray:
    """SOURCE ``tauq (n_train, n_targets)`` float64 for one training trajectory (Eq. 24).

    ``ckpt_paths`` are its C checkpoints in training order; ``lr_k`` is the ``(L,)`` array
    from :func:`segment_lr_times_steps`.  ``S_bar`` and ``F_r`` are symmetric, so the Q
    queries are pushed through the walk instead of the N train gradients, which is what
    makes this affordable.  ``prod S_bar`` is never formed: each factor is applied in its own
    segment's eigenbasis (the bases differ).  Segments run backwards with scoring nested
    inside, so one walked query is held at a time (~9 GB vs ~27 GB at ``query_chunk=500`` on
    ResNet-9) and the cost is C gradient passes per query chunk, not C*L.
    """
    from corrected_lds.tda import _kronfluence as kf

    ell = int(args.segments)
    lr_k = np.asarray(lr_k, dtype=np.float64)
    if lr_k.shape != (ell,):
        raise ValueError(f"[source] lr_k has shape {lr_k.shape}, expected ({ell},)")
    seg_of = checkpoint_segments(len(ckpt_paths), ell)
    per_seg = len(ckpt_paths) // ell

    factors = segment_factors(ckpt_paths, data, model_ctor, args=args, device=device,
                              work_dir=work_dir)
    # release kronfluence's cached blocks: the walk is the method's largest allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_tr = torch.as_tensor(data["X_train"])
    y_tr = torch.as_tensor(data["y_train"]).long()
    X_q = torch.as_tensor(data["X_query"])
    y_q = torch.as_tensor(data["y_query"]).long()
    n_train, n_q = len(y_tr), len(y_q)
    loss = _train_loss(args)

    # theta^s: the query gradient is taken at the final checkpoint (Eq. 24)
    gdt = args.torch_grad_dtype
    # the maps run in float32 or better whatever `gdt` is (see SourceArgs.torch_grad_dtype)
    mdt = torch.float64 if gdt == torch.float64 else torch.float32
    final = _load_into(model_ctor, ckpt_paths[-1], device)
    final_mods = kf.tracked_modules(final)

    tauq = np.zeros((n_train, n_q), dtype=np.float64)
    for q0 in range(0, n_q, int(args.query_chunk)):
        q1 = min(n_q, q0 + int(args.query_chunk))
        v = kf.per_sample_grads(final, final_mods, X_q[q0:q1].to(device),
                                y_q[q0:q1].to(device), _measurement, dtype=gdt)
        for l in reversed(range(ell)):
            if l < ell - 1:                       # apply S_bar_{l+1}: v_l = S_bar_{l+1} v_{l+1}
                nxt = factors[l + 1]
                v = {n: kf.eigen_transform(
                    v[n], nxt[n][0].to(gdt), nxt[n][1].to(gdt),
                    f_backward(nxt[n][2].to(mdt), lr_k[l + 1]).to(gdt)) for n in v}
            cur = factors[l]                      # then this segment's response F_r
            w = {n: kf.eigen_transform(
                v[n], cur[n][0].to(gdt), cur[n][1].to(gdt),
                f_segment(cur[n][2].to(mdt), lr_k[l]).to(gdt)) for n in v}
            for c in [c for c in range(len(ckpt_paths)) if seg_of[c] == l]:
                model = _load_into(model_ctor, ckpt_paths[c], device)
                mods = kf.tracked_modules(model)
                for b0 in range(0, n_train, int(args.grad_batch)):
                    b1 = min(n_train, b0 + int(args.grad_batch))
                    g = kf.per_sample_grads(model, mods, X_tr[b0:b1].to(device),
                                            y_tr[b0:b1].to(device), loss, dtype=gdt)
                    acc = None
                    for n, wn in w.items():       # sum over modules = the full-parameter dot
                        part = torch.einsum("qoi,boi->bq", wn, g[n])
                        acc = part if acc is None else acc.add_(part)
                    tauq[b0:b1, q0:q1] += acc.double().cpu().numpy() / per_seg
                    del g, acc
                del model, mods
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del w
            if verbose:
                print(f"    [source] queries {q0}:{q1}  segment {l} scored "
                      f"({per_seg} ckpt, lr_k={lr_k[l]:.4g})", flush=True)
        del v
    return tauq / float(n_train)                  # the 1/N of Eq. 20


# -- per-run partials: recombinable over an arbitrary run set --------------- #
def run_partial_path(part_dir, r) -> str:
    return os.path.join(part_dir, f"tauq_source_r_{int(r):03d}.npz")


def _partial_valid(path, *, run, ckpts_md5, prov_json, args_json, lr_k) -> bool:
    """A partial is reusable only if its run id, ordered checkpoint contents, SOURCE
    arguments, realized per-segment ``lr_k``, provenance and kronfluence version all match
    (``lr_k`` comes from the recipe, so a changed schedule or momentum leaves the args equal).
    """
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001 — a truncated write is simply "not reusable"
        return False
    if "source_args" not in z.files:
        return False
    return (int(z["run"]) == int(run)
            and str(z["ckpts_md5"]) == ckpts_md5
            and str(z["provenance"]) == prov_json
            and str(z["source_args"]) == args_json
            and str(z["kronfluence_version"]) == KRONFLUENCE_VERSION
            and np.array_equal(z["lr_k"], np.asarray(lr_k)))


def combine_source_partials(part_dir, runs) -> np.ndarray:
    """Plain mean of the per-run ``tauq`` partials, the paper's multiple-training-run
    estimator (§3.3), over any run ids rather than a prefix, so disjoint replicates need no
    re-scoring.  Returns ``(n_train, Q)`` float64.
    """
    runs = [int(r) for r in runs]
    if not runs:
        raise ValueError("combine_source_partials: `runs` is empty")
    if len(set(runs)) != len(runs):
        raise ValueError(f"combine_source_partials: duplicate runs in {runs}")
    total = None
    for r in runs:
        p = run_partial_path(part_dir, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"combine_source_partials: missing partial for run "
                                    f"{r}: {p}")
        t = np.load(p, allow_pickle=False)["tauq"].astype(np.float64)
        total = t if total is None else total + t
    return total / len(runs)


# -- the public entry point ------------------------------------------------- #
def compute_source_partials(traj_ckpts, data, model_ctor, *, recipe, args=None,
                            device="cuda", work_dir=None, provenance=None, runs=None,
                            verbose=False) -> list:
    """Compute the per-run partials and return their paths; does not combine.

    ``runs`` selects which trajectories to compute (default: all).  Each run writes its own
    ``tauq_source_r_NNN.npz`` (temp-then-rename) and ``_kf_rNNN/`` factor store, so N
    processes may compute disjoint ``runs`` into the same ``work_dir`` concurrently.  Isolate
    them with ``CUDA_VISIBLE_DEVICES``, not a ``cuda:N`` device: kronfluence's ``Analyzer``
    allocates on the current cuda device, so ``device="cuda:1"`` silently lands on GPU 0.
    """
    if not traj_ckpts:
        raise ValueError("compute_source_partials: no trajectories given")
    args = args or SourceArgs()
    work_dir = work_dir or os.path.join(os.getcwd(), "source_work")
    os.makedirs(work_dir, exist_ok=True)
    lr_k = segment_lr_times_steps(recipe, int(data["meta"]["n_train"]), args)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    args_json = json.dumps(args.to_dict(), sort_keys=True)
    wanted = list(range(len(traj_ckpts))) if runs is None else sorted({int(r) for r in runs})
    for r in wanted:
        if not 0 <= r < len(traj_ckpts):
            raise ValueError(f"[source] run {r} out of range for {len(traj_ckpts)} "
                             f"trajectories")

    out = []
    for r in wanted:
        paths = traj_ckpts[r]
        path = run_partial_path(work_dir, r)
        ckpts_md5 = _files_md5(paths)
        out.append(path)
        if _partial_valid(path, run=r, ckpts_md5=ckpts_md5, prov_json=prov_json,
                          args_json=args_json, lr_k=lr_k):
            if verbose:
                print(f"    [source] run {r} partial present (identity ok) -> reuse",
                      flush=True)
            continue
        run_dir = os.path.join(work_dir, f"_kf_r{r:03d}")
        shutil.rmtree(run_dir, ignore_errors=True)     # never resume a half-written store
        try:
            tauq_r = compute_run_tauq(paths, data, model_ctor, args=args, lr_k=lr_k,
                                      device=device, work_dir=run_dir, verbose=verbose)
            tmp = path + f".tmp{os.getpid()}.npz"      # pid-tagged: concurrent-safe
            np.savez(tmp, tauq=tauq_r.astype(np.float32), run=r, ckpts_md5=ckpts_md5,
                     lr_k=lr_k, provenance=prov_json, source_args=args_json,
                     kronfluence_version=KRONFLUENCE_VERSION)
            os.replace(tmp, path)                      # atomic
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)  # bound disk to one run's factors
    return out


def compute_tauq_source(traj_ckpts, data, model_ctor, *, recipe, args=None, device="cuda",
                        work_dir=None, provenance=None, verbose=False) -> np.ndarray:
    """Run-averaged SOURCE ``tauq (n_train, n_targets)`` from trajectory checkpoints.

    ``traj_ckpts`` is one ordered checkpoint-path list per training run, thinned to
    ``args.ckpt_every``: the same full-set trajectories ``compute_tauq_tracin`` takes
    (``tracin.train_trajectories``).  Computes missing partials, then plain-means them; to
    split across GPUs, first run :func:`compute_source_partials` with disjoint ``runs``.
    """
    args = args or SourceArgs()
    work_dir = work_dir or os.path.join(os.getcwd(), "source_work")
    compute_source_partials(traj_ckpts, data, model_ctor, recipe=recipe, args=args,
                            device=device, work_dir=work_dir, provenance=provenance,
                            verbose=verbose)
    tauq = combine_source_partials(work_dir, range(len(traj_ckpts)))
    if verbose:
        lr_k = segment_lr_times_steps(recipe, int(data["meta"]["n_train"]), args)
        print(f"[source] tauq {tauq.shape} from {len(traj_ckpts)} run(s) x "
              f"{len(traj_ckpts[0])} checkpoints (plain mean, L={args.segments}, "
              f"lr_k={np.array2string(lr_k, precision=4)})", flush=True)
    return tauq
