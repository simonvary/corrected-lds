"""TracInCP (Pruthi et al. 2020, arXiv:2002.08484, Eq. 1), random-projected and
trajectory-ensembled as the TRAK paper's baseline runs it (TRAK, arXiv:2303.14186, App. A.3).

:func:`compute_tauq_tracin` gives the ``tauq (n_train, n_targets)`` of the TRAK and EK-FAC
builders, a plain mean over runs (like EK-FAC, so the 1/E member-noise law holds exactly):

    tauq[i, q] = (1/E) * sum_r sum_t  w_t * <P_r grad_i, P_r grad_q>   at ckpt (r, t)

* Gradients: of the training loss (CE + the recipe's label_smoothing) over all parameters,
  at train and query points alike (TracIn's definition, not the pool's margin readout), in
  eval mode on the unaugmented stored example, as the official code does.  Positive tau =
  loss reduction on q = higher margin, so there is no sign flip anywhere.
* Checkpoints (TracIn Sec. 3.3): epoch ends of full-train-set runs seeded
  ``TRACIN_SEED_BASE + r``, thinned by ``ckpt_every`` (TRAK App. A.3) but always keeping
  the final epoch.  ``units_per_epoch=k`` counts 1/k-epoch units, for mid-epoch checkpoints
  from an external harness (Bae et al.'s GPT-2: six over three epochs, ``pools/bergson.py``).
* Weights (Eq. 1): ``"schedule"`` reads eta_t off the recipe's stepped LR schedule (see
  :func:`ckpt_lr_weights`); ``"equal"`` is the official-repo simplification (TracIn Sec. 4.2).
* Projections (TracIn App. E): traker's at ``seed=0``, one matrix per run (``model_id=r``)
  shared by both sides, as unbiased dot products need.  No ``proj_dim >= n/2`` guard: that
  protects TRAK's kernel inversion; TracIn dot products are unbiased at any dimension.
* Not implemented: the App. F last-layer rank-1 variant (the TRAK-baseline comparison wants
  full-model gradients).  Momentum and weight decay act via the trajectory (Remark 3.4).
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from corrected_lds.banks.presets import (
    BertRecipe,
    Gpt2WikitextBaeRecipe,
    MnistRecipe,
    Resnet9Recipe,
)
from corrected_lds.pools.train import linear_decay_lr, triangular_lr
from corrected_lds.tda.trak import TRAKER_VERSION, _ckpt_valid, _save_ckpt_atomic

__all__ = [
    "TRACIN_SEED_BASE",
    "TracinArgs",
    "ckpt_epochs",
    "ckpt_lr_weights",
    "combine_tracin_partials",
    "compute_tauq_tracin",
    "epoch_mean_lrs",
    "resolve_ckpt_epochs",
    "run_partial_path",
    "train_trajectories",
    "trajectory_paths",
]

TRACIN_SEED_BASE = 30_000     # run r -> seed base + r (TRAK uses 10k, feat-aug 20k, cells 50k+)

_LR_WEIGHTINGS = ("schedule", "equal")


@dataclass(frozen=True)
class TracinArgs:
    """The TracIn arguments that change ``tauq`` (``proj_dim`` is a tauq.npz column instead).
    ``to_dict()`` is the provenance payload: it must stay JSON-native and round-trip stable."""

    ckpt_every: int = 1               # keep every c-th epoch checkpoint (TracIn Sec. 3.3: c=1)
    lr_weighting: str = "schedule"    # Eq. 1 eta_t from the recipe's schedule | "equal" (Sec. 4.2)
    label_smoothing: float = 0.0      # the train loss whose gradients are traced
    use_half_precision: bool = False  # fp16 grads+projection (traker's default is fp16; ours off)
    # () = the ckpt_every grid; else the saved checkpoints to score (see resolve_ckpt_epochs)
    checkpoint_epochs: tuple = ()
    # k > 1 counts ckpt_every and checkpoint_epochs in 1/k-epoch units (mid-epoch checkpoints)
    units_per_epoch: int = 1
    # module names whose weight + bias are traced; () = every parameter.  GPT-2 needs it: its
    # tied wte carries ~91% of the squared gradient norm (all-parameter TracIn ~ embedding-only)
    tracked: tuple = ()
    # a fresh projection per (run, checkpoint), so the JL error averages instead of repeating
    projection_per_checkpoint: bool = False

    def __post_init__(self):
        if int(self.ckpt_every) < 1:
            raise ValueError(f"ckpt_every must be >= 1, got {self.ckpt_every}")
        if int(self.units_per_epoch) < 1:
            raise ValueError(f"units_per_epoch must be >= 1, got {self.units_per_epoch}")
        if self.lr_weighting not in _LR_WEIGHTINGS:
            raise ValueError(f"lr_weighting must be one of {_LR_WEIGHTINGS}, "
                             f"got {self.lr_weighting!r}")
        # lists from presets/JSON -> tuples: hashable, and equal across a JSON round trip
        object.__setattr__(self, "checkpoint_epochs",
                           tuple(int(e) for e in (self.checkpoint_epochs or ())))
        object.__setattr__(self, "tracked", tuple(str(m) for m in (self.tracked or ())))

    def to_dict(self) -> dict:
        """Optional keys appear only when set: the dict is compared verbatim with the string
        stored in every released tauq, and a new default key would make the CLI refuse them."""
        d = {"ckpt_every": int(self.ckpt_every), "lr_weighting": str(self.lr_weighting),
             "label_smoothing": float(self.label_smoothing),
             "use_half_precision": bool(self.use_half_precision)}
        if self.checkpoint_epochs:
            d["checkpoint_epochs"] = [int(e) for e in self.checkpoint_epochs]
        if int(self.units_per_epoch) != 1:      # absent means the epoch grid
            d["units_per_epoch"] = int(self.units_per_epoch)
        if self.tracked:                         # absent means every parameter
            d["tracked"] = list(self.tracked)
        if self.projection_per_checkpoint:       # absent means one per run
            d["projection_per_checkpoint"] = True
        return d


# -- checkpoint schedule + Eq. 1 weights from the recipe's own LR schedule --- #
def ckpt_epochs(epochs, ckpt_every) -> tuple:
    """The 1-based checkpoint epochs ``(c, 2c, ..., epochs)``.  A non-divisible ``epochs`` is
    refused: dropping the final checkpoint would delete the highest-loss-reduction segment."""
    epochs, c = int(epochs), int(ckpt_every)
    if epochs < 1 or c < 1 or epochs % c != 0:
        raise ValueError(f"[tracin] epochs={epochs} is not a positive multiple of "
                         f"ckpt_every={c}; the final epoch must be a checkpoint.")
    return tuple(range(c, epochs + 1, c))


def resolve_ckpt_epochs(epochs, ckpt_every, checkpoint_epochs=(), units_per_epoch=1) -> tuple:
    """The checkpoint epochs to score: the ``ckpt_every`` grid, or a pinned subset of it.

    ``units_per_epoch=k`` counts in 1/k-epoch units (``k=2``, 3 epochs, ``ckpt_every=1`` ->
    ``(1, ..., 6)``).  ``ckpt_every`` sets what training writes, this what the estimator
    reads.  Pinning serves a checkpoint count no stride gives: SOURCE App. B.1 wants six over
    20 epochs, ``floor(20k/6) = (3, 6, 10, 13, 16, 20)``.  A pinned list must be a subset of
    the saved grid (else a missing file or the wrong epoch's weights), strictly increasing
    (Eq. 1 reads it as an ordered partition) and end at ``epochs`` (see :func:`ckpt_epochs`).
    """
    epochs = int(epochs) * int(units_per_epoch)
    saved = ckpt_epochs(epochs, ckpt_every)
    if not checkpoint_epochs:
        return saved
    want = tuple(int(e) for e in checkpoint_epochs)
    if list(want) != sorted(set(want)):
        raise ValueError(f"[tracin] checkpoint_epochs must be strictly increasing with no "
                         f"repeats, got {list(want)}")
    if want[0] < 1 or want[-1] != int(epochs):
        raise ValueError(f"[tracin] checkpoint_epochs must lie in [1, {int(epochs)}] and "
                         f"END at epoch {int(epochs)} (the final checkpoint carries the "
                         f"largest loss reduction), got {list(want)}")
    missing = [e for e in want if e not in set(saved)]
    if missing:
        raise ValueError(f"[tracin] checkpoint_epochs {missing} are not saved by "
                         f"ckpt_every={int(ckpt_every)} (which writes {list(saved)}); "
                         f"the scored epochs must be a subset of the trained ones.")
    return want


def _recipe_lr_lambda(recipe, steps_per_epoch):
    """The trainers' LambdaLR multiplier for the recipe, so Eq. 1 reads the stepped schedule."""
    if isinstance(recipe, Resnet9Recipe):
        return triangular_lr(recipe.epochs, recipe.peak_epoch, steps_per_epoch)
    if isinstance(recipe, BertRecipe):
        return linear_decay_lr(recipe.epochs, steps_per_epoch)
    if isinstance(recipe, MnistRecipe):
        return lambda _step: 1.0
    if isinstance(recipe, Gpt2WikitextBaeRecipe):       # trained by bergson, not by us
        if recipe.lr_schedule != "constant":
            raise ValueError(f"[tracin] only bergson's constant schedule is wired for "
                             f"{type(recipe).__name__}, got {recipe.lr_schedule!r}")
        return lambda _step: 1.0
    raise ValueError(f"[tracin] no pinned LR schedule wired for recipe "
                     f"{type(recipe).__name__}; add its lambda here WITH a test against "
                     f"the trainer's stepped LRs.")


def epoch_mean_lrs(recipe, n_train, units_per_epoch=1) -> np.ndarray:
    """Mean learning rate stepped in each epoch of a full-set run, ``(epochs,)`` float64.
    Step k runs at ``peak_lr * lambda(k)`` (trainers step ``sched`` after the optimizer) and
    epoch e covers steps ``[e*spe, (e+1)*spe)``, ``spe = ceil(n_train / batch)``.  With
    ``units_per_epoch=k``: ``(k*epochs,)`` per-unit means; a ragged unit is refused, since it
    would give two checkpoints different step counts under one weight."""
    spe = max(1, int(np.ceil(int(n_train) / recipe.batch)))
    k = int(units_per_epoch)
    if spe % k:
        raise ValueError(f"[tracin] {spe} steps per epoch do not split into "
                         f"units_per_epoch={k} equal units")
    lam = _recipe_lr_lambda(recipe, spe)
    mult = np.array([lam(k_) for k_ in range(recipe.epochs * spe)], dtype=np.float64)
    return float(recipe.peak_lr) * mult.reshape(recipe.epochs * k, spe // k).mean(axis=1)


def ckpt_lr_weights(recipe, n_train, args: TracinArgs) -> np.ndarray:
    """Eq. 1 weight per checkpoint.  ``"schedule"``: the sum of the mean stepped LRs of the
    epochs since the previous checkpoint, each visiting every example once (at
    ``ckpt_every=1`` exactly the paper's eta_i); a 1/k-epoch unit visits each example 1/k
    times in expectation, so it adds its mean LR / k.  ``"equal"``: ones."""
    k = int(args.units_per_epoch)
    cps = resolve_ckpt_epochs(recipe.epochs, args.ckpt_every, args.checkpoint_epochs, k)
    if args.lr_weighting == "equal":
        return np.ones(len(cps), dtype=np.float64)
    means = epoch_mean_lrs(recipe, n_train, k)
    prev = 0
    out = []
    for t in cps:
        out.append(means[prev:t].sum() / k)
        prev = t
    return np.array(out, dtype=np.float64)


# -- trajectory checkpoints (full-set runs, stamped, resumable per run) ----- #
def trajectory_paths(traj_dir, r, cps, units_per_epoch=1) -> list:
    """Run ``r``'s checkpoint paths for the 1-based epochs ``cps`` (the CLI's cost estimate
    reads them too); unit files (``k > 1``) carry ``k`` in the name, never passing for epochs."""
    run_dir = os.path.join(traj_dir, f"run_{int(r):02d}")
    k = int(units_per_epoch)
    if k == 1:
        return [os.path.join(run_dir, f"ep_{int(e):03d}.pt") for e in cps]
    return [os.path.join(run_dir, f"unit_{int(u):03d}_of{k}.pt") for u in cps]


def _check_runs(runs, n_runs, where):
    """``runs`` -> a validated set of run indices inside ``range(n_runs)``."""
    rs = sorted({int(r) for r in runs})
    bad = [r for r in rs if not 0 <= r < n_runs]
    if not rs or bad:
        raise ValueError(f"[tracin] {where}: runs {list(runs)!r} must be a non-empty "
                         f"subset of 0..{n_runs - 1} (bad: {bad}).")
    return set(rs)


def train_trajectories(traj_dir, data, recipe, n_runs, train_fn, *, ckpt_every,
                       runs=None, verbose=True) -> list:
    """Train (or resume) the full-set trajectory runs; returns each run's ordered checkpoint paths.

    Run ``r`` trains on ``arange(n_train)``: TracIn traces training on S itself, and a full-set
    trajectory scores every query (Remark 3.5).  ``train_fn(data, sub_idx, seed, on_epoch_end)``
    runs with seed ``TRACIN_SEED_BASE + r``; its hook saves ``state_dict`` at each kept epoch end
    (atomic, RNG-neutral).  ``runs`` trains only that subset, so multi-GPU workers on disjoint
    runs can share ``traj_dir``; the result still spans all ``n_runs``, so check skipped paths.
    A different or missing ``_provenance.json`` stamp (protocol + recipe; ``ckpt_every`` only
    selects files) over existing checkpoints is refused.  A run missing any checkpoint is
    retrained whole: GPU-atomics noise makes its files consistent only within one process."""
    os.makedirs(traj_dir, exist_ok=True)
    stamp_path = os.path.join(traj_dir, "_provenance.json")
    stamp = json.dumps({"protocol": "full-set-trajectory", "seed_base": TRACIN_SEED_BASE,
                        "recipe": recipe.to_dict(), "dataset": data["meta"]["dataset"],
                        "n_train": int(data["meta"]["n_train"])}, sort_keys=True)
    if os.path.exists(stamp_path):
        with open(stamp_path) as fh:
            stored = fh.read()
        if stored != stamp:
            raise RuntimeError(f"[tracin] {traj_dir} holds trajectories from a DIFFERENT "
                               f"protocol/recipe; refusing to mix.\n  stored : {stored}\n"
                               f"  current: {stamp}")
    else:
        loose = [os.path.join(dp, f) for dp, _, fs in os.walk(traj_dir)
                 for f in fs if f.endswith(".pt")]
        if loose:
            raise RuntimeError(f"[tracin] {traj_dir} holds {len(loose)} UNSTAMPED "
                               f"checkpoint(s); refusing to reuse them. Delete the "
                               f"directory or use a fresh tag.")
        with open(stamp_path, "w") as fh:
            fh.write(stamp)

    n = int(data["meta"]["n_train"])
    cps = ckpt_epochs(recipe.epochs, ckpt_every)
    todo = (set(range(int(n_runs))) if runs is None
            else _check_runs(runs, int(n_runs), "train_trajectories"))
    out = []
    for r in range(int(n_runs)):
        paths = trajectory_paths(traj_dir, r, cps)
        out.append(paths)
        if r not in todo or all(_ckpt_valid(p) for p in paths):
            continue
        run_dir = os.path.dirname(paths[0])
        shutil.rmtree(run_dir, ignore_errors=True)      # never mix two processes' epochs
        os.makedirs(run_dir, exist_ok=True)
        by_epoch = dict(zip(cps, paths))
        seed = TRACIN_SEED_BASE + r

        def on_epoch_end(epoch, model, _by_epoch=by_epoch):
            path = _by_epoch.get(int(epoch) + 1)        # loop epoch is 0-based
            if path is not None:
                _save_ckpt_atomic(model.state_dict(), path)

        model = train_fn(data, np.arange(n, dtype=np.int64), seed, on_epoch_end)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        missing = [p for p in paths if not _ckpt_valid(p)]
        if missing:
            raise RuntimeError(f"[tracin] run {r} finished but {len(missing)} checkpoint(s) "
                               f"were never saved (first: {missing[0]}); the trainer must "
                               f"run all {recipe.epochs} epochs and call on_epoch_end.")
        if verbose:
            print(f"[tracin] trained trajectory {r + 1}/{n_runs} (full {n}-point set, "
                  f"seed {seed}, {len(cps)} checkpoints)", flush=True)
    return out


# -- per-sample projected loss gradients (torch.func, chunked) -------------- #
def _traced_params(model, tracked=()):
    """The parameters TracIn differentiates, in ``named_parameters`` order: all, or the weight
    and bias of each ``tracked`` module (a name owning none is refused: a typo must not shrink
    the set).  ``functional_call`` takes the rest from the module, so an untracked tied ``wte``
    stays tied.  Freezes the module: a trainable untracked parameter would make each per-sample
    gradient keep a graph pinning the chunk's activations (~17 GiB per batch-8 GPT-2 chunk)."""
    model.requires_grad_(False)
    named = {k: v.detach() for k, v in model.named_parameters()}
    if not tracked:
        return named
    want = set(tracked)
    out = {k: v for k, v in named.items() if k.rsplit(".", 1)[0] in want}
    missing = sorted(want - {k.rsplit(".", 1)[0] for k in out})
    if missing:
        raise ValueError(f"[tracin] tracked module(s) own no parameters: {missing[:3]}"
                         f"{' ...' if len(missing) > 3 else ''}")
    return out


def _grad_chunk_fn_image(model, params, buffers, X, y, ls):
    """``chunk(i0, i1) -> (i1-i0, P)`` flattened per-sample train-loss gradients, in
    ``named_parameters`` order (the flattening the projector sees — order is identity)."""
    from torch.func import functional_call, grad, vmap

    def loss_one(p, b, x, yy):
        logits = functional_call(model, (p, b), (x.unsqueeze(0),))
        return F.cross_entropy(logits, yy.unsqueeze(0), label_smoothing=ls)

    g1 = vmap(grad(loss_one), in_dims=(None, None, 0, 0))

    def chunk(i0, i1):
        gs = g1(params, buffers, X[i0:i1], y[i0:i1])
        return torch.cat([v.reshape(i1 - i0, -1) for v in gs.values()], dim=1)

    return chunk


def _grad_chunk_fn_text(model, params, buffers, ids, tt, am, y, ls):
    """Text twin of :func:`_grad_chunk_fn_image` — the model is called by keyword like
    traker's text task (and our ``_BertLogits`` forward)."""
    from torch.func import functional_call, grad, vmap

    def loss_one(p, b, i, t, a, yy):
        # functional_call's args is a required positional (torch>=2.5): pass () with kwargs
        logits = functional_call(model, (p, b), (), kwargs={
            "input_ids": i.unsqueeze(0), "token_type_ids": t.unsqueeze(0),
            "attention_mask": a.unsqueeze(0)})
        return F.cross_entropy(logits, yy.unsqueeze(0), label_smoothing=ls)

    g1 = vmap(grad(loss_one), in_dims=(None, None, 0, 0, 0, 0))

    def chunk(i0, i1):
        gs = g1(params, buffers, ids[i0:i1], tt[i0:i1], am[i0:i1], y[i0:i1])
        return torch.cat([v.reshape(i1 - i0, -1) for v in gs.values()], dim=1)

    return chunk


def _grad_chunk_fn_lm(model, params, buffers, ids):
    """Causal-LM twin: per-document mean next-token loss (``wikitext_gpt2`` convention 3) on
    both sides, as Bae et al. measure and TracIn traces.  ``functional_call`` keeps the
    ``lm_head``/``wte`` tie (``tie_weights=True``), so ``wte`` receives both gradients."""
    from torch.func import functional_call, grad, vmap

    from corrected_lds.pools.tasks.wikitext_gpt2 import per_doc_losses

    def loss_one(p, b, i):
        x = i.unsqueeze(0)
        return per_doc_losses(functional_call(model, (p, b), (x,)), x)[0]

    g1 = vmap(grad(loss_one), in_dims=(None, None, 0))

    def chunk(i0, i1):
        gs = g1(params, buffers, ids[i0:i1])
        return torch.cat([v.reshape(i1 - i0, -1) for v in gs.values()], dim=1)

    return chunk


def _make_projector(grad_dim, proj_dim, device, *, dtype, require_cuda,
                    proj_max_batch_size=8):
    """traker's projector at ``seed=0`` (per-run matrices via ``model_id``), or None (identity)
    for ``proj_dim=None``.  fast_jl ``CudaProjector`` on CUDA, else ``BasicProjector`` unless
    ``require_cuda`` fails closed: its matrix alone is ~40 GB at BERT scale."""
    if proj_dim is None:
        return None
    from corrected_lds.tda.trak import _basic_projector
    dev = torch.device(device)
    if dev.type == "cuda":
        try:
            from trak.projectors import CudaProjector, ProjectionType
            return CudaProjector(grad_dim=int(grad_dim), proj_dim=int(proj_dim), seed=0,
                                 proj_type=ProjectionType.rademacher, device=dev,
                                 max_batch_size=int(proj_max_batch_size))
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as e:
            if require_cuda:
                raise RuntimeError(
                    f"[tracin] require_cuda_projector=True but the CUDA projector is "
                    f"unavailable at k={proj_dim} ({str(e).splitlines()[0][:120]}); "
                    f"install/fix fast_jl, don't downgrade.") from e
            print(f"    [tracin] CUDA projector unavailable "
                  f"({str(e).splitlines()[0][:80]}); falling back to BasicProjector at "
                  f"k={proj_dim}.", flush=True)
    elif require_cuda:
        raise RuntimeError(f"[tracin] require_cuda_projector=True on non-CUDA device "
                           f"{device!r}.")
    return _basic_projector(grad_dim, proj_dim, device, dtype)


def _project_split(chunk_fn, n, batch, projector, model_id, dtype):
    """One split's projected per-sample gradients, ``(n, d)`` float32 on the compute
    device (``d = P`` when ``projector`` is None)."""
    outs = []
    for i0 in range(0, int(n), int(batch)):
        i1 = min(int(n), i0 + int(batch))
        g = chunk_fn(i0, i1)
        if projector is None:
            outs.append(g.detach().float())
        else:
            outs.append(projector.project(g.detach().to(dtype).contiguous(),
                                          model_id=int(model_id)).float())
    return torch.cat(outs)


def _single_run_tauq(ckpt_paths, weights, data, model_ctor, *, projector, args, device,
                     batch, mid, verbose=False) -> np.ndarray:
    """One trajectory's ``sum_t w_t * Phi_t @ Psi_t^T`` -> ``(n_train, Q)`` float64, from each
    checkpoint's projected train (Phi) and query (Psi) loss gradients in eval mode."""
    n, Q = int(data["meta"]["n_train"]), int(data["meta"]["n_targets"])
    text = "input_ids" in data
    ls = float(args.label_smoothing)
    dtype = torch.float16 if args.use_half_precision else torch.float32
    score = np.zeros((n, Q))
    for t, (ck, w) in enumerate(zip(ckpt_paths, weights, strict=True)):
        # one projection per run, or per (run, checkpoint); shared by both sides either way
        pid = (mid * len(ckpt_paths) + t) if args.projection_per_checkpoint else mid
        model = model_ctor().to(device).eval()
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        params = _traced_params(model, args.tracked)
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        if "lm_ids" in data:                     # causal LM (wikitext): no labels
            tr = _grad_chunk_fn_lm(model, params, buffers, data["lm_ids"])
            qy = _grad_chunk_fn_lm(model, params, buffers, data["q_lm_ids"])
        elif text:
            tr = _grad_chunk_fn_text(model, params, buffers, data["input_ids"],
                                     data["token_type_ids"], data["attention_mask"],
                                     data["labels"], ls)
            qy = _grad_chunk_fn_text(model, params, buffers, data["q_input_ids"],
                                     data["q_token_type_ids"], data["q_attention_mask"],
                                     data["q_labels"], ls)
        else:
            tr = _grad_chunk_fn_image(model, params, buffers, data["X_train"],
                                      data["y_train"], ls)
            qy = _grad_chunk_fn_image(model, params, buffers, data["X_query"],
                                      data["y_query"], ls)
        phi = _project_split(tr, n, batch, projector, pid, dtype)     # (n, d)
        psi = _project_split(qy, Q, batch, projector, pid, dtype)     # (Q, d)
        score += float(w) * (phi @ psi.T).double().cpu().numpy()
        del model, phi, psi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if verbose:
        print(f"    [tracin] run {mid} scored over {len(ckpt_paths)} checkpoints "
              f"(mean |tau_r| {np.abs(score).mean():.3e})", flush=True)
    return score


# -- per-run partials: kept, recombinable over any run set ------------------ #
def run_partial_path(part_dir, r) -> str:
    return os.path.join(part_dir, f"tauq_tracin_r_{int(r):03d}.npz")


def _files_md5(paths) -> str:
    """One md5 over the ordered checkpoint contents (order pairs them with the weights)."""
    h = hashlib.md5()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"|")
    return h.hexdigest()


def _partial_valid(path, *, mid, ckpts_md5, prov_json, args_json, weights) -> bool:
    """Whether a partial is reusable: its run id, checkpoint contents, TracIn arguments, Eq. 1
    weights, provenance and traker version must all match (else it is recomputed)."""
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001 — a truncated write is simply "not reusable"
        return False
    if "tracin_args" not in z.files:
        return False
    return (int(z["mid"]) == int(mid)
            and str(z["ckpts_md5"]) == ckpts_md5
            and str(z["provenance"]) == prov_json
            and str(z["tracin_args"]) == args_json
            and str(z["traker_version"]) == TRAKER_VERSION
            and np.array_equal(z["lr_weights"], np.asarray(weights)))


def combine_tracin_partials(part_dir, runs) -> np.ndarray:
    """Plain mean of the per-run ``tauq`` partials over any run ids, not only a prefix (so
    disjoint replicates are a re-combine); returns ``(n_train, Q)`` float64."""
    runs = [int(r) for r in runs]
    if not runs:
        raise ValueError("combine_tracin_partials: `runs` is empty")
    if len(set(runs)) != len(runs):
        raise ValueError(f"combine_tracin_partials: duplicate runs in {runs}")
    total = None
    for r in runs:
        p = run_partial_path(part_dir, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"combine_tracin_partials: missing partial for run "
                                    f"{r}: {p}")
        t = np.load(p, allow_pickle=False)["tauq"].astype(np.float64)
        total = t if total is None else total + t
    return total / len(runs)


# -- the public entry point ------------------------------------------------- #
def compute_tauq_tracin(traj_ckpts, data, model_ctor, *, recipe, args=None,
                        proj_dim=1_024, device="cuda", batch=256, work_dir=None,
                        provenance=None, verbose=False, require_cuda_projector=False,
                        proj_max_batch_size=8, runs=None):
    """Trajectory-ensembled TracInCP ``tauq (n_train, Q)`` from ``traj_ckpts``
    (per-run checkpoint path lists, e.g. :func:`train_trajectories` output).

    ``recipe`` gives the Eq. 1 weights and must be the one the trajectories were trained with.
    ``proj_dim=None`` skips projection (tests).  Per-run partials in ``work_dir`` make it
    resumable; it returns their plain mean (:func:`combine_tracin_partials` for subsets)."""
    if not traj_ckpts:
        raise ValueError("compute_tauq_tracin: no trajectory checkpoints given")
    args = args or TracinArgs()
    n = int(data["meta"]["n_train"])
    weights = ckpt_lr_weights(recipe, n, args)
    for r, paths in enumerate(traj_ckpts):
        if len(paths) != len(weights):
            raise ValueError(f"compute_tauq_tracin: run {r} has {len(paths)} checkpoints "
                             f"but the recipe/ckpt_every imply {len(weights)}.")
    work_dir = work_dir or os.path.join(os.getcwd(), "tracin_work")
    os.makedirs(work_dir, exist_ok=True)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    args_json = json.dumps(args.to_dict(), sort_keys=True)

    todo = (set(range(len(traj_ckpts))) if runs is None
            else _check_runs(runs, len(traj_ckpts), "compute_tauq_tracin"))
    projector = None
    if proj_dim is not None and any(
            r in todo and not _partial_valid(
                run_partial_path(work_dir, r),
                mid=r, ckpts_md5=_files_md5(paths), prov_json=prov_json,
                args_json=args_json, weights=weights)
            for r, paths in enumerate(traj_ckpts)):
        probe = model_ctor()
        grad_dim = sum(p.numel() for p in _traced_params(probe, args.tracked).values())
        del probe
        projector = _make_projector(
            grad_dim, proj_dim, device,
            dtype=torch.float16 if args.use_half_precision else torch.float32,
            require_cuda=require_cuda_projector, proj_max_batch_size=proj_max_batch_size)

    for r, paths in enumerate(traj_ckpts):
        if r not in todo:
            continue
        path = run_partial_path(work_dir, r)
        ckpts_md5 = _files_md5(paths)
        if _partial_valid(path, mid=r, ckpts_md5=ckpts_md5, prov_json=prov_json,
                          args_json=args_json, weights=weights):
            if verbose:
                print(f"    [tracin] run {r} partial present (identity ok) -> reuse",
                      flush=True)
            continue
        tauq_r = _single_run_tauq(paths, weights, data, model_ctor, projector=projector,
                                  args=args, device=device, batch=batch, mid=r,
                                  verbose=verbose)
        tmp = path + ".tmp.npz"                              # atomic: temp then rename
        np.savez(tmp, tauq=tauq_r.astype(np.float32), mid=r, ckpts_md5=ckpts_md5,
                 lr_weights=weights, provenance=prov_json, tracin_args=args_json,
                 traker_version=TRAKER_VERSION)
        os.replace(tmp, path)
    if runs is not None:            # one worker of a split: partials only, nothing to mean
        return None
    tauq = combine_tracin_partials(work_dir, range(len(traj_ckpts)))
    if verbose:
        print(f"[tracin] tauq {tauq.shape} from {len(traj_ckpts)} trajectory(ies) x "
              f"{len(weights)} checkpoints (plain mean, {args.lr_weighting} weights)",
              flush=True)
    return tauq
