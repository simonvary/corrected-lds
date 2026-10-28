"""Ensembled TRAK at the paper's projection, via the official ``trak.TRAKer``.

Produces ``tauq`` of shape ``(n_train, n_targets)``: ``tauq[i, q]`` is example i's TRAK
score for query q, so ``g_q(S) = sum_{i in S} tauq[i, q]`` is the LDS prediction.

Member ``m`` trains on its own random alpha-subset with seed ``ENS_SEED_BASE + m`` (paper
Sec. 3 / App. E.3), so resumes rebuild identical members and the first K of N checkpoints
are a K-member ensemble.  Members combine by paper Eq. 15 (scores and ``out_to_loss``
weights averaged separately, then multiplied) or, for ablation, by a plain mean, both in one
featurize pass (:func:`_ensemble_accumulate`) holding one model's projected grads in
``scratch_dir`` at a time (~3-4 GB at n=50k, k=20 480; keep it off quota'd volumes).

``n_models`` (the paper's M) is ``E * e_features``: ``e_features`` members form one Eq. 15
estimator (the preset knob; the frozen schema-1 npz key ``models_per_replica``) and ``E``
such units are plain-mean replicas, the pool_tda slice axis that mirrors the pool's R seeds
and obeys tda_noise's 1/E law.  EK-FAC and TracIn have ``e_features = 1``: ``E = n_models``.

traker's classification output is the correct-class log-odds margin, identical to the pool
readout (:func:`corrected_lds.pools.train.per_query_margin`).  Pinned to ``traker==0.3.2``;
the version is stamped into every accumulator and tauq file.
"""

import json
import os
import shutil

import numpy as np
import torch

try:
    from importlib.metadata import version as _pkg_version
    TRAKER_VERSION = _pkg_version("traker")
except Exception:  # noqa: BLE001 — any metadata failure means "unknown"
    TRAKER_VERSION = "unknown"

__all__ = ["AUG_FEAT_SEED", "ENS_SEED_BASE", "TRAKER_VERSION", "check_proj_dim",
           "compute_tauq", "compute_tauq_lm", "compute_tauq_text", "lm_margins",
           "lm_out_to_loss", "make_feat_aug", "member_subset", "pieces_path",
           "train_ensemble"]

ENS_SEED_BASE = 10_000        # ensemble member m -> training seed ENS_SEED_BASE + m
AUG_FEAT_SEED = 20_000        # featurize-on-augmented base seed (per-model = base + model_id)


def check_proj_dim(proj_dim, n_train):
    """Refuse ``proj_dim >= n_train / 2``: a projection comparable to the maskable universe
    gave pure-noise LDS (k=20 480 on a 20 000-row universe scored 0.010)."""
    if int(proj_dim) * 2 >= int(n_train):
        raise ValueError(
            f"[trak] proj_dim={proj_dim} >= n_train/2 (n={n_train}): a projection "
            f"dimension comparable to the maskable universe measured pure-noise LDS. "
            f"Lower proj_dim.")


# -- ensemble checkpoints (subset-trained, seed-determined, resumable) ------ #
def _save_ckpt_atomic(state_dict, path):
    """Save via temp file + os.replace, so a crash never leaves a corrupt .pt to resume from."""
    tmp = path + ".tmp"
    torch.save({k: v.cpu() for k, v in state_dict.items()}, tmp)
    os.replace(tmp, path)


def _ckpt_valid(path):
    """A checkpoint counts as done only if it exists and loads (rejects a truncated .pt)."""
    if not os.path.exists(path):
        return False
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:  # noqa: BLE001 — a truncated/corrupt file is simply "not done"
        return False


def member_subset(m, n, alpha) -> np.ndarray:
    """Member ``m``'s training rows: ``round(alpha * n)`` of ``n`` in the draw order of
    ``default_rng(ENS_SEED_BASE + m)``.  The one home of the rule, so a member trained by an
    external harness (``pools/bergson.py``) gets the rows :func:`train_ensemble` would use."""
    n_sub = max(1, round(alpha * int(n)))
    return np.random.default_rng(ENS_SEED_BASE + int(m)).choice(int(n), size=n_sub,
                                                                replace=False)


def train_ensemble(ckpt_dir, data, recipe, n_models, train_fn, verbose=True,
                   members=None):
    """Train or resume the TRAK ensemble; returns all ``n_models`` index-ordered checkpoint paths.

    Member ``m`` trains on its own :func:`member_subset` (the paper's protocol, not the pool's
    masks) by a ``PoolTask``'s ``train_fn(data, sub_idx, seed)``, seed ``ENS_SEED_BASE + m``.
    Checkpoints under a different or missing ``_provenance.json`` stamp are refused, so stale
    members never mix into a fresh tauq.  ``members`` trains only that subset, so per-GPU
    processes on disjoint subsets can share ``ckpt_dir``: members are seed-determined and
    saved atomically."""
    os.makedirs(ckpt_dir, exist_ok=True)
    stamp_path = os.path.join(ckpt_dir, "_provenance.json")
    stamp = json.dumps({"protocol": "subset-trained", "seed_base": ENS_SEED_BASE,
                        "recipe": recipe.to_dict(), "dataset": data["meta"]["dataset"],
                        "n_train": int(data["meta"]["n_train"])}, sort_keys=True)
    if os.path.exists(stamp_path):
        with open(stamp_path) as fh:
            stored = fh.read()
        if stored != stamp:
            raise RuntimeError(f"[trak] {ckpt_dir} holds checkpoints from a DIFFERENT "
                               f"protocol/recipe; refusing to mix.\n  stored : {stored}\n"
                               f"  current: {stamp}")
    else:
        if any(f.endswith(".pt") for f in os.listdir(ckpt_dir)):
            raise RuntimeError(f"[trak] {ckpt_dir} holds UNSTAMPED checkpoints; refusing "
                               f"to reuse them. Delete the directory or use a fresh tag.")
        with open(stamp_path, "w") as fh:
            fh.write(stamp)

    n = int(data["meta"]["n_train"])
    n_sub = max(1, round(recipe.alpha * n))
    paths = [os.path.join(ckpt_dir, f"member_{m:03d}.pt") for m in range(int(n_models))]
    todo = (set(range(int(n_models))) if members is None
            else _check_members(members, int(n_models), "train_ensemble"))
    for m, path in enumerate(paths):
        if m not in todo or _ckpt_valid(path):
            continue
        seed = ENS_SEED_BASE + m
        sub = member_subset(m, n, recipe.alpha)
        model = train_fn(data, sub, seed)
        _save_ckpt_atomic(model.state_dict(), path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"[trak] trained ensemble member {m + 1}/{n_models} "
                  f"(random {n_sub}/{n} subset, seed {seed})", flush=True)
    return paths


def make_feat_aug(recipe, seed_base=AUG_FEAT_SEED):
    """``feat_aug(x_batch, model_id) -> x_batch`` for featurize-on-augmented (cifar): train
    batches get the recipe's augmentation, queries stay clean.  Each model's generator, seeded
    ``seed_base + model_id`` and advanced across batches, gives any worker the same stream."""
    from corrected_lds.pools.tasks.cifar_resnet9 import augment
    gens = {}

    def feat_aug(xb, mid):
        g = gens.get(int(mid))
        if g is None:
            g = torch.Generator(device=xb.device).manual_seed(int(seed_base) + int(mid))
            gens[int(mid)] = g
        return augment(xb, recipe.pad, recipe.cutout, g, per_sample=recipe.per_sample_aug)

    return feat_aug


# -- projector plumbing ----------------------------------------------------- #
def _basic_projector(grad_dim, proj_dim, device, dtype):
    """A ``BasicProjector`` whose ``free_memory`` is a no-op: traker 0.3.2 frees
    ``proj_matrix`` in ``finalize_features`` and regenerates it only for a new ``model_id``,
    but scoring must reuse the matrix its model was featurized with.  ``dtype`` tracks
    ``use_half_precision``, as in traker's own construction."""
    from trak.projectors import BasicProjector, ProjectionType

    class _PersistentBasicProjector(BasicProjector):
        def free_memory(self):     # keep proj_matrix across featurize -> score
            pass

    return _PersistentBasicProjector(grad_dim=int(grad_dim), proj_dim=int(proj_dim),
                                     seed=0, proj_type=ProjectionType.rademacher,
                                     device=device, dtype=dtype)


_CUDA_INIT = None                           # traker's own CudaProjector.__init__


def pin_projector_sms(num_sms):
    """Make every traker ``CudaProjector`` built from now on (``ChunkedCudaProjector``'s chunks
    included) use ``num_sms`` streaming multiprocessors, or the GPU's own count for None.

    fast_jl draws its projection from the seed and the SM count, so one seed gives a different
    projection on each GPU model: a release reproduces only at the count it was made with."""
    global _CUDA_INIT
    from trak.projectors import CudaProjector
    if _CUDA_INIT is None:
        _CUDA_INIT = CudaProjector.__init__
    if num_sms is None:
        CudaProjector.__init__ = _CUDA_INIT
        return

    def init(self, *args, **kwargs):
        _CUDA_INIT(self, *args, **kwargs)
        self.num_sms = int(num_sms)

    CudaProjector.__init__ = init


def _is_projector_resource_error(e) -> bool:
    """Whether ``e`` is fast_jl's launch config exceeding the GPU (resources, batch size)."""
    s = str(e).lower()
    return ("too large" in s or "resources requested" in s or "too many resources" in s)


def _assert_cuda_projector(traker, proj_dim, where):
    """Fail closed for ``require_cuda_projector``: traker silently selects ``BasicProjector``
    when ``fast_jl`` is missing, so check the projector it actually selected."""
    name = type(getattr(traker, "projector", None)).__name__
    if "Cuda" not in name:                       # CudaProjector / ChunkedCudaProjector
        raise RuntimeError(
            f"[{where}] require_cuda_projector=True but traker selected {name!r} at "
            f"k={proj_dim} (fast_jl unavailable). Refusing the pure-torch BasicProjector "
            f"fallback (catastrophic at this scale) — install/fix fast_jl, don't downgrade.")


# -- per-model unweighted pieces (Eq. 15 inputs) ---------------------------- #
def _finalize_unweighted(traker, exp_name, model_id, device):
    """One model's ``(scores_m (n,Q) f64, q_m (n,) f64)``: the unweighted projected-kernel
    scores and the ``out_to_loss`` term (Q = 1-p), kept apart for Eq. 15.  Mirrors
    ``trak.TRAKer.finalize_scores`` (traker 0.3.2) exactly, up to its final multiply."""
    saver = traker.saver
    num_targets = saver.experiments[exp_name]["num_targets"]
    # establish the scores mmap + shape (as finalize_scores does on its first model_id)
    saver.load_current_store(model_id, exp_name, num_targets)
    scores_shape = saver.current_store[f"{exp_name}_scores"].shape
    # load this model's finalized train features + target grads + out_to_loss
    saver.load_current_store(model_id)
    saver.load_current_store(model_id, exp_name, num_targets)
    g = torch.as_tensor(saver.current_store["features"], device=device)
    g_target = torch.as_tensor(saver.current_store[f"{exp_name}_grads"], device=device)
    acc = torch.zeros(*scores_shape, device="cpu")             # (n_train, Q), unweighted
    traker.score_computer.get_scores(g, g_target, accumulator=acc)
    q = np.asarray(saver.current_store["out_to_loss"]).reshape(-1).astype(np.float64)  # (n,)
    return acc.numpy().astype(np.float64), q


def _single_model_pieces(state_dict, data, model_ctor, *, proj_dim, device, batch,
                         save_dir, exp_name, force_basic_projector,
                         proj_max_batch_size, model_id, verbose,
                         require_cuda_projector=False, featurize_augment=None,
                         use_half_precision=False, train_batches=None):
    """One image model's unweighted TRAK pieces (featurize -> score -> extract).

    ``featurize_augment(x_batch, model_id)`` augments train featurize batches only (the paper's
    cifar protocol).  ``train_batches``, a zero-arg factory of ``(xb, yb)`` in canonical row
    order, replaces ``X_train`` for streaming tasks (imagenet) and excludes ``featurize_augment``.
    Queries always use ``X_query``, as the margins readout does.  ``use_half_precision`` (tauq
    identity; off by default, fp16 in traker) also sets the fallback ``BasicProjector`` dtype."""
    from trak import TRAKer

    model = model_ctor().to(device).eval()
    n_train = int(data["meta"]["n_train"])
    Q = int(data["meta"]["n_targets"])
    os.makedirs(save_dir, exist_ok=True)

    projector = None
    if force_basic_projector:
        grad_dim = sum(p.numel() for p in model.parameters())
        projector = _basic_projector(grad_dim, proj_dim, device,
                                     torch.float16 if use_half_precision else torch.float32)

    traker = TRAKer(model=model, task="image_classification", train_set_size=n_train,
                    save_dir=save_dir, device=device, proj_dim=int(proj_dim),
                    projector=projector, lambda_reg=0.0,
                    proj_max_batch_size=int(proj_max_batch_size), projector_seed=0,
                    use_half_precision=bool(use_half_precision), logging_level=30)
    if require_cuda_projector and not force_basic_projector:
        _assert_cuda_projector(traker, proj_dim, "trak")
    if train_batches is not None and featurize_augment is not None:
        raise ValueError("train_batches and featurize_augment are mutually exclusive: "
                         "a streaming featurize path owns its own decode/augment policy.")
    Xq, yq = data["X_query"], data["y_query"]
    try:
        traker.load_checkpoint(state_dict, model_id=int(model_id))
        if train_batches is not None:                  # streaming tasks (imagenet)
            for xb, yb in train_batches():
                traker.featurize(batch=(xb.contiguous(), yb.contiguous()),
                                 num_samples=len(yb))
        else:
            Xtr, ytr = data["X_train"], data["y_train"]
            for i in range(0, n_train, batch):
                xb = Xtr[i:i + batch]
                if featurize_augment is not None:
                    xb = featurize_augment(xb, int(model_id))
                traker.featurize(batch=(xb.contiguous(), ytr[i:i + batch].contiguous()),
                                 num_samples=int(min(batch, n_train - i)))
        traker.finalize_features()
        traker.start_scoring_checkpoint(exp_name=exp_name, checkpoint=state_dict,
                                        model_id=int(model_id), num_targets=Q)
        for i in range(0, Q, batch):
            traker.score(batch=(Xq[i:i + batch].contiguous(), yq[i:i + batch].contiguous()),
                         num_samples=int(min(batch, Q - i)))
        scores_m, q_m = _finalize_unweighted(traker, exp_name, int(model_id), device)
    except (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError)) as e:
        if force_basic_projector or not _is_projector_resource_error(e):
            raise
        if require_cuda_projector:                         # fail-closed: no silent fallback
            raise RuntimeError(
                f"[trak] CUDA projector refused the launch config at k={proj_dim} and "
                f"require_cuda_projector=True — refusing the BasicProjector fallback. "
                f"Original: {str(e).splitlines()[0][:120]}") from e
        print(f"    [trak] CUDA projector refused the launch config "
              f"({str(e).splitlines()[0][:80]}); falling back to BasicProjector at "
              f"k={proj_dim}.", flush=True)
        del traker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(save_dir, ignore_errors=True)
        return _single_model_pieces(state_dict, data, model_ctor, proj_dim=proj_dim,
                                    device=device, batch=batch,
                                    save_dir=save_dir, exp_name=exp_name,
                                    force_basic_projector=True,
                                    proj_max_batch_size=proj_max_batch_size,
                                    model_id=model_id, verbose=verbose,
                                    require_cuda_projector=require_cuda_projector,
                                    featurize_augment=featurize_augment,
                                    use_half_precision=use_half_precision,
                                    train_batches=train_batches)
    return scores_m, q_m


def _single_model_pieces_text(state_dict, data, model_ctor, *, proj_dim, device, batch,
                              save_dir, exp_name, model_id, use_half_precision,
                              require_cuda_projector=True):
    """One BERT's unweighted TRAK pieces via traker's ``text_classification`` task.

    Batches are ``(input_ids, token_type_ids, attention_mask, labels)``, the order traker
    0.3.2 expects.  ``require_cuda_projector`` fails closed: pure torch is catastrophic here."""
    from trak import TRAKer

    model = model_ctor().to(device).eval()
    n_train = int(data["meta"]["n_train"])
    Q = int(data["meta"]["n_targets"])
    os.makedirs(save_dir, exist_ok=True)
    traker = TRAKer(model=model, task="text_classification", train_set_size=n_train,
                    save_dir=save_dir, device=device, proj_dim=int(proj_dim), lambda_reg=0.0,
                    use_half_precision=bool(use_half_precision), projector_seed=0,
                    logging_level=30)
    if require_cuda_projector:                    # traker silently picks BasicProjector otherwise
        _assert_cuda_projector(traker, proj_dim, "trak-text")
    tr = (data["input_ids"], data["token_type_ids"], data["attention_mask"], data["labels"])
    qy = (data["q_input_ids"], data["q_token_type_ids"], data["q_attention_mask"],
          data["q_labels"])
    try:
        traker.load_checkpoint(state_dict, model_id=int(model_id))
        for i in range(0, n_train, batch):
            traker.featurize(batch=tuple(t[i:i + batch].contiguous() for t in tr),
                             num_samples=int(min(batch, n_train - i)))
        traker.finalize_features()
        traker.start_scoring_checkpoint(exp_name=exp_name, checkpoint=state_dict,
                                        model_id=int(model_id), num_targets=Q)
        for i in range(0, Q, batch):
            traker.score(batch=tuple(t[i:i + batch].contiguous() for t in qy),
                         num_samples=int(min(batch, Q - i)))
        return _finalize_unweighted(traker, exp_name, int(model_id), device)
    except (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError)) as e:
        if require_cuda_projector and _is_projector_resource_error(e):
            raise RuntimeError(
                f"[trak-text] CUDA projector refused the launch config at k={proj_dim} and "
                f"require_cuda_projector=True — refusing the BasicProjector fallback. "
                f"Original: {str(e).splitlines()[0][:120]}") from e
        raise


# -- causal LM (GPT-2 / WikiText) ------------------------------------------- #
def lm_margins(logits, ids):
    """``(B,)`` TRAK model output for causal-LM documents: the sum over ``T - 1`` next tokens
    of the log-odds ``log p / (1 - p) = logit[correct] - logsumexp(other logits)``, traker's
    classification margin (canonical output (19)); the paper's LM choice (App. F.3, mT5)."""
    lg = logits[:, :-1, :].float()
    y = ids[:, 1:].unsqueeze(-1)
    correct = lg.gather(-1, y).squeeze(-1)
    others = lg.scatter(-1, y, float("-inf")).logsumexp(dim=-1)
    return (correct - others).sum(dim=-1)


def lm_out_to_loss(logits, ids):
    """``(B,)`` TRAK's ``Q`` for causal-LM documents: the mean over tokens of ``1 - p(correct)``.

    Per token ``d(-log p_t) = -(1 - p_t) d f_t`` for the log-odds ``f_t``, so a document's
    loss gradient is ``sum_t (1 - p_t) grad f_t`` (up to ``1/(T-1)``) while traker scales
    ``grad f = sum_t grad f_t`` by one ``Q``.  The mean equates them when ``1 - p_t`` does not
    covary with ``grad f_t``: the approximation made, which per-token featurization removes."""
    lg = logits[:, :-1, :].float()
    p = lg.softmax(dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (1.0 - p).mean(dim=-1)


def _lm_model_output():
    """A traker ``AbstractModelOutput`` for the causal LM (:func:`lm_margins` /
    :func:`lm_out_to_loss`); built on call so this module imports without traker."""
    from trak.modelout_functions import AbstractModelOutput

    class CausalLMModelOutput(AbstractModelOutput):
        def __init__(self):                       # abstract in traker 0.3.2
            super().__init__()

        @staticmethod
        def get_output(model, weights, buffers, input_id):
            ids = input_id.unsqueeze(0)
            logits = torch.func.functional_call(model, (weights, buffers), (ids,))
            return lm_margins(logits, ids).sum()

        def get_out_to_loss_grad(self, model, weights, buffers, batch):
            (ids,) = batch
            logits = torch.func.functional_call(model, (weights, buffers), (ids,))
            return lm_out_to_loss(logits, ids).detach().unsqueeze(-1)

    return CausalLMModelOutput()


def _tracked_gradient_computer(tracked):
    """traker's functional gradient computer over the ``tracked`` modules' parameters only.

    traker's ``grad_wrt`` differentiates every parameter before dropping the rest, paying for
    GPT-2's 38.6M-parameter tied ``wte`` per sample.  Here (via TracIn's ``_traced_params``)
    untracked parameters come from the frozen module, keeping the ``wte``/``lm_head`` tie;
    ``load_model_params`` reruns on every ``load_checkpoint``, re-reading each member."""
    from trak.gradient_computers import FunctionalGradientComputer

    from corrected_lds.tda.tracin import _traced_params

    class TrackedGradientComputer(FunctionalGradientComputer):
        def load_model_params(self, model):
            self.func_weights = _traced_params(model, tracked)
            self.func_buffers = dict(model.named_buffers())

    return TrackedGradientComputer


def _single_model_pieces_lm(state_dict, data, model_ctor, *, proj_dim, device, batch,
                            save_dir, exp_name, model_id, use_half_precision, tracked,
                            require_cuda_projector=True):
    """One GPT-2's unweighted TRAK pieces: :func:`_single_model_pieces_text` with the
    causal-LM output, gradients over the ``tracked`` modules and ``(input_ids,)`` batches,
    read in eval mode (dropout off) like every attributor on this bank."""
    from trak import TRAKer

    from corrected_lds.tda.tracin import _traced_params

    model = model_ctor().to(device).eval()
    traced = _traced_params(model, tracked)
    n_train = int(data["meta"]["n_train"])
    Q = int(data["meta"]["n_targets"])
    os.makedirs(save_dir, exist_ok=True)
    projector = None                    # CUDA: traker picks fast_jl (asserted below)
    if torch.device(device).type != "cuda" and not require_cuda_projector:
        projector = _basic_projector(sum(v.numel() for v in traced.values()), proj_dim,
                                     device, torch.float16 if use_half_precision
                                     else torch.float32)
    traker = TRAKer(model=model, task=_lm_model_output(), train_set_size=n_train,
                    save_dir=save_dir, device=device, proj_dim=int(proj_dim), lambda_reg=0.0,
                    gradient_computer=_tracked_gradient_computer(tracked),
                    grad_wrt=list(traced), projector=projector,
                    use_half_precision=bool(use_half_precision), projector_seed=0,
                    logging_level=30)
    if require_cuda_projector:                    # traker silently picks BasicProjector otherwise
        _assert_cuda_projector(traker, proj_dim, "trak-lm")
    tr, qy = data["lm_ids"], data["q_lm_ids"]
    try:
        traker.load_checkpoint(state_dict, model_id=int(model_id))
        for i in range(0, n_train, batch):
            traker.featurize(batch=(tr[i:i + batch].contiguous(),),
                             num_samples=int(min(batch, n_train - i)))
        traker.finalize_features()
        traker.start_scoring_checkpoint(exp_name=exp_name, checkpoint=state_dict,
                                        model_id=int(model_id), num_targets=Q)
        for i in range(0, Q, batch):
            traker.score(batch=(qy[i:i + batch].contiguous(),),
                         num_samples=int(min(batch, Q - i)))
        return _finalize_unweighted(traker, exp_name, int(model_id), device)
    except (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError)) as e:
        if require_cuda_projector and _is_projector_resource_error(e):
            raise RuntimeError(
                f"[trak-lm] CUDA projector refused the launch config at k={proj_dim} and "
                f"require_cuda_projector=True — refusing the BasicProjector fallback. "
                f"Original: {str(e).splitlines()[0][:120]}") from e
        raise


# -- per-member pieces cache (multi-GPU featurization) ---------------------- #
def _check_members(members, n_models, where):
    """``members`` -> a validated set of member indices inside ``range(n_models)``."""
    ms = sorted({int(m) for m in members})
    bad = [m for m in ms if not 0 <= m < n_models]
    if not ms or bad:
        raise ValueError(f"[trak] {where}: members {list(members)!r} must be a non-empty "
                         f"subset of 0..{n_models - 1} (bad: {bad}).")
    return set(ms)


def pieces_path(work_dir, mid) -> str:
    """Where one member's unweighted pieces are cached: ``<work_dir>/pieces/m<mid>.npz``."""
    return os.path.join(str(work_dir), "pieces", f"m{int(mid):03d}.npz")


def _save_pieces(path, scores_m, q_m, mid, prov_json):
    """Atomically cache one member's ``(scores_m, q_m)`` in f64, as the accumulator sums them
    (so cached and live members agree bit for bit), stamped with provenance + traker version."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", scores=np.asarray(scores_m, dtype=np.float64),
             q=np.asarray(q_m, dtype=np.float64), mid=int(mid), provenance=prov_json,
             traker_version=TRAKER_VERSION)
    os.replace(path + ".tmp.npz", path)


def _load_pieces(path, mid, prov_json):
    """The cached pieces at ``path``, or ``None`` if absent.  A file whose member index,
    provenance or traker version differs is refused (never recomputed): it is another run's."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    want = {"scores", "q", "mid", "provenance", "traker_version"}
    stamp = (int(z["mid"]), str(z["provenance"]), str(z["traker_version"])) \
        if want <= set(z.files) else None
    if stamp != (int(mid), prov_json, TRAKER_VERSION):
        raise RuntimeError(
            f"[trak] {path} is a pieces cache from a DIFFERENT run (member/provenance/"
            f"traker-version mismatch); refusing to mix. Delete it / use a fresh tag.")
    return z["scores"].astype(np.float64), z["q"].astype(np.float64)


# -- the separated-average (Eq. 15) accumulator ----------------------------- #
def _resolve_work_dirs(work_dir, scratch_dir):
    scratch_dir = scratch_dir or work_dir
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    return work_dir, scratch_dir


_AGGREGATIONS = ("separated-avg-eq15", "plain-mean")


def _ensemble_accumulate(ckpt_paths, per_model_fn, *, work_dir, scratch_dir, prov_json,
                         device, verbose, save_every=1,
                         aggregations=("separated-avg-eq15",), unit_size=None,
                         subsets=None, member_slices=False, snapshot_at=(),
                         members=None, member_ids=None, pieces_dir=None) -> dict:
    """Shared ensemble accumulator for every modality and both estimators.

    ``per_model_fn(state_dict, model_id, mdir) -> (scores_m (n,Q), q_m (n,))`` gives one
    member's unweighted pieces; this keeps the running sums in an atomic accumulator (saved
    every ``save_every`` members and after the last) and returns ``{aggregation: tauq (n, Q)}``:

    * ``"separated-avg-eq15"`` (paper Eq. 15, the TRAK default):
      ``(sum_m scores_m / M) * (sum_m q_m / M)[:, None]``;
    * ``"plain-mean"`` (ablation): ``sum_m (scores_m * q_m[:, None]) / M``, which differs by
      ``cov_m(scores_m, q_m)``; requesting both adds one (n, Q) f64 sum, not a pass.

    The same pass adds pool_tda slices: with ``subsets (M, n)``, ``unit_size`` adds
    ``"g_units" (n_units, M, Q)``, each complete unit's Eq. 15 tauq applied to the masks
    (``models_per_replica = unit_size``), and ``member_slices`` adds ``"g_members_pm"``
    ``(count, M, Q)``, ``subsets @ (scores_m * q_m[:, None])`` per member.  ``snapshot_at`` adds
    ``"snapshots": {E: {aggregation: tauq}}`` at counts E in ``(0, n)``: members are
    seed-determined and taken in index order, so each E-prefix equals a fresh ``--n-models E``
    run.  None of these can be rebuilt mid-run, so a resume missing one is refused, as is one
    with another provenance, traker version, ``unit_size``, mask draw or member block.

    ``members`` switches to pieces-only mode: cache those members' pieces (:func:`pieces_path`),
    touch no sum, return ``{"pieces": ids}``.  After one such process per GPU on disjoint subsets,
    an ordinary pass loads the valid cached pieces, matching a sequential pass bit for bit.
    ``member_ids`` (default ``range(len(ckpt_paths))``) holds each checkpoint's global roster
    index, used as projector ``model_id``, augmentation seed and cache key in ``pieces_dir``
    (default ``work_dir``), so a block of the roster yields slices that concatenate onto a
    released file (``concat_pool_tda``); count, units and snapshots stay local to the block."""
    bad = [a for a in aggregations if a not in _AGGREGATIONS]
    if bad or not aggregations:
        raise ValueError(f"unknown aggregation(s) {bad or aggregations}; "
                         f"known: {_AGGREGATIONS}")
    want_prod = "plain-mean" in aggregations
    track_units = unit_size is not None
    n = len(ckpt_paths)
    if members is not None and member_ids is not None:
        raise ValueError("[trak] members (pieces-only) indexes the full roster; it does "
                         "not combine with member_ids (a block).")
    member_ids = list(range(n)) if member_ids is None else [int(m) for m in member_ids]
    if len(member_ids) != n or (n and (member_ids[0] < 0
                                       or bool(np.any(np.diff(member_ids) <= 0)))):
        raise ValueError(f"[trak] member_ids must give one non-negative, strictly "
                         f"increasing roster index per checkpoint ({len(member_ids)} for "
                         f"{n}): slice order IS replica order.")
    pieces_dir = work_dir if pieces_dir is None else pieces_dir
    if members is not None:                               # pieces-only mode
        todo = _check_members(members, n, "pieces")
        cached = []
        for mid in sorted(todo):
            path = pieces_path(work_dir, mid)
            if _load_pieces(path, mid, prov_json) is None:
                sd = torch.load(ckpt_paths[mid], map_location=device, weights_only=True)
                mdir = os.path.join(scratch_dir, f"m{mid:03d}")
                scores_m, q_m = per_model_fn(sd, mid, mdir)
                shutil.rmtree(mdir, ignore_errors=True)
                del sd
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _save_pieces(path, scores_m, q_m, mid, prov_json)
                if verbose:
                    print(f"    [trak] pieces cached for model {mid + 1}/{n} (mean "
                          f"|score| {np.abs(scores_m).mean():.2e}, mean q "
                          f"{q_m.mean():.3f})", flush=True)
            elif verbose:
                print(f"    [trak] pieces for model {mid + 1}/{n} already cached "
                      f"(provenance ok) -> skip", flush=True)
            cached.append(mid)
        return {"pieces": cached}
    snapshot_at = tuple(sorted({int(e) for e in snapshot_at}))
    if snapshot_at and not (0 < snapshot_at[0] and snapshot_at[-1] < n):
        raise ValueError(f"[trak] snapshot_at={snapshot_at} must lie strictly inside "
                         f"(0, {n}) — the final count is the run itself, not a snapshot.")
    if track_units:
        unit_size = int(unit_size)
        if unit_size < 1 or n % unit_size != 0:
            raise ValueError(f"[trak] unit_size={unit_size} must be >= 1 and divide "
                             f"the member count {n} (the final unit must be "
                             f"complete).")
        if "separated-avg-eq15" not in aggregations:
            raise ValueError("[trak] unit_size tracks Eq. 15 units; add "
                             "'separated-avg-eq15' to aggregations.")
    if (track_units or member_slices) and subsets is None:
        raise ValueError("[trak] pool_tda slices need the pool masks (subsets).")
    S64 = None
    if subsets is not None:
        from corrected_lds.banks.artifacts import subsets_digest
        S64 = np.asarray(subsets, dtype=np.float64)
        masks_sha = subsets_digest(subsets)
    acc_path = os.path.join(work_dir, "tauq_acc.npz")
    score_sum, q_sum, prod_sum, count = None, None, None, 0
    g_units, us_score, us_q, g_members, snaps = [], None, None, [], {}
    if os.path.exists(acc_path):
        z = np.load(acc_path, allow_pickle=False)
        if "score_sum" not in z.files:
            raise RuntimeError(f"{acc_path} is an incompatible TRAK accumulator "
                               f"(no 'score_sum'); delete it before re-running.")
        z_prov = str(z["provenance"]) if "provenance" in z.files else "{}"
        z_ver = str(z["traker_version"]) if "traker_version" in z.files else "unknown"
        if z_prov != prov_json or z_ver != TRAKER_VERSION:
            raise RuntimeError(
                f"{acc_path} provenance/traker-version mismatch (accumulator: {z_ver}/"
                f"{z_prov}; run: {TRAKER_VERSION}/{prov_json}). Refusing to mix. Delete "
                f"it / use a fresh tag.")
        z_mids = (z["member_ids"].tolist() if "member_ids" in z.files
                  else list(range(int(z["count"]))))
        if z_mids != member_ids[:len(z_mids)]:
            raise RuntimeError(
                f"{acc_path} accumulated a DIFFERENT member block ({z_mids[:2]}..."
                f"{z_mids[-1:]}); this run wants {member_ids[:2]}...{member_ids[-1:]}. "
                f"Refusing to mix blocks -- delete it or use a fresh work dir.")
        if want_prod and "prod_sum" not in z.files:
            raise RuntimeError(
                f"{acc_path} was accumulated WITHOUT the plain-mean prod_sum; sums for "
                f"a requested aggregation cannot be reconstructed mid-run — delete it "
                f"(or drop plain-mean).")
        count = int(z["count"])
        if track_units:
            if "g_units" not in z.files:
                raise RuntimeError(
                    f"{acc_path} was accumulated WITHOUT unit tracking; slices for the "
                    f"already-scored members cannot be reconstructed — delete it (or "
                    f"drop unit_size).")
            if int(z["unit_size"]) != unit_size or str(z["subsets_sha16"]) != masks_sha:
                raise RuntimeError(
                    f"{acc_path} tracked units under unit_size={int(z['unit_size'])} / "
                    f"masks {z['subsets_sha16']!s}; this run wants {unit_size} / "
                    f"{masks_sha}. Refusing to mix.")
            g_units = [u.astype(np.float64) for u in z["g_units"]]
            if count % unit_size != 0:
                if "unit_score_sum" not in z.files:
                    raise RuntimeError(f"{acc_path} stops mid-unit without the unit "
                                       f"sums; delete it before re-running.")
                us_score = z["unit_score_sum"].astype(np.float64)
                us_q = z["unit_q_sum"].astype(np.float64)
        if member_slices:
            if "g_members_pm" not in z.files:
                raise RuntimeError(
                    f"{acc_path} was accumulated WITHOUT member slices; slices for the "
                    f"already-scored members cannot be reconstructed — delete it (or "
                    f"drop member_slices).")
            if str(z["subsets_sha16"]) != masks_sha:
                raise RuntimeError(f"{acc_path} member slices bind masks "
                                   f"{z['subsets_sha16']!s}, not {masks_sha}.")
            g_members = [u.astype(np.float64) for u in z["g_members_pm"]]
        score_sum = z["score_sum"].astype(np.float64)
        q_sum = z["q_sum"].astype(np.float64)
        prod_sum = z["prod_sum"].astype(np.float64) if "prod_sum" in z.files else None
        for e in snapshot_at:
            if e > count:
                continue                       # not reached yet — taken later this run
            got = {a: z[f"snap_{e}__{a}"].astype(np.float64)
                   for a in aggregations if f"snap_{e}__{a}" in z.files}
            if len(got) != len(aggregations):
                raise RuntimeError(
                    f"{acc_path} already passed member {e} WITHOUT the requested "
                    f"snapshot (have {sorted(got)}); a prefix state cannot be "
                    f"reconstructed — delete it (or drop snapshot_at={e}).")
            snaps[e] = got
        if verbose:
            print(f"    [trak] resuming ensemble accumulator from {count}/{n} models",
                  flush=True)

    for i in range(count, n):
        mid = member_ids[i]              # global id: projector seed, augmentation, cache key
        cached = _load_pieces(pieces_path(pieces_dir, mid), mid, prov_json)
        if cached is not None:                            # multi-GPU pre-featurized
            scores_m, q_m = cached
            if verbose:
                print(f"    [trak] model {i + 1}/{n} (id {mid}): cached pieces "
                      f"(provenance ok)", flush=True)
        else:
            sd = torch.load(ckpt_paths[i], map_location=device, weights_only=True)
            mdir = os.path.join(scratch_dir, f"m{mid:03d}")   # big per-model store -> scratch
            scores_m, q_m = per_model_fn(sd, mid, mdir)
            shutil.rmtree(mdir, ignore_errors=True)           # bound scratch to one model
            del sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score_sum = scores_m if score_sum is None else score_sum + scores_m
        q_sum = q_m if q_sum is None else q_sum + q_m
        if want_prod:
            pm = scores_m * q_m[:, None]
            prod_sum = pm if prod_sum is None else prod_sum + pm
        if member_slices:
            g_members.append(S64 @ (scores_m * q_m[:, None]))
        if track_units:
            us_score = scores_m if us_score is None else us_score + scores_m
            us_q = q_m if us_q is None else us_q + q_m
            if (i + 1) % unit_size == 0:                  # unit complete -> one slice
                tau_u = (us_score / unit_size) * (us_q / unit_size)[:, None]
                g_units.append(S64 @ tau_u)
                us_score, us_q = None, None
        count = i + 1                                     # local progress in the block
        if count in snapshot_at:                          # the E-prefix is a fresh E-run
            snaps[count] = {}
            if "separated-avg-eq15" in aggregations:
                snaps[count]["separated-avg-eq15"] = \
                    (score_sum / count) * (q_sum / count)[:, None]
            if want_prod:
                snaps[count]["plain-mean"] = prod_sum / count
            if verbose:
                print(f"    [trak] prefix snapshot taken at n_models={count}",
                      flush=True)
        if count % int(save_every) == 0 or count == n:
            Qd = score_sum.shape[1]
            extra = {"prod_sum": prod_sum} if want_prod else {}
            if track_units:
                extra["g_units"] = (np.asarray(g_units) if g_units
                                    else np.zeros((0, S64.shape[0], Qd)))
                extra["unit_size"] = unit_size
                if us_score is not None:                  # mid-unit state (f32 on disk)
                    extra["unit_score_sum"] = us_score.astype(np.float32)
                    extra["unit_q_sum"] = us_q.astype(np.float32)
            if member_slices:
                extra["g_members_pm"] = (np.asarray(g_members) if g_members
                                         else np.zeros((0, S64.shape[0], Qd)))
            if track_units or member_slices:
                extra["subsets_sha16"] = masks_sha
            for e, per_agg in snaps.items():              # persist across resumes (f32)
                for a, t in per_agg.items():
                    extra[f"snap_{e}__{a}"] = np.asarray(t, dtype=np.float32)
            np.savez(acc_path + ".tmp.npz", score_sum=score_sum, q_sum=q_sum,
                     count=count, provenance=prov_json, traker_version=TRAKER_VERSION,
                     member_ids=np.asarray(member_ids, dtype=np.int64), **extra)
            os.replace(acc_path + ".tmp.npz", acc_path)   # atomic: temp then rename
        if verbose:
            print(f"    [trak] model {i + 1}/{n} (id {mid}) scored (mean |score| "
                  f"{np.abs(scores_m).mean():.2e}, mean q {q_m.mean():.3f})", flush=True)
    out = {}
    if "separated-avg-eq15" in aggregations:
        out["separated-avg-eq15"] = (score_sum / count) * (q_sum / count)[:, None]
    if want_prod:
        out["plain-mean"] = prod_sum / count
    if track_units:
        out["g_units"] = np.asarray(g_units)
    if member_slices:
        out["g_members_pm"] = np.asarray(g_members)
    if snapshot_at:
        out["snapshots"] = snaps
    return out


def compute_tauq(ckpt_paths, data, model_ctor, *, proj_dim=2_048, device="cuda",
                 batch=256, work_dir, scratch_dir=None,
                 exp_name="lds", force_basic_projector=False, proj_max_batch_size=8,
                 provenance=None, verbose=False, require_cuda_projector=False,
                 featurize_augment=None, use_half_precision=False, train_batches=None,
                 save_every=1, aggregation="separated-avg-eq15", unit_size=None,
                 subsets=None, member_slices=False, snapshot_at=(), members=None,
                 member_ids=None, pieces_dir=None):
    """Ensembled TRAK scores for the image path.

    ``work_dir`` (small, persistent) holds the resumable accumulator, stamped with ``provenance``
    and checked on resume; ``scratch_dir`` (large, ephemeral) holds one model's feature store.
    ``use_half_precision`` is tauq identity (default off; traker's is fp16), ``train_batches``
    the streaming featurize source (imagenet) and ``save_every`` paces the accumulator writes.
    A string ``aggregation`` returns that ``tauq (n_train, Q)``; a sequence (or any of
    ``unit_size``, ``member_slices``, ``snapshot_at``, ``members``) returns
    :func:`_ensemble_accumulate`'s dict."""
    work_dir, scratch_dir = _resolve_work_dirs(work_dir, scratch_dir)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    single = (isinstance(aggregation, str) and unit_size is None
              and not member_slices and not snapshot_at and members is None)
    aggs = (aggregation,) if isinstance(aggregation, str) else tuple(aggregation)

    def per_model(sd, mid, mdir):
        return _single_model_pieces(
            sd, data, model_ctor, proj_dim=proj_dim, device=device, batch=batch,
            save_dir=mdir, exp_name=exp_name,
            force_basic_projector=force_basic_projector,
            proj_max_batch_size=proj_max_batch_size, model_id=mid, verbose=verbose,
            require_cuda_projector=require_cuda_projector,
            featurize_augment=featurize_augment, use_half_precision=use_half_precision,
            train_batches=train_batches)

    out = _ensemble_accumulate(ckpt_paths, per_model, work_dir=work_dir,
                               scratch_dir=scratch_dir, prov_json=prov_json,
                               device=device, verbose=verbose, save_every=save_every,
                               aggregations=aggs, unit_size=unit_size, subsets=subsets,
                               member_slices=member_slices, snapshot_at=snapshot_at,
                               members=members, member_ids=member_ids,
                               pieces_dir=pieces_dir)
    return out[aggregation] if single else out


def compute_tauq_text(ckpt_paths, data, model_ctor, *, proj_dim=4_096, device="cuda",
                      batch=32, work_dir, scratch_dir=None, exp_name="lds_text",
                      use_half_precision=False, provenance=None, verbose=False,
                      require_cuda_projector=True, save_every=1,
                      aggregation="separated-avg-eq15", unit_size=None, subsets=None,
                      member_slices=False, snapshot_at=(), members=None,
                      member_ids=None, pieces_dir=None):
    """Ensembled TRAK scores for the text (BERT/QNLI) path, with the estimators, slices,
    resumable accumulator and pieces-only ``members`` mode of :func:`compute_tauq`."""
    work_dir, scratch_dir = _resolve_work_dirs(work_dir, scratch_dir)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    single = (isinstance(aggregation, str) and unit_size is None
              and not member_slices and not snapshot_at and members is None)
    aggs = (aggregation,) if isinstance(aggregation, str) else tuple(aggregation)

    def per_model(sd, mid, mdir):
        return _single_model_pieces_text(
            sd, data, model_ctor, proj_dim=proj_dim, device=device, batch=batch,
            save_dir=mdir, exp_name=exp_name, model_id=mid,
            use_half_precision=use_half_precision,
            require_cuda_projector=require_cuda_projector)

    out = _ensemble_accumulate(ckpt_paths, per_model, work_dir=work_dir,
                               scratch_dir=scratch_dir, prov_json=prov_json,
                               device=device, verbose=verbose, save_every=save_every,
                               aggregations=aggs, unit_size=unit_size, subsets=subsets,
                               member_slices=member_slices, snapshot_at=snapshot_at,
                               members=members, member_ids=member_ids,
                               pieces_dir=pieces_dir)
    return out[aggregation] if single else out


def compute_tauq_lm(ckpt_paths, data, model_ctor, *, tracked, proj_dim=1_024,
                    device="cuda", batch=8, work_dir, scratch_dir=None,
                    exp_name="lds_lm", use_half_precision=False, provenance=None,
                    verbose=False, require_cuda_projector=True, save_every=1,
                    aggregation="separated-avg-eq15", unit_size=None, subsets=None,
                    member_slices=False, snapshot_at=(), members=None,
                    member_ids=None, pieces_dir=None):
    """Ensembled TRAK scores for the causal-LM (GPT-2/WikiText) path: as
    :func:`compute_tauq_text`, with gradients over the ``tracked`` modules only."""
    work_dir, scratch_dir = _resolve_work_dirs(work_dir, scratch_dir)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    single = (isinstance(aggregation, str) and unit_size is None
              and not member_slices and not snapshot_at and members is None)
    aggs = (aggregation,) if isinstance(aggregation, str) else tuple(aggregation)

    def per_model(sd, mid, mdir):
        return _single_model_pieces_lm(
            sd, data, model_ctor, proj_dim=proj_dim, device=device, batch=batch,
            save_dir=mdir, exp_name=exp_name, model_id=mid,
            use_half_precision=use_half_precision, tracked=tracked,
            require_cuda_projector=require_cuda_projector)

    out = _ensemble_accumulate(ckpt_paths, per_model, work_dir=work_dir,
                               scratch_dir=scratch_dir, prov_json=prov_json,
                               device=device, verbose=verbose, save_every=save_every,
                               aggregations=aggs, unit_size=unit_size, subsets=subsets,
                               member_slices=member_slices, snapshot_at=snapshot_at,
                               members=members, member_ids=member_ids,
                               pieces_dir=pieces_dir)
    return out[aggregation] if single else out
