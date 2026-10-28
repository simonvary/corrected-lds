"""EK-FAC influence functions as the second TDA method, via Bae et al.'s ``kronfluence``.

``compute_tauq_ekfac`` returns ``trak.compute_tauq``'s ``tauq (n_train, n_targets)``
contract.  Members are combined by a plain mean, not TRAK's separated-average product, so
there is no finite-ensemble product bias and the 1/E noise law holds exactly.  Covers images,
BERT text and a causal LM (GPT-2 on WikiText, measured by the query loss as in Bae et al.).

Notes, each verified against kronfluence 1.0.1:

1. ``load_pairwise_scores(...)['all_modules']`` is ``(num_query, num_train)``: we store ``.T``.
2. ``compute_measurement`` returns ``-margin`` (as kronfluence's reference tasks do) and an IF
   score estimates the effect of removing i; the negations cancel, so ``S @ tauq``
   correlates positively with the pool margins and no flip is applied.
3. Only Linear/Conv2d are tracked, so BatchNorm affine parameters are excluded (unlike
   full-gradient TRAK): a methodological difference, recorded in provenance.
4. ``prepare_model`` is destructive: load the checkpoint first (after -> RecursionError) and
   never reuse the prepared model; ``_ensure_container`` wraps a bare Linear/Conv2d root.
5. ``use_empirical_fisher=False`` (library and paper default) samples labels from the model,
   a second randomness channel, so ``factor_seed`` is seeded per member and part of provenance.

Damping is a bias knob that seed-averaging never removes: ``None`` -> kronfluence's
``0.1 * mean eigenvalue`` heuristic (the Grosse et al. convention); the library's ``1e-8``
default is effectively no damping.  Pre-register it: choosing it by the LDS fits the metric.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
import torch

from corrected_lds.pools.train import per_query_margin

try:
    from importlib.metadata import version as _pkg_version
    KRONFLUENCE_VERSION = _pkg_version("kronfluence")
except Exception:  # noqa: BLE001 — any metadata failure means "unknown"
    KRONFLUENCE_VERSION = "unknown"

__all__ = [
    "KRONFLUENCE_VERSION",
    "EkfacArgs",
    "combine_if_partials",
    "compute_tauq_ekfac",
    "partial_path",
]

_STRATEGIES = ("ekfac", "kfac", "diagonal", "identity")


@dataclass(frozen=True)
class EkfacArgs:
    """Everything that changes ``tauq`` on the EK-FAC side.  ``to_dict()`` is the
    provenance payload, so it must stay JSON-native and round-trip stable."""

    strategy: str = "ekfac"                  # ekfac | kfac | diagonal | identity
    damping: float | None = None             # None -> kronfluence's 0.1*mean-eigenvalue heuristic
    use_empirical_fisher: bool = False       # False -> sampled-label (true) Fisher
    factor_seed: int = 0                     # member m fits factors under factor_seed + m
    label_smoothing: float = 0.0             # the train loss whose influence is computed
    covariance_max_examples: int | None = None    # None -> whole train set
    lambda_max_examples: int | None = None
    tracked: tuple[str, ...] | None = None   # None -> all supported Linear/Conv2d
    # SVD rank of the query gradients (None -> exact); identity, as it changes tau.  Exact
    # scoring re-walks the train set per query batch (kronfluence score/pairwise.py:223) and
    # a bert-base query gradient is 327 MB, so compressing lets the batch grow.
    low_rank: int | None = None
    # --- pure chunking, not provenance identity ---
    factor_batch: int | None = 256
    query_batch: int = 500
    train_batch: int | None = 256

    def __post_init__(self):
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {self.strategy!r}")
        if self.damping is not None and self.damping < 0:
            raise ValueError(f"damping must be >= 0 or None (heuristic), got {self.damping}")
        if self.low_rank is not None and self.low_rank < 1:
            raise ValueError(f"low_rank must be >= 1 or None (exact), got {self.low_rank}")

    def to_dict(self) -> dict:
        """JSON-native provenance payload, compared after a json round trip, so ``tracked``
        is a list.  Batch sizes are excluded (pure chunking)."""
        return {
            "strategy": str(self.strategy),
            "damping": None if self.damping is None else float(self.damping),
            "use_empirical_fisher": bool(self.use_empirical_fisher),
            "factor_seed": int(self.factor_seed),
            "label_smoothing": float(self.label_smoothing),
            "covariance_max_examples": (None if self.covariance_max_examples is None
                                        else int(self.covariance_max_examples)),
            "lambda_max_examples": (None if self.lambda_max_examples is None
                                    else int(self.lambda_max_examples)),
            "low_rank": None if self.low_rank is None else int(self.low_rank),
            "tracked": None if self.tracked is None else [str(t) for t in self.tracked],
        }


# -- kronfluence glue (imported lazily) ------------------------------------- #
def _make_task(args: EkfacArgs):
    """The ``kronfluence.Task`` with the paper's margin measurement.

    Both hooks use ``reduction='sum'`` because kronfluence sums over the batch itself.  The
    measurement is exactly the pool readout (``per_query_margin``), negated per note 2."""
    import torch.nn.functional as F
    from kronfluence.task import Task

    ls = float(args.label_smoothing)
    tracked = None if args.tracked is None else [str(t) for t in args.tracked]

    class _MarginTask(Task):
        """Image and text share every line but the batch shape, so the estimator cannot differ.

        A batch is ``(*inputs, y)``: ``(x, y)`` for images, ``(input_ids, token_type_ids,
        attention_mask, y)`` for BERT (the 4-tuple ``tracin.py`` builds)."""

        def compute_train_loss(self, batch, model, sample=False):
            *inputs, y = batch
            logits = model(*inputs)
            if not sample:
                return F.cross_entropy(logits, y, reduction="sum", label_smoothing=ls)
            with torch.no_grad():                       # true Fisher: labels ~ model's own p
                probs = torch.softmax(logits.detach(), dim=-1)
                y = torch.multinomial(probs, num_samples=1).flatten()
            return F.cross_entropy(logits, y, reduction="sum", label_smoothing=ls)

        def compute_measurement(self, batch, model):
            *inputs, y = batch
            return -per_query_margin(model(*inputs), y).sum()

        def get_attention_mask(self, batch):
            """Padding must not enter the Fisher: kronfluence masks it out of the activation
            covariance, a biased module's ones-column and the token count
            (``module/linear.py:34-44``).  Images have no mask."""
            return batch[2] if len(batch) == 4 else None

        def get_influence_tracked_modules(self):
            return tracked

    return _MarginTask()


def _make_lm_task(args: EkfacArgs):
    """The causal-LM ``Task``: next-token cross-entropy summed over tokens and batch.

    A batch is ``(input_ids,)``; the targets are the inputs, shifted.  The measurement is the
    query loss, i.e. ``-margin`` under the pool's ``margins = -loss``, so note 2 carries over:
    a positive score means including i lowers the query loss.  ``sample=True`` draws targets
    from the model (the true Fisher); full 512-token chunks need no padding or mask."""
    import torch.nn.functional as F
    from kronfluence.task import Task

    tracked = None if args.tracked is None else [str(t) for t in args.tracked]

    def _shifted(batch, model):
        (ids,) = batch
        logits = model(ids)[:, :-1, :]
        return logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)

    class _LmTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            logits, y = _shifted(batch, model)
            if sample:
                with torch.no_grad():
                    probs = torch.softmax(logits.detach(), dim=-1)
                    y = torch.multinomial(probs, num_samples=1).flatten()
            return F.cross_entropy(logits, y, reduction="sum")

        def compute_measurement(self, batch, model):
            logits, y = _shifted(batch, model)
            return F.cross_entropy(logits, y, reduction="sum")

        def get_influence_tracked_modules(self):
            return tracked

    return _LmTask()


def _ensure_container(model):
    """kronfluence replaces tracked modules inside their parent, so a bare Linear/Conv2d
    root cannot be wrapped (and recurses forever).  Wrap it."""
    from torch import nn
    if isinstance(model, (nn.Linear, nn.Conv2d)):
        return nn.Sequential(model)
    return model


def _tensor_dataset(*cols):
    from torch.utils.data import TensorDataset
    return TensorDataset(*[torch.as_tensor(c).detach().cpu() for c in cols[:-1]],
                         torch.as_tensor(cols[-1]).detach().cpu().long())


def _datasets(data):
    """``(train_ds, query_ds)`` for either modality.

    ``"input_ids" in data`` is ``tracin.py``'s text test, so all attributors agree on a bank's
    modality; ``"lm_ids"`` is the causal-LM twin (``load_wikitext2``), token ids only."""
    from torch.utils.data import TensorDataset
    if "lm_ids" in data:
        return (TensorDataset(torch.as_tensor(data["lm_ids"]).detach().cpu().long()),
                TensorDataset(torch.as_tensor(data["q_lm_ids"]).detach().cpu().long()))
    if "input_ids" in data:
        # text labels are "labels"/"q_labels", not the images' "y_train"/"y_query"
        return (_tensor_dataset(data["input_ids"], data["token_type_ids"],
                                data["attention_mask"], data["labels"]),
                _tensor_dataset(data["q_input_ids"], data["q_token_type_ids"],
                                data["q_attention_mask"], data["q_labels"]))
    return (_tensor_dataset(data["X_train"], data["y_train"]),
            _tensor_dataset(data["X_query"], data["y_query"]))


def _single_model_tauq(state_dict, data, model_ctor, *, args, device, member_dir, mid,
                       verbose=False) -> np.ndarray:
    """One checkpoint's ``tauq (n_train, Q)`` float64: fit EK-FAC factors, score all
    (query, train) pairs, transpose.  ``member_dir`` is the caller's to delete."""
    from kronfluence import Analyzer, FactorArguments, ScoreArguments, prepare_model

    task = _make_lm_task(args) if "lm_ids" in data else _make_task(args)
    model = model_ctor()
    model.load_state_dict(state_dict)                 # before prepare_model (note 4)
    model = _ensure_container(model).to(device)
    model = prepare_model(model, task)

    train_ds, query_ds = _datasets(data)
    on_cpu = torch.device(device).type == "cpu"

    analyzer = Analyzer(analysis_name=f"m{mid:03d}", model=model, task=task, cpu=on_cpu,
                        output_dir=member_dir, disable_tqdm=True, log_level=40,
                        disable_model_save=True)
    # Pass every value explicitly, even None (= "entire dataset" / the damping heuristic):
    # omitting one silently swaps in kronfluence's defaults, so provenance would be wrong.
    factor_args = FactorArguments(
        strategy=args.strategy,
        use_empirical_fisher=bool(args.use_empirical_fisher),
        covariance_max_examples=args.covariance_max_examples,
        lambda_max_examples=args.lambda_max_examples,
    )
    # seed the sampled-label Fisher draw (note 5) per member so a rerun reproduces tau
    torch.manual_seed(int(args.factor_seed) + int(mid))
    analyzer.fit_all_factors(factors_name="f", dataset=train_ds,
                             per_device_batch_size=args.factor_batch,
                             factor_args=factor_args, overwrite_output_dir=True)

    score_args = ScoreArguments(damping_factor=args.damping,      # None -> heuristic
                                query_gradient_low_rank=args.low_rank)   # None -> exact
    analyzer.compute_pairwise_scores(
        scores_name="s", factors_name="f", query_dataset=query_ds, train_dataset=train_ds,
        per_device_query_batch_size=int(args.query_batch),
        per_device_train_batch_size=args.train_batch,
        score_args=score_args, overwrite_output_dir=True)
    scores = analyzer.load_pairwise_scores("s")["all_modules"]        # (Q, n_train)
    tauq = scores.to(torch.float64).cpu().numpy().T                   # -> (n_train, Q)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(f"    [ekfac] member {mid} scored: tauq {tauq.shape}, "
              f"mean |tau| {np.abs(tauq).mean():.3e}", flush=True)
    return tauq


# -- per-member partials: kept, recombinable over an arbitrary member set --- #
def partial_path(part_dir, mid) -> str:
    return os.path.join(part_dir, f"tauq_if_m_{int(mid):03d}.npz")


def _file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _partial_valid(path, *, mid, ckpt_md5, prov_json, args_json) -> bool:
    """A partial is reusable only if its member index, checkpoint content, EK-FAC arguments
    and provenance all match.  ``args_json`` is checked separately from ``prov_json`` so the
    cache identity does not depend on the caller passing provenance."""
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001 — a truncated write is simply "not reusable"
        return False
    if "ekfac_args" not in z.files:
        return False
    return (int(z["mid"]) == int(mid)
            and str(z["ckpt_md5"]) == ckpt_md5
            and _member_provenance(str(z["provenance"])) == _member_provenance(prov_json)
            and str(z["ekfac_args"]) == args_json
            and str(z["kronfluence_version"]) == KRONFLUENCE_VERSION)


def _member_provenance(prov_json):
    """The run provenance minus ``n_models`` (each checkpoint is pinned by ``ckpt_md5``), so
    partials stay reusable when the same ensemble grows, e.g. from E=5 to E=10."""
    try:
        prov = json.loads(prov_json)
    except ValueError:
        return None
    if isinstance(prov, dict):
        prov.pop("n_models", None)
    return prov


def combine_if_partials(part_dir, members) -> np.ndarray:
    """Plain mean of the per-member ``tauq`` partials over any member ids, not just a prefix,
    so disjoint replicates need no re-scoring.  Returns (n_train, Q) float64."""
    members = [int(m) for m in members]
    if not members:
        raise ValueError("combine_if_partials: `members` is empty")
    if len(set(members)) != len(members):
        raise ValueError(f"combine_if_partials: duplicate members in {members}")
    total = None
    for mid in members:
        p = partial_path(part_dir, mid)
        if not os.path.exists(p):
            raise FileNotFoundError(f"combine_if_partials: missing partial for member "
                                    f"{mid}: {p}")
        t = np.load(p, allow_pickle=False)["tauq"].astype(np.float64)
        total = t if total is None else total + t
    return total / len(members)


# -- the public entry point ------------------------------------------------- #
def compute_tauq_ekfac(ckpt_paths, data, model_ctor, *, args=None, device="cuda",
                       work_dir=None, provenance=None, verbose=False, members=None):
    """Seed-averaged EK-FAC influence ``tauq (n_train, Q)`` from ``ckpt_paths``.

    Resumable per member: a partial matching its checkpoint content and the run's provenance
    is reused.  Plain mean over all members (:func:`combine_if_partials` for other subsets).
    ``members`` (indices into ``ckpt_paths``) computes only those partials and returns
    ``None``: one worker of a multi-GPU split, isolated with ``CUDA_VISIBLE_DEVICES`` since
    kronfluence allocates on the current device.  A final call without it only combines."""
    if not ckpt_paths:
        raise ValueError("compute_tauq_ekfac: no checkpoints given")
    args = args or EkfacArgs()
    work_dir = work_dir or os.path.join(os.getcwd(), "ekfac_work")
    os.makedirs(work_dir, exist_ok=True)
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    args_json = json.dumps(args.to_dict(), sort_keys=True)

    todo = range(len(ckpt_paths)) if members is None else sorted({int(m) for m in members})
    bad = [m for m in todo if not 0 <= m < len(ckpt_paths)]
    if bad:
        raise ValueError(f"compute_tauq_ekfac: members {bad} outside 0..{len(ckpt_paths) - 1}")
    for mid in todo:
        ckpt = ckpt_paths[mid]
        path = partial_path(work_dir, mid)
        ckpt_md5 = _file_md5(ckpt)
        if _partial_valid(path, mid=mid, ckpt_md5=ckpt_md5, prov_json=prov_json,
                          args_json=args_json):
            if verbose:
                print(f"    [ekfac] member {mid} partial present (identity ok) -> reuse",
                      flush=True)
            continue
        member_dir = os.path.join(work_dir, f"_kf_m{mid:03d}")
        shutil.rmtree(member_dir, ignore_errors=True)       # never resume a half-written store
        try:
            sd = torch.load(ckpt, map_location=device, weights_only=True)
            tauq_m = _single_model_tauq(sd, data, model_ctor, args=args, device=device,
                                        member_dir=member_dir, mid=mid, verbose=verbose)
            del sd
            tmp = path + ".tmp.npz"                         # atomic: temp then rename
            np.savez(tmp, tauq=tauq_m.astype(np.float32), mid=mid, ckpt_md5=ckpt_md5,
                     provenance=prov_json, kronfluence_version=KRONFLUENCE_VERSION,
                     ekfac_args=args_json)
            os.replace(tmp, path)
        finally:
            shutil.rmtree(member_dir, ignore_errors=True)   # bound disk to one member
    if members is not None:         # one worker of a split: partials only, nothing to mean
        return None
    tauq = combine_if_partials(work_dir, range(len(ckpt_paths)))
    if verbose:
        print(f"[ekfac] tauq {tauq.shape} from {len(ckpt_paths)} member(s) "
              f"(plain mean, strategy={args.strategy}, damping={args.damping})", flush=True)
    return tauq
