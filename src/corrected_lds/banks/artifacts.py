"""Bank contracts: load + validate ``pool.npz`` / ``tauq.npz`` / ``pool_tda_*.npz``, and
pairing-enforced predictions (alignment is checked in code, never assumed from layout).

``pool_tda_<method>.npz`` mirrors ``pool.npz`` for an attributor: ``g[k, j, q]`` is replica
k's prediction on mask j, query q, a replica being one self-contained estimator (trak: one
Eq. 15 separated average over ``e_features`` members; ekfac: one factor-fit member; tracin:
one trajectory).  Replicas combine by plain mean, so any subset is an exact smaller-ensemble
estimate; ``e_features`` is fixed at bake time: masking drops the per-member pieces Eq. 15 needs.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from corrected_lds.banks.masks import sample_subsets
from corrected_lds.banks.presets import DATA_HASHES, bank_hashes

__all__ = [
    "SEED_MODES",
    "Pool",
    "PoolTda",
    "TauScores",
    "combine_replicas",
    "load_pool",
    "load_pool_tda",
    "load_tauq",
    "predictions",
    "predictions_tda",
    "subsets_digest",
    "write_pool_tda",
    "write_tauq",
]

# How a pool's R replicas were seeded.  Not interchangeable: the corrected estimator assumes
# the seed noise in ``margins`` is independent across masks, which holds only for "nested".
#   "nested"              every cell has its own seed (``pools/runner.cell_seed``).
#   "shared-per-replica"  one seed per replica, shared across its masks (bergson's protocol):
#                         its offset c(r) cancels in cross-mask correlations, so the total
#                         variance overstates the floor; correct such a pool with
#                         ``estimator.seed_noise_variance_centred``.
SEED_MODES = ("nested", "shared-per-replica")

_POOL_KEYS = ("margins", "subsets", "M", "R", "alpha", "recipe", "dataset", "seed_mode",
              "subsets_source", "schema", "code_version", "completed_cells", "done_rows",
              "train_idx", "query_idx", "train_hash", "query_hash")
_TAUQ_KEYS = ("tauq", "dataset", "n_train", "n_targets", "train_idx", "query_idx",
              "train_hash", "query_hash", "n_models", "proj_dim", "provenance")


@dataclass(frozen=True)
class Pool:
    """Validated view of a complete ``pool.npz``.  ``margins`` is ``(R, M, Q)`` float64."""

    margins: np.ndarray
    subsets: np.ndarray
    recipe: dict
    dataset: str
    alpha: float
    train_idx: np.ndarray
    query_idx: np.ndarray
    train_hash: str
    query_hash: str
    code_version: str
    path: str
    seed_mode: str = "nested"

    @property
    def R(self) -> int:
        return int(self.margins.shape[0])

    @property
    def M(self) -> int:
        return int(self.margins.shape[1])

    @property
    def Q(self) -> int:
        return int(self.margins.shape[2])

    @property
    def n(self) -> int:
        return int(self.subsets.shape[1])


@dataclass(frozen=True)
class TauScores:
    """Validated view of a ``tauq.npz``.  ``tauq`` is ``(n_train, n_targets)``."""

    tauq: np.ndarray
    dataset: str
    n_train: int
    n_targets: int
    train_idx: np.ndarray
    query_idx: np.ndarray
    train_hash: str
    query_hash: str
    n_models: int
    proj_dim: int
    provenance: dict
    path: str


def _require(z, keys, path, kind):
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise ValueError(f"[{kind}] {path} is missing required key(s) {missing}; this loader "
                         f"reads only schema-3 files written by this pipeline.")


def _check_idx(z, key, length, path, kind):
    """An index draw must be exactly ``(length,)``, non-negative, and duplicate-free —
    a repeated index silently double-counts one example in every downstream product."""
    idx = np.asarray(z[key], dtype=np.int64)
    if idx.shape != (length,):
        raise ValueError(f"[{kind}] {path}: {key} shape {idx.shape} != ({length},).")
    if idx.min(initial=0) < 0 or len(np.unique(idx)) != length:
        raise ValueError(f"[{kind}] {path}: {key} must be non-negative and duplicate-free.")
    return idx


def _check_hash_pins(z, dataset, path, kind):
    """For canonical datasets the stamped side digests must equal the pins, not just exist."""
    if dataset not in DATA_HASHES:
        return                                       # synthetic datasets carry free-form ids
    want_tr, want_qy = bank_hashes(dataset)
    got_tr, got_qy = str(z["train_hash"]), str(z["query_hash"])
    if (got_tr, got_qy) != (want_tr, want_qy):
        raise ValueError(f"[{kind}] {path}: stamped data hashes (train {got_tr}, query "
                         f"{got_qy}) do not match the pinned hashes for {dataset!r} "
                         f"(train {want_tr}, query {want_qy}).")


def load_pool(path) -> Pool:
    """Load + validate a complete pool: refuses missing schema-3 keys, an unknown seed_mode or
    subsets_source, partial or non-finite pools, masks other than the pinned sampler's draw,
    malformed index draws, an ``alpha`` unlike the recipe's, and data hashes unlike the pins."""
    path = str(path)
    z = np.load(path)
    _require(z, _POOL_KEYS, path, "pool")
    if int(z["schema"]) != 3:
        raise ValueError(f"[pool] {path}: schema {int(z['schema'])} != 3.")
    seed_mode = str(z["seed_mode"])
    if seed_mode not in SEED_MODES:
        raise ValueError(f"[pool] {path}: seed_mode {seed_mode!r} not in {sorted(SEED_MODES)}.")
    if str(z["subsets_source"]) != "sample_subsets:777":
        raise ValueError(f"[pool] {path}: subsets_source {str(z['subsets_source'])!r} is not "
                         f"the pinned sampler.")
    margins = z["margins"].astype(np.float64)
    subsets = z["subsets"]
    R, M, Q = margins.shape
    if (M, R) != (int(z["M"]), int(z["R"])) or subsets.shape[0] != M:
        raise ValueError(f"[pool] {path}: stored dims (M={int(z['M'])}, R={int(z['R'])}) do not "
                         f"match arrays margins{margins.shape} subsets{subsets.shape}.")
    if int(z["done_rows"]) != R or int(z["completed_cells"]) != R * M \
            or not np.isfinite(margins).all():
        raise ValueError(
            f"[pool] {path} is not a COMPLETE pool (done_rows={int(z['done_rows'])}/{R}, "
            f"completed_cells={int(z['completed_cells'])}/{R * M}, "
            f"finite={bool(np.isfinite(margins).all())}). The statistics half consumes only "
            f"complete banks; finish the build first.")
    recipe = json.loads(str(z["recipe"]))
    alpha = float(z["alpha"])
    if alpha != float(recipe["alpha"]):
        raise ValueError(f"[pool] {path}: stored alpha {alpha} != recipe alpha "
                         f"{recipe['alpha']}; the metadata is internally inconsistent.")
    if not np.array_equal(subsets, sample_subsets(M, subsets.shape[1], alpha, 777)):
        raise ValueError(f"[pool] {path}: stored masks are NOT the pinned mask sampler's draw "
                         f"for (M={M}, n={subsets.shape[1]}, alpha={alpha}, seed=777); the "
                         f"pool's mask identity is corrupt.")
    dataset = str(z["dataset"])
    train_idx = _check_idx(z, "train_idx", int(subsets.shape[1]), path, "pool")
    query_idx = _check_idx(z, "query_idx", Q, path, "pool")
    _check_hash_pins(z, dataset, path, "pool")
    return Pool(margins=margins, subsets=subsets.astype(np.int8), recipe=recipe,
                dataset=dataset, alpha=alpha, train_idx=train_idx, query_idx=query_idx,
                train_hash=str(z["train_hash"]), query_hash=str(z["query_hash"]),
                code_version=str(z["code_version"]), path=path, seed_mode=seed_mode)


def load_tauq(path) -> TauScores:
    """Load + validate a tauq file: schema keys, a finite ``tauq (n_train, n_targets)``,
    well-formed index draws, and (for canonical datasets) stamped hashes equal to the pins."""
    path = str(path)
    z = np.load(path)
    _require(z, _TAUQ_KEYS, path, "tauq")
    tauq = z["tauq"]
    n_train, n_targets = int(z["n_train"]), int(z["n_targets"])
    if tauq.shape != (n_train, n_targets):
        raise ValueError(f"[tauq] {path}: tauq shape {tauq.shape} != ({n_train}, {n_targets}).")
    if not np.isfinite(tauq).all():
        raise ValueError(f"[tauq] {path}: tauq contains non-finite values.")
    dataset = str(z["dataset"])
    train_idx = _check_idx(z, "train_idx", n_train, path, "tauq")
    query_idx = _check_idx(z, "query_idx", n_targets, path, "tauq")
    _check_hash_pins(z, dataset, path, "tauq")
    return TauScores(tauq=tauq, dataset=dataset, n_train=n_train, n_targets=n_targets,
                     train_idx=train_idx, query_idx=query_idx,
                     train_hash=str(z["train_hash"]), query_hash=str(z["query_hash"]),
                     n_models=int(z["n_models"]), proj_dim=int(z["proj_dim"]),
                     provenance=json.loads(str(z["provenance"])), path=path)


def write_tauq(path, tauq, source, *, provenance, n_models=1, proj_dim=0) -> str:
    """Atomically write a ``tauq_*.npz`` carrying the pairing identity :func:`predictions`
    enforces; torch-free, for any attribution method, in-repo or external.  ``source`` is a
    validated :class:`Pool` (identity copied from the bank the scores target) or a loader
    ``meta`` dict; ``n_models``/``proj_dim`` default to 1 / 0 (not applicable).  End to end::

        pool = load_pool("banks/cifar10/pool.npz")
        scores = ...                                    # (pool.n, pool.Q), your method
        write_tauq("banks/cifar10/tauq_mymethod.npz", scores, pool,
                   provenance={"method": "mymethod", "version": "1.0"})
        evaluate_pool(pool, load_tauq("banks/cifar10/tauq_mymethod.npz"))
    """
    tauq = np.asarray(tauq)
    if isinstance(source, Pool):
        ident = {"dataset": source.dataset, "n_train": source.n, "n_targets": source.Q,
                 "train_idx": source.train_idx, "query_idx": source.query_idx,
                 "train_hash": source.train_hash, "query_hash": source.query_hash}
    else:
        ident = {"dataset": str(source["dataset"]), "n_train": int(source["n_train"]),
                 "n_targets": int(source["n_targets"]),
                 "train_idx": np.asarray(source["train_idx"], dtype=np.int64),
                 "query_idx": np.asarray(source["query_idx"], dtype=np.int64),
                 "train_hash": str(source["train_hash"]),
                 "query_hash": str(source["query_hash"])}
    if tauq.shape != (ident["n_train"], ident["n_targets"]):
        raise ValueError(f"[tauq] scores shape {tauq.shape} != (n_train, n_targets) = "
                         f"({ident['n_train']}, {ident['n_targets']}) of the target bank.")
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    tmp = str(path) + f".tmp.{os.getpid()}"
    np.savez(tmp, tauq=tauq.astype(np.float32), n_models=int(n_models),
             proj_dim=int(proj_dim), provenance=prov_json, **ident)
    os.replace(tmp + ".npz", str(path))
    return str(path)


def predictions(pool: Pool, tau: TauScores) -> np.ndarray:
    """LDS predictions ``g = subsets @ tauq``, ``(M, Q)`` float64, refusing a pool/tauq pair
    whose identity differs, since a shape-compatible tauq can be silently wrong.  ``train_idx``
    binds rows: mask column ``i`` and tauq row ``i`` must be the same physical example."""
    bad = []
    if pool.dataset != tau.dataset:
        bad.append(f"dataset {pool.dataset!r} vs {tau.dataset!r}")
    if pool.n != tau.n_train:
        bad.append(f"n_train {pool.n} vs {tau.n_train}")
    elif not np.array_equal(pool.train_idx, tau.train_idx):
        bad.append("train_idx differs")
    if pool.Q != tau.n_targets:
        bad.append(f"n_targets {pool.Q} vs {tau.n_targets}")
    elif not np.array_equal(pool.query_idx, tau.query_idx):
        bad.append("query_idx differs")
    if pool.train_hash != tau.train_hash:
        bad.append(f"train_hash {pool.train_hash} vs {tau.train_hash}")
    if pool.query_hash != tau.query_hash:
        bad.append(f"query_hash {pool.query_hash} vs {tau.query_hash}")
    if bad:
        raise ValueError(f"[pair] pool {pool.path} and tauq {tau.path} do NOT describe "
                         f"the same experiment: " + "; ".join(bad))
    return pool.subsets.astype(np.float64) @ tau.tauq.astype(np.float64)


# -- pool_tda: replica-resolved TDA predictions ----------------------------- #
POOL_TDA_SCHEMA = 1
_INNER_KINDS = ("separated-avg-eq15", "single-member", "single-trajectory")
_POOL_TDA_KEYS = ("g", "models_per_replica", "seed_base", "inner", "method",
                  "provenance", "dataset", "n_train", "n_targets", "train_idx",
                  "query_idx", "train_hash", "query_hash", "M", "subsets_source",
                  "subsets_sha16", "schema")


def subsets_digest(subsets) -> str:
    """sha256/16 of the int8 mask bytes: binds a ``pool_tda`` to the exact masks it used."""
    a = np.ascontiguousarray(np.asarray(subsets, dtype=np.int8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class PoolTda:
    """Validated view of a ``pool_tda_<method>.npz``.  ``g`` is ``(R_tda, M, Q)`` float64, one
    self-contained estimator per replica slice."""

    g: np.ndarray
    method: str
    inner: str
    e_features: int                # stored under the schema-1 npz key models_per_replica
    seed_base: int
    provenance: dict
    dataset: str
    n_train: int
    n_targets: int
    train_idx: np.ndarray
    query_idx: np.ndarray
    train_hash: str
    query_hash: str
    M: int
    subsets_source: str
    subsets_sha16: str
    path: str

    @property
    def R_tda(self) -> int:
        return int(self.g.shape[0])

    @property
    def Q(self) -> int:
        return int(self.g.shape[2])


def load_pool_tda(path) -> PoolTda:
    """Load + validate a ``pool_tda``: schema keys, a finite non-empty ``g (R_tda, M, Q)``, a
    known ``inner`` kind, well-formed index draws, and (for canonical datasets) pinned hashes."""
    path = str(path)
    z = np.load(path, allow_pickle=False)
    _require(z, _POOL_TDA_KEYS, path, "pool_tda")
    if int(z["schema"]) != POOL_TDA_SCHEMA:
        raise ValueError(f"[pool_tda] {path}: schema {int(z['schema'])} != "
                         f"{POOL_TDA_SCHEMA}.")
    inner = str(z["inner"])
    if inner not in _INNER_KINDS:
        raise ValueError(f"[pool_tda] {path}: unknown inner kind {inner!r}; known: "
                         f"{_INNER_KINDS}.")
    g = z["g"].astype(np.float64)
    n_train, n_targets, M = int(z["n_train"]), int(z["n_targets"]), int(z["M"])
    if g.ndim != 3 or g.shape[0] < 1 or g.shape[1] != M or g.shape[2] != n_targets:
        raise ValueError(f"[pool_tda] {path}: g shape {g.shape} != (R_tda >= 1, M={M}, "
                         f"Q={n_targets}).")
    if not np.isfinite(g).all():
        raise ValueError(f"[pool_tda] {path}: g contains non-finite values.")
    mpr = int(z["models_per_replica"])            # schema-1 key; python name e_features
    if mpr < 1:
        raise ValueError(f"[pool_tda] {path}: models_per_replica {mpr} < 1.")
    dataset = str(z["dataset"])
    train_idx = _check_idx(z, "train_idx", n_train, path, "pool_tda")
    query_idx = _check_idx(z, "query_idx", n_targets, path, "pool_tda")
    _check_hash_pins(z, dataset, path, "pool_tda")
    return PoolTda(g=g, method=str(z["method"]), inner=inner, e_features=mpr,
                   seed_base=int(z["seed_base"]), provenance=json.loads(str(z["provenance"])),
                   dataset=dataset, n_train=n_train, n_targets=n_targets,
                   train_idx=train_idx, query_idx=query_idx,
                   train_hash=str(z["train_hash"]), query_hash=str(z["query_hash"]),
                   M=M, subsets_source=str(z["subsets_source"]),
                   subsets_sha16=str(z["subsets_sha16"]), path=path)


def write_pool_tda(path, g, source, *, subsets, method, inner, e_features,
                   seed_base, provenance, subsets_source="sample_subsets:777") -> str:
    """Atomically write a ``pool_tda_<method>.npz``: the pairing identity from ``source`` (as
    in :func:`write_tauq`) plus the mask bind (``M`` and :func:`subsets_digest`).  Torch-free."""
    g = np.asarray(g, dtype=np.float64)
    subsets = np.asarray(subsets)
    if isinstance(source, Pool):
        ident = {"dataset": source.dataset, "n_train": source.n, "n_targets": source.Q,
                 "train_idx": source.train_idx, "query_idx": source.query_idx,
                 "train_hash": source.train_hash, "query_hash": source.query_hash}
    else:
        ident = {"dataset": str(source["dataset"]), "n_train": int(source["n_train"]),
                 "n_targets": int(source["n_targets"]),
                 "train_idx": np.asarray(source["train_idx"], dtype=np.int64),
                 "query_idx": np.asarray(source["query_idx"], dtype=np.int64),
                 "train_hash": str(source["train_hash"]),
                 "query_hash": str(source["query_hash"])}
    if inner not in _INNER_KINDS:
        raise ValueError(f"[pool_tda] unknown inner kind {inner!r}; known: {_INNER_KINDS}.")
    if subsets.ndim != 2 or subsets.shape[1] != ident["n_train"]:
        raise ValueError(f"[pool_tda] subsets shape {subsets.shape} != (M, n_train="
                         f"{ident['n_train']}).")
    M = int(subsets.shape[0])
    if g.ndim != 3 or g.shape[0] < 1 or g.shape[1] != M or g.shape[2] != ident["n_targets"]:
        raise ValueError(f"[pool_tda] g shape {g.shape} != (R_tda >= 1, M={M}, "
                         f"Q={ident['n_targets']}) of the target bank.")
    if not np.isfinite(g).all():
        raise ValueError("[pool_tda] g contains non-finite values; refusing to write.")
    if int(e_features) < 1:
        raise ValueError(f"[pool_tda] e_features {e_features} < 1.")
    prov_json = json.dumps(provenance or {}, sort_keys=True, default=str)
    tmp = str(path) + f".tmp.{os.getpid()}"
    np.savez(tmp, g=g, method=str(method), inner=str(inner),
             models_per_replica=int(e_features), seed_base=int(seed_base),
             provenance=prov_json, M=M, subsets_source=str(subsets_source),
             subsets_sha16=subsets_digest(subsets), schema=POOL_TDA_SCHEMA, **ident)
    os.replace(tmp + ".npz", str(path))
    return str(path)


def combine_replicas(pool_tda: PoolTda, replicas=None) -> np.ndarray:
    """Plain mean of the selected replica slices (``None`` = all) -> ``(M, Q)`` float64, an
    exact smaller-ensemble estimate for any replica set."""
    R = pool_tda.R_tda
    if replicas is None:
        return pool_tda.g.mean(axis=0)
    replicas = [int(r) for r in replicas]
    if not replicas:
        raise ValueError("combine_replicas: `replicas` is empty")
    if len(set(replicas)) != len(replicas):
        raise ValueError(f"combine_replicas: duplicate replicas in {replicas}")
    bad = [r for r in replicas if not 0 <= r < R]
    if bad:
        raise ValueError(f"combine_replicas: replica(s) {bad} outside [0, {R}).")
    return pool_tda.g[replicas].mean(axis=0)


def predictions_tda(pool: Pool, pool_tda: PoolTda, replicas=None) -> np.ndarray:
    """LDS predictions ``(M, Q)`` from a replica subset of a ``pool_tda``, refusing a pairing
    that differs from the pool, including the mask bind (the exact masks it was projected on)."""
    bad = []
    if pool.dataset != pool_tda.dataset:
        bad.append(f"dataset {pool.dataset!r} vs {pool_tda.dataset!r}")
    if pool.n != pool_tda.n_train:
        bad.append(f"n_train {pool.n} vs {pool_tda.n_train}")
    elif not np.array_equal(pool.train_idx, pool_tda.train_idx):
        bad.append("train_idx differs")
    if pool.Q != pool_tda.n_targets:
        bad.append(f"n_targets {pool.Q} vs {pool_tda.n_targets}")
    elif not np.array_equal(pool.query_idx, pool_tda.query_idx):
        bad.append("query_idx differs")
    if pool.train_hash != pool_tda.train_hash:
        bad.append(f"train_hash {pool.train_hash} vs {pool_tda.train_hash}")
    if pool.query_hash != pool_tda.query_hash:
        bad.append(f"query_hash {pool.query_hash} vs {pool_tda.query_hash}")
    if pool.M != pool_tda.M:
        bad.append(f"M {pool.M} vs {pool_tda.M}")
    elif subsets_digest(pool.subsets) != pool_tda.subsets_sha16:
        bad.append("mask draw differs (subsets_sha16)")
    if bad:
        raise ValueError(f"[pair] pool {pool.path} and pool_tda {pool_tda.path} do NOT "
                         f"describe the same experiment: " + "; ".join(bad))
    return combine_replicas(pool_tda, replicas)


def _concat_members(paths, parts):
    """The member range ``[lo, hi)`` that concatenated ``pool_tda`` parts cover, or None if no
    part has a ``members`` stamp (``compute-tau --block`` writes one).  Each part holds
    ``R_tda * e_features`` members from its stamp's start (an unstamped first part from 0);
    later parts must be stamped and abut.  Only starts are read: older merges kept the first
    part's range (cifar10's released pool_tda_trak20 says [100, 160] for members 100-339)."""
    stamps = [part.provenance.get("members") for part in parts]
    if not any(stamps):
        return None
    lo = int(stamps[0][0]) if stamps[0] else 0
    at = lo
    for i, (path, part, stamp) in enumerate(zip(paths, parts, stamps)):
        if i and not stamp:
            raise ValueError(f"[pool_tda] {path} does not say which members it holds; "
                             f"refusing to place it after members {lo}-{at - 1}.")
        if i and int(stamp[0]) != at:
            raise ValueError(f"[pool_tda] {path} starts at member {int(stamp[0])}, but the "
                             f"parts before it end at member {at - 1}; refusing to "
                             f"concatenate across a gap or an overlap.")
        at += int(part.R_tda) * int(part.e_features)
    return [lo, at]


def concat_pool_tda(out_path, paths, pool: Pool, *, provenance=None) -> str:
    """Join replica-resolved parts into one ``pool_tda`` along the replica axis, in order.

    Slices are self-contained estimators bound to one mask draw, so separately featurized blocks
    (``compute-tau --members A-B``) or a released file plus extending blocks concatenate exactly.
    Parts must agree on pairing, mask bind and estimator (``method``, ``inner``, ``e_features``,
    ``seed_base``), or a mixed file would silently average different estimators; stamped parts
    must abut, and the result carries their joint range (:func:`_concat_members`)."""
    paths = [str(x) for x in paths]
    if not paths:
        raise ValueError("[pool_tda] concat needs at least one part.")
    parts = [load_pool_tda(x) for x in paths]
    head = parts[0]
    fields = ("dataset", "n_train", "n_targets", "train_hash", "query_hash", "M",
              "method", "inner", "e_features", "seed_base", "subsets_source",
              "subsets_sha16")
    for path, part in zip(paths[1:], parts[1:]):
        bad = [f for f in fields if getattr(part, f) != getattr(head, f)]
        for f in ("train_idx", "query_idx"):
            if not np.array_equal(getattr(part, f), getattr(head, f)):
                bad.append(f)
        if bad:
            raise ValueError(f"[pool_tda] {path} disagrees with {paths[0]} on "
                             f"{', '.join(bad)}; refusing to concatenate parts that are "
                             f"not the same estimator on the same pairing.")
    digest = subsets_digest(pool.subsets)
    if head.subsets_sha16 != digest:
        raise ValueError(f"[pool_tda] the parts bind masks {head.subsets_sha16}, but the "
                         f"given pool's masks digest to {digest}; refusing to rebind.")
    g = np.concatenate([part.g for part in parts], axis=0)
    prov = provenance
    if prov is None:
        prov = {**head.provenance, "concat_of": [{"path": os.path.basename(x),
                                                  "R_tda": int(part.R_tda)}
                                                 for x, part in zip(paths, parts)],
                "n_models": int(g.shape[0]) * int(head.e_features)}
        prov.pop("members", None)                 # the head's own range, not the merge's
        span = _concat_members(paths, parts)
        if span is not None:
            prov["members"] = span
    return write_pool_tda(out_path, g, pool, subsets=pool.subsets, method=head.method,
                          inner=head.inner, e_features=head.e_features,
                          seed_base=head.seed_base, provenance=prov,
                          subsets_source=head.subsets_source)
