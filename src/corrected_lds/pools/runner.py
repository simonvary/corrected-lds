"""Pool orchestration, task-agnostic and torch-free.

A pool is an ``(R, M, Q)`` array: cell ``(r, j)`` is one model trained on physical subset ``j``
with seed ``cell_seed(r, j)`` and read out as ``(Q,)`` margins by a :class:`PoolTask`, which
must be reproducible up to ~1e-6 GPU atomics.  Parallelism is shards only (one process per GPU,
then :func:`merge_shards`): never run two writers on one npz file, as they silently revert each
other's cells.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import corrected_lds
from corrected_lds.banks.masks import sample_subsets

__all__ = [
    "CELL_SEED_BASE",
    "MASK_SEED",
    "PoolTask",
    "build_pool",
    "cell_owner",
    "cell_seed",
    "identity_from_meta",
    "init_pool",
    "merge_shards",
    "pool_meta",
    "remaining_cells",
    "shard_cells",
]

CELL_SEED_BASE = 50_000
MASK_SEED = 777                    # fixed subset-sampler seed (prefix-stable across M)


def cell_seed(r: int, j: int) -> int:
    """Nested per-cell training seed ``BASE + r*10000 + j``: index-stable, so growing M or R
    never changes an existing cell; ``j < 10000`` keeps it collision-free."""
    r, j = int(r), int(j)
    if not 0 <= j < 10_000:
        raise ValueError(f"cell_seed needs 0 <= j < 10000 (got j={j}); at j=10000 it would "
                         f"collide with (r+1, 0).")
    return CELL_SEED_BASE + r * 10_000 + j


def cell_owner(r: int, j: int, N: int) -> int:
    """Which of ``N`` static shards owns cell ``(r, j)``: ``blake2b`` of the utf-8 key ``"r:j"``
    (pinned: 8-byte digest, big-endian; not the salted ``hash``), stable as M or R grow."""
    h = hashlib.blake2b(f"{int(r)}:{int(j)}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % int(N)


def shard_cells(M: int, R: int, shard: int, N: int):
    """Row-major ``(r, j)`` cells of the ``R x M`` grid owned by ``shard`` (``0 <= shard < N``)."""
    return [(r, j) for r in range(R) for j in range(M) if cell_owner(r, j, N) == int(shard)]


def _owner_grid(M, R, N):
    return np.array([[cell_owner(r, j, N) for j in range(M)] for r in range(R)], dtype=int)


@dataclass(frozen=True)
class PoolTask:
    """The model, training and readout behind one pool cell: ``train_cell(data, subset_idx,
    seed)`` trains one model on the physical subset, ``readout(model, data)`` returns ``(Q,)``;
    ``recipe`` has ``alpha`` (subset fraction) and a stable provenance identity ``to_dict()``."""

    recipe: Any
    train_cell: Callable[[dict, np.ndarray, int], Any]
    readout: Callable[[Any, dict], np.ndarray]


def _leading_true(mask) -> int:
    bad = np.flatnonzero(~np.asarray(mask, bool))
    return int(bad[0]) if bad.size else int(np.size(mask))


def pool_meta(margins):
    """``(completed_cells, done_rows)`` of an ``(R, M, Q)`` pool (NaN = not yet trained), where
    ``done_rows`` counts the complete leading rows (the downstream loader's view)."""
    cell_done = ~np.isnan(margins).any(axis=2)                # (R, M) bool
    return int(cell_done.sum()), _leading_true(cell_done.all(axis=1))


def remaining_cells(margins):
    """Row-major ``(r, j)`` list of not-yet-trained cells."""
    r, j = np.where(np.isnan(margins).any(axis=2))
    return list(zip(r.tolist(), j.tolist()))


def _code_version(pkg_dir=None) -> str:
    """``corrected-lds-<sha>[-dirty]`` in the package's own git checkout (via ``rev-parse``, so a
    tag cannot rename the code that built a file), else ``corrected-lds-<__version__>``."""
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(corrected_lds.__file__))
    root = os.path.dirname(os.path.dirname(pkg_dir))

    def git(*args):
        out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                             timeout=10, check=False)
        return out.stdout.strip() if out.returncode == 0 else None

    try:
        top, sha = git("rev-parse", "--show-toplevel"), git("rev-parse", "--short=7", "HEAD")
        if top and sha and os.path.realpath(top) == os.path.realpath(root):
            dirty = git("status", "--porcelain", "--untracked-files=no") != ""
            return f"corrected-lds-{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return f"corrected-lds-{corrected_lds.__version__}"


def _trained_by(d) -> set:
    """The code versions recorded as having trained a stored pool file's cells."""
    if "code_versions" in d.files:
        return set(json.loads(str(d["code_versions"])))
    return {str(d["code_version"])} if "code_version" in d.files else set()


def identity_from_meta(meta):
    """Data-identity keys from a loader's ``meta`` (``train_idx``/``query_idx`` rows and their
    split's content digests ``train_hash``/``query_hash``); present keys must match exactly."""
    ident = {}
    for k in ("train_idx", "query_idx"):
        if k in meta:
            ident[k] = np.asarray(meta[k], dtype=np.int64)
    for k in ("train_hash", "query_hash"):
        if k in meta:
            ident[k] = str(meta[k])
    return ident


def _identity_mismatches(d, ident):
    """Stored-vs-requested data-identity mismatches; a key on one side only is one too, since
    saving would otherwise relabel old readouts with new metadata."""
    bad = []
    for k in ("train_idx", "query_idx", "train_hash", "query_hash"):
        stored = d[k] if k in d.files else None
        want = ident.get(k)
        if stored is None and want is None:
            continue
        if stored is None or want is None:
            bad.append(f"{k} {'missing in stored file' if stored is None else 'not provided'}")
        elif k.endswith("_idx"):
            if not np.array_equal(np.asarray(stored, dtype=np.int64), want):
                bad.append(f"{k} differs")
        elif str(stored) != want:
            bad.append(f"{k} {str(stored)!r} vs {want!r}")
    return bad


def _guarded_stored(path, d, Q, R, M, recipe_json, dataset, subsets, subsets_source, ident):
    """``d["margins"]`` once every :func:`init_pool` guard but completeness passes."""
    stored = d["margins"]
    n_train = subsets.shape[1]
    R_old, M_old = int(stored.shape[0]), int(stored.shape[1])
    stored_recipe = str(d["recipe"]) if "recipe" in d.files else None
    stored_dataset = str(d["dataset"]) if "dataset" in d.files else "MISSING"
    stored_seed_mode = str(d["seed_mode"]) if "seed_mode" in d.files else "MISSING"
    stored_source = (str(d["subsets_source"]) if "subsets_source" in d.files
                     else "MISSING")
    stored_subsets = d["subsets"]
    prefix_ok = (stored_subsets.shape == (M_old, n_train)
                 and np.array_equal(stored_subsets, subsets[:M_old]))
    ok = (int(stored.shape[2]) == Q and stored_recipe == recipe_json
          and stored_recipe is not None and stored_dataset == dataset
          and stored_seed_mode == "nested" and stored_source == subsets_source
          and prefix_ok)
    if not ok:
        raise RuntimeError(
            f"[pool] {path} exists but does NOT match the requested pool (recipe/mask-prefix/"
            f"Q/dataset/seed_mode/subsets_source); refusing to overwrite. stored R={R_old} "
            f"M={M_old} Q={int(stored.shape[2])} dataset={stored_dataset} "
            f"seed_mode={stored_seed_mode} subsets_source={stored_source} "
            f"recipe={'set' if stored_recipe else 'MISSING'}; requested R={R} M={M} Q={Q} "
            f"dataset={dataset}. Use a fresh directory or restore a matching pool.")
    bad_ident = _identity_mismatches(d, ident)
    if bad_ident:
        raise RuntimeError(
            f"[pool] {path} exists but its DATA IDENTITY does not match the requested "
            f"data ({'; '.join(bad_ident)}); refusing — its cells were trained on a "
            f"different train/query draw, and continuing would relabel them.")
    if R_old > R or M_old > M:
        raise RuntimeError(f"[pool] stored (R={R_old}, M={M_old}) exceeds requested "
                           f"(R={R}, M={M}); refusing to shrink.")
    return stored


def init_pool(pool_dir, n_train, Q, M, R, recipe, dataset, filename="pool.npz",
              compress=False, ident=None, shard=None):
    """Set up an ``(R, M, Q)`` pool and its atomic saver, resuming or extending (in M and R) a
    stored one; returns ``(margins, subsets, sub_idx, path, save)``.

    A stored pool must match exactly in recipe JSON, dataset, seed_mode, subsets_source and
    data identity (:func:`identity_from_meta`), its masks must prefix the requested ones, and
    only a complete pool may grow (never shrink); a mismatch raises rather than relabel cells.
    Saves stamp ``schema=3``, ``code_version`` and ``code_versions`` (every version that wrote
    cells).  A new shard file (``shard=(i, N)``) starts with its owned cells copied from a
    complete, guarded ``pool_dir/pool.npz``, so a sharded extension trains only the new cells."""
    subsets = sample_subsets(M, n_train, recipe.alpha, seed=MASK_SEED)   # (M, n) int8, prefix-stable
    subsets_source = f"sample_subsets:{MASK_SEED}"
    recipe_json = json.dumps(recipe.to_dict(), sort_keys=True)
    ident = dict(ident or {})
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, filename)
    base = os.path.join(pool_dir, "pool.npz")
    margins = np.full((R, M, Q), np.nan)
    guards = (Q, R, M, recipe_json, dataset, subsets, subsets_source, ident)
    trained = set()                                        # code behind the kept cells
    if os.path.exists(path):
        d = np.load(path)
        stored = _guarded_stored(path, d, *guards)
        trained = _trained_by(d)
        R_old, M_old = int(stored.shape[0]), int(stored.shape[1])
        if (R_old, M_old) != (R, M) and not _stored_complete(d, stored):
            raise RuntimeError(f"[pool] cannot extend an INCOMPLETE pool (R={R_old}, M={M_old}); "
                               f"finish it first, then extend.")
        margins[:R_old, :M_old] = stored
        grew = "" if (R_old, M_old) == (R, M) else f" -> extend to (R={R}, M={M})"
        print(f"[pool] resuming (R={R_old}, M={M_old}, completed {pool_meta(margins)[0]}/{R * M})"
              f"{grew}", flush=True)
    elif shard is not None and path != base and os.path.exists(base):
        i, N = int(shard[0]), int(shard[1])
        d = np.load(base)
        stored = _guarded_stored(base, d, *guards)
        trained = _trained_by(d)
        if not _stored_complete(d, stored):
            raise RuntimeError(f"[pool] cannot seed shard {i} of {N} from an INCOMPLETE "
                               f"{base}; finish it (or move it away) first.")
        R_old, M_old = int(stored.shape[0]), int(stored.shape[1])
        owned = _owner_grid(M_old, R_old, N) == i
        margins[:R_old, :M_old][owned] = stored[owned]
        print(f"[pool] shard {i} of {N}: {int(owned.sum())} cell(s) copied from {base} "
              f"(R={R_old}, M={M_old}); {len(shard_cells(M, R, i, N)) - int(owned.sum())} "
              f"to train", flush=True)

    sub_idx = [np.flatnonzero(subsets[j]) for j in range(M)]
    stamp = _code_version()

    def save(margins, trained_by=None):
        """``trained_by``: the code versions behind the new cells (default: this process)."""
        completed, done_rows = pool_meta(margins)
        versions = trained | ({stamp} if trained_by is None else set(trained_by))
        tmp = path + ".tmp"                                # atomic: write temp, then os.replace
        saver = np.savez_compressed if compress else np.savez   # shards are mostly-NaN -> compress
        saver(tmp, margins=margins, subsets=subsets.astype(np.int8), completed_cells=completed,
              done_rows=done_rows, M=M, R=R, alpha=recipe.alpha, recipe=recipe_json,
              dataset=dataset, seed_mode="nested", subsets_source=subsets_source,
              schema=3, code_version=stamp, code_versions=json.dumps(sorted(versions)),
              **ident)
        os.replace(tmp + ".npz", path)

    return margins, subsets, sub_idx, path, save


def _stored_complete(d, stored):
    """Whether a stored pool is exactly complete, as extending requires: every value finite
    (inf is not truth) and its counters equal the recomputed ones (else it was hand-edited)."""
    completed, done_rows = pool_meta(stored)
    return (np.isfinite(stored).all()
            and completed == stored.shape[0] * stored.shape[1]
            and "completed_cells" in d.files and int(d["completed_cells"]) == completed
            and "done_rows" in d.files and int(d["done_rows"]) == done_rows)


def build_pool(pool_dir, task: PoolTask, data, M, R, *, save_every=10, shard=None,
               verbose=True):
    """Generate, resume or extend the pool for ``task``, saving every ``save_every`` models;
    returns the pool/shard path.  ``shard=(i, N)`` trains only shard ``i``'s cells into its own
    compressed ``pool_shard_i.npz`` (safe concurrently; then :func:`merge_shards`).  Readouts must
    be exactly ``(Q,)`` and finite, since resume treats any non-NaN cell (even ``inf``) as done."""
    try:
        import torch
    except ImportError:                                    # CPU-only synthetic tasks (tests)
        torch = None
    n, Q = data["meta"]["n_train"], data["meta"]["n_targets"]
    dataset = data["meta"]["dataset"]
    if shard is None:
        filename, compress, cells = "pool.npz", False, None
    else:
        i, N = int(shard[0]), int(shard[1])
        filename, compress, cells = f"pool_shard_{i}.npz", True, shard_cells(M, R, i, N)
    margins, _subsets, sub_idx, path, save = init_pool(
        pool_dir, n, Q, M, R, task.recipe, dataset, filename=filename, compress=compress,
        ident=identity_from_meta(data["meta"]), shard=shard)
    todo = (remaining_cells(margins) if cells is None
            else [c for c in cells if np.isnan(margins[c[0], c[1]]).any()])
    for k, (r, j) in enumerate(todo):
        model = task.train_cell(data, sub_idx[j], cell_seed(r, j))
        a = np.asarray(task.readout(model, data), dtype=np.float64)
        if a.shape != (Q,):
            raise ValueError(f"[pool] cell ({r}, {j}) readout has shape {a.shape}, expected "
                             f"({Q},) -- a scalar would silently broadcast across every query.")
        if not np.isfinite(a).all():
            raise ValueError(f"[pool] cell ({r}, {j}) readout has "
                             f"{int((~np.isfinite(a)).sum())} non-finite value(s); inf would "
                             f"count as 'done' forever under the NaN-truth resume.")
        margins[r, j] = a
        del model
        if (k + 1) % save_every == 0 or (k + 1) == len(todo):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            save(margins)
            if verbose:
                completed, done_rows = pool_meta(margins)
                print(f"[pool] {filename}: {completed}/{R * M} models "
                      f"(done_rows {done_rows}/{R})", flush=True)
    if not os.path.exists(path):        # nothing left to train (e.g. a shard seeded with
        save(margins)                   # only copied cells): the merge still needs its file
    return path


def merge_shards(pool_dir, recipe, data, M, R, N, *, allow_mixed_code=False):
    """Assemble ``pool.npz`` from the ``N`` shard files alone; returns its path.  Each shard must
    match the requested provenance and data identity and fill only its owned cells; every cell
    must come from exactly one shard and be finite, cells of an existing ``pool.npz`` must come
    back bit-identical (copied, not retrained), no two seeds may give identical margins on one
    mask, and mixed code versions need ``allow_mixed_code``."""
    n, Q = data["meta"]["n_train"], data["meta"]["n_targets"]
    dataset = data["meta"]["dataset"]
    ident = identity_from_meta(data["meta"])
    # init_pool guards any existing pool.npz and returns the saver and canonical subsets
    prior, subsets, _sub_idx, path, save = init_pool(pool_dir, n, Q, M, R, recipe, dataset,
                                                    ident=ident)
    recipe_json = json.dumps(recipe.to_dict(), sort_keys=True)
    source = f"sample_subsets:{MASK_SEED}"
    margins = np.full((R, M, Q), np.nan)
    owner = _owner_grid(M, R, N)
    filled = np.zeros((R, M), dtype=int)
    writers, trained = set(), set()
    for i in range(int(N)):
        sp = os.path.join(pool_dir, f"pool_shard_{i}.npz")
        if not os.path.exists(sp):
            raise RuntimeError(f"[pool] merge: shard {i} missing ({sp}); run all {N} shards first.")
        d = np.load(sp)
        sd = d["margins"]
        if sd.shape != (R, M, Q):
            raise RuntimeError(f"[pool] merge: shard {i} shape {sd.shape} != {(R, M, Q)}.")
        s_recipe = str(d["recipe"]) if "recipe" in d.files else None
        s_dataset = str(d["dataset"]) if "dataset" in d.files else "MISSING"
        s_seed_mode = str(d["seed_mode"]) if "seed_mode" in d.files else "MISSING"
        s_source = str(d["subsets_source"]) if "subsets_source" in d.files else "MISSING"
        if not (s_recipe == recipe_json and s_dataset == dataset and s_seed_mode == "nested"
                and s_source == source
                and "subsets" in d.files and np.array_equal(d["subsets"], subsets)):
            raise RuntimeError(
                f"[pool] merge: shard {i} provenance (recipe/dataset/seed_mode/subsets_source/"
                f"masks) does NOT match the requested pool; refusing to merge a stale shard.")
        bad_ident = _identity_mismatches(d, ident)
        if bad_ident:
            raise RuntimeError(
                f"[pool] merge: shard {i} DATA IDENTITY does not match the requested data "
                f"({'; '.join(bad_ident)}); its cells were trained on a different train/query "
                f"draw — refusing to publish them under this identity.")
        done = ~np.isnan(sd).any(axis=2)                   # (R, M) cells this shard filled
        if (owner[done] != i).any():
            raise RuntimeError(f"[pool] merge: shard {i} filled cell(s) it does not own.")
        margins[done] = sd[done]
        filled += done
        writers.add(str(d["code_version"]) if "code_version" in d.files else "unstamped")
        trained |= _trained_by(d)
    if not np.all(filled == 1):                            # every cell by exactly one shard
        raise RuntimeError(f"[pool] merge: {int((filled == 0).sum())} cell(s) written by NO shard "
                           f"and {int((filled > 1).sum())} by >1; every cell must come from "
                           f"exactly one shard.")
    if not np.isfinite(margins).all():                        # reject inf/NaN 'truth'
        raise RuntimeError("[pool] merge: pool contains non-finite values; refusing to publish.")
    had = ~np.isnan(prior).any(axis=2)                        # cells of the existing pool.npz
    changed = had & ~(margins == prior).all(axis=2)
    if changed.any():
        raise RuntimeError(f"[pool] merge: {int(changed.sum())} cell(s) of the existing "
                           f"pool.npz would CHANGE; a shard retrained cells it should have "
                           f"copied (created before pool.npz was in place?) -- refusing.")
    for j in range(M):                         # every seed must have trained its own model
        _, inv, counts = np.unique(margins[:, j, :], axis=0, return_inverse=True,
                                   return_counts=True)
        if (counts > 1).any():
            seeds = np.flatnonzero(inv.ravel() == int(np.argmax(counts))).tolist()
            raise RuntimeError(f"[pool] merge: seeds {seeds} gave identical margins on "
                               f"mask {j}; a cell was trained without its own seed -- "
                               f"refusing.")
    if len(writers) > 1 and not allow_mixed_code:
        raise RuntimeError(f"[pool] merge: the shards were written by different code "
                           f"({', '.join(sorted(writers))}); pass --allow-mixed-code to "
                           f"merge them anyway.")
    save(margins, trained_by=trained)
    return path
