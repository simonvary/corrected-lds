"""Compare a rebuilt bank file with the released one, and check a bank's tauqs against its
pool_tdas (CPU, torch-free).  Rebuilt files never match the release by sha256 (provenance
spellings and float noise differ), so the arrays are compared instead.
"""

import os
import re

import numpy as np

from corrected_lds.banks.artifacts import (
    load_pool,
    load_pool_tda,
    load_tauq,
    predictions,
    predictions_tda,
)

__all__ = ["compare_files", "consistency"]


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
    if "margins" in keys:
        return "pool", load_pool(path)
    return ("pool_tda", load_pool_tda(path)) if "g" in keys else ("tauq", load_tauq(path))


def compare_files(new_path, ref_path) -> dict:
    """Align a rebuilt pool, tauq or pool_tda with the released file on their common prefix
    (seeds or replicas, and masks) and measure bit-identity, max abs and relative difference,
    and correlation; ``mismatch`` lists identity, shape or mask conflicts that prevent it."""
    kind, new = _load(new_path)
    ref_kind, ref = _load(ref_path)
    if kind != ref_kind:
        return {"kind": kind, "mismatch": [f"a {kind} against a {ref_kind}"]}
    bad = [k for k in ("dataset", "train_hash", "query_hash") if getattr(new, k) != getattr(ref, k)]
    bad += [k for k in ("train_idx", "query_idx")
            if not np.array_equal(getattr(new, k), getattr(ref, k))]
    if kind == "tauq":
        a, b = new.tauq, ref.tauq
        if a.shape != b.shape:
            bad.append(f"shape {a.shape} vs {b.shape}")
    elif kind == "pool":
        R, M = min(new.R, ref.R), min(new.M, ref.M)
        if new.recipe != ref.recipe:
            bad.append("recipe")
        if new.Q != ref.Q or not np.array_equal(new.subsets[:M], ref.subsets[:M]):
            bad.append("masks or Q")
        a, b = new.margins[:R, :M], ref.margins[:R, :M]
    else:                                     # pool_tda: its masks are bound by digest
        if (new.M, new.Q, new.subsets_sha16) != (ref.M, ref.Q, ref.subsets_sha16):
            bad.append("masks or Q")
        R = min(new.R_tda, ref.R_tda)
        a, b = new.g[:R], ref.g[:R]
    if bad:
        return {"kind": kind, "mismatch": bad}
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b).max()
    return {"kind": kind, "mismatch": [], "shape": a.shape, "bitwise": bool(np.array_equal(a, b)),
            "max_abs": float(d), "max_rel": float(d / max(float(np.abs(b).max()), 1e-300)),
            "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1])}


def consistency(bank_dir, rtol=1e-7) -> list:
    """Each plain-mean ``tauq_<m>_e<E>.npz`` of a bank against ``pool_tda_<m>.npz``: ``S @ tauq``
    must equal the mean of the first E replicas (the release agrees to <= 3e-8 rel).  Skips
    Eq. 15 TRAK files (the tauq pools all E*e members) and tauqs without a pool_tda."""
    pool = load_pool(os.path.join(bank_dir, "pool.npz"))
    rows, loaded = [], {}
    for name in sorted(os.listdir(bank_dir)):
        m = re.fullmatch(r"tauq_(.+)_e(\d+)\.npz", name)
        partner = f"pool_tda_{m[1]}.npz" if m else ""
        if not m or not os.path.exists(os.path.join(bank_dir, partner)):
            continue
        if partner not in loaded:             # one at a time: pool_tda files are large
            loaded = {partner: load_pool_tda(os.path.join(bank_dir, partner))}
        ptda, E = loaded[partner], int(m[2])
        if ptda.inner == "separated-avg-eq15":
            continue
        rel = float("nan")
        if E <= ptda.R_tda:
            pred = predictions(pool, load_tauq(os.path.join(bank_dir, name)))
            mean = predictions_tda(pool, ptda, replicas=range(E))
            rel = float(np.abs(pred - mean).max() / np.abs(mean).max())
        rows.append({"tauq": name, "pool_tda": partner, "E": E, "max_rel": rel,
                     "ok": rel <= rtol})
    return rows
