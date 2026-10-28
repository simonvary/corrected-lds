"""Random fixed-size subset masks — the sampler every released bank was built with.

The RNG is keyed by ``(crc32(b"subsets"), seed)``; released pools use ``seed=777``.  The key
and sampling path must stay byte-identical or cached pools drift (the tests pin sha256s).
"""

import zlib

import numpy as np

__all__ = ["sample_subsets"]


def sample_subsets(M: int, n: int, alpha: float, seed: int) -> np.ndarray:
    """M fixed-size subset masks, shape (M, n), int8: each row has exactly ``k = floor(alpha*n)``
    ones drawn uniformly without replacement, the TRAK/datamodels convention (Park et al.
    2023, Def. 2.3: "each of size alpha*n").  Rows are prefix-stable in M (growing M only
    appends rows), which the pool extend guard relies on."""
    k = int(np.floor(alpha * n))
    if not (0 <= k <= n):
        raise ValueError(f"alpha={alpha} gives k={k} outside [0, n={n}]")
    r = np.random.default_rng((zlib.crc32(b"subsets"), seed))
    out = np.zeros((M, n), dtype=np.int8)
    if k == 0:
        return out
    if k == n:
        out[:] = 1
        return out
    scores = r.random(size=(M, n))
    sel = np.argpartition(scores, k, axis=1)[:, :k]     # k smallest scores per row
    np.put_along_axis(out, sel, np.int8(1), axis=1)
    return out
