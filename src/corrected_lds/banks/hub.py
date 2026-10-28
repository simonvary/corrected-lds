"""Bank distribution: manifest-verified fetch of released banks.

The committed manifest is the trust anchor: a full sha256 per file, checked before any download
is kept.  Fetching is stdlib-only and sends the ``HF_TOKEN`` bearer only within the strict
huggingface.co scope of :func:`_sends_token`; only pushing needs ``huggingface_hub`` (``[hub]``).
"""

import hashlib
import json
import os
import urllib.parse
import urllib.request

import numpy as np

__all__ = ["bank_files", "download", "fetch_banks", "manifest_block", "push_bank",
           "sha256_file", "validate_bank"]

_CHUNK = 1 << 22
_TOKEN_HOST = "huggingface.co"     # exact match — subdomains and lookalikes get nothing


def _sends_token(url) -> bool:
    """True iff ``url`` may carry the bearer: https AND host exactly ``huggingface.co``."""
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" and u.hostname == _TOKEN_HOST


def sha256_file(path) -> str:
    """Full (streamed) sha256 hex digest of a file — the manifest's per-file pin."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(_CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(url, dest, sha256, token=None) -> str:
    """Fetch ``url`` to ``dest``, keeping it only if its full sha256 equals ``sha256``.

    Atomic; a ``dest`` already present that verifies is kept without network access, so files
    placed by other means (e.g. ``hf download``) still get the manifest check.  ``token`` goes
    only to first-hop https requests to exactly huggingface.co, as an unredirected header: the
    Hub 302s large payloads to signed CDN URLs, where a forwarded bearer would leak off-host.
    A same-host redirect (a renamed repo) therefore loses auth and fails with 401.
    """
    dest = str(dest)
    if os.path.exists(dest) and sha256_file(dest) == sha256:
        print(f"[fetch] {dest} already present and verified.")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.tmp.{os.getpid()}"
    print(f"[fetch] {url} -> {dest}")
    req = urllib.request.Request(url)
    if token and _sends_token(url):
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
            while True:
                b = r.read(_CHUNK)
                if not b:
                    break
                out.write(b)
    except BaseException:                       # network/interrupt: no .tmp litter either
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    got = sha256_file(tmp)
    if got != sha256:
        os.remove(tmp)
        raise SystemExit(f"[fetch] sha256 mismatch for {dest}: got {got}, manifest "
                         f"pins {sha256}; refusing to keep the download.")
    os.replace(tmp, dest)
    return dest


def fetch_banks(manifest_path, root, names=None, token=None) -> int:
    """Download the banks listed in ``manifest_path`` (``names=None`` = all) into
    ``root/<name>/``, each file from ``<url>/<bank>/<file>`` and verified via :func:`download`.
    Returns a CLI exit code: 0 on success, 1 while the manifest lists no banks yet."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    if manifest.get("format") != 2:
        raise SystemExit(f"[fetch] {manifest_path} is not a format-2 manifest.")
    banks = manifest["banks"]
    if not banks:
        print(f"[fetch] {manifest_path} lists no banks yet — they are published at release.")
        return 1
    for name in (names or sorted(banks)):
        if name not in banks:
            raise SystemExit(f"[fetch] unknown bank {name!r}; manifest has {sorted(banks)}")
        for f, sha256 in banks[name].items():
            download(f"{manifest['url']}/{name}/{f}", os.path.join(root, name, f), sha256,
                     token=token)
    return 0


# -- push: validate -> one pinned commit -> manifest block ------------------ #
def _is_alt_pool(f):
    """``pool_<label>.npz`` other than a ``pool_tda_*``: a second readout of the SAME
    retrained models (e.g. ``pool_dropout_readout.npz``), never the default."""
    return f.startswith("pool_") and f.endswith(".npz") and not f.startswith("pool_tda_")


def bank_files(bank_dir):
    """The files that are the bank, in order: ``pool.npz``, alternate-readout
    ``pool_<label>.npz``, ``tauq_*.npz``, ``pool_tda_*.npz``, each group sorted.  Refuses a
    directory without a pool.npz; logs, shards and checkpoint dirs are never bank files."""
    bank_dir = str(bank_dir)
    if not os.path.exists(os.path.join(bank_dir, "pool.npz")):
        raise ValueError(f"[push] {bank_dir} has no pool.npz — not a bank.")
    names = os.listdir(bank_dir)
    alts = sorted(f for f in names if _is_alt_pool(f))
    tauqs = sorted(f for f in names if f.startswith("tauq_") and f.endswith(".npz"))
    ptdas = sorted(f for f in names if f.startswith("pool_tda_") and f.endswith(".npz"))
    return ["pool.npz", *alts, *tauqs, *ptdas]


def validate_bank(bank_dir):
    """Run the gates a consumer will (loaders plus the ``predictions*`` pairing gates, so the
    pairing rules live in one place) on every bank file, then return the file list."""
    from corrected_lds.banks.artifacts import (
        load_pool,
        load_pool_tda,
        load_tauq,
        predictions,
        predictions_tda,
    )
    files = bank_files(bank_dir)
    pool = load_pool(os.path.join(bank_dir, files[0]))
    for f in files[1:]:
        path = os.path.join(bank_dir, f)
        if f.startswith("pool_tda_"):
            predictions_tda(pool, load_pool_tda(path))
        elif _is_alt_pool(f):      # every tauq / pool_tda must pair with it too
            alt = load_pool(path)
            same = (alt.dataset == pool.dataset and alt.train_hash == pool.train_hash
                    and alt.query_hash == pool.query_hash
                    and np.array_equal(alt.train_idx, pool.train_idx)
                    and np.array_equal(alt.query_idx, pool.query_idx)
                    and np.array_equal(alt.subsets, pool.subsets))
            if not same:
                raise ValueError(f"[push] {f} is not a readout of pool.npz's masks and "
                                 f"examples — an alternate pool must share both.")
        else:
            predictions(pool, load_tauq(path))
    return files


def manifest_block(name, files, repo_id, revision):
    """The banks-manifest.json update for one pushed bank (pure, no I/O): its ``url`` moves to
    ``revision``, the commit oid the push returned, which still holds the other banks' files
    unchanged (so every file stays commit-pinned and :func:`_sends_token` accepts it), and
    ``banks`` gains ``name``'s ``{file: sha256}``.  ``files`` is ``(basename, sha256)`` pairs."""
    return {"url": f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}",
            "banks": {str(name): {f: h for f, h in files}}}


def _hf():
    """Lazily import ``huggingface_hub`` (the tests' stub point) and return the module, since
    ``create_commit`` needs ``CommitOperationAdd``, not just ``HfApi``."""
    try:
        import huggingface_hub
    except ImportError as e:
        raise ImportError("[push] pushing banks needs huggingface_hub — install the "
                          "extra: pip install 'corrected-lds[hub]'") from e
    return huggingface_hub


def _rel_in(bank_dir, path):
    """Repo subpath for an extra: bank-dir-relative when inside (``ckpts/…`` keeps
    its layout), basename otherwise."""
    rel = os.path.relpath(path, bank_dir)
    return os.path.basename(path) if rel.startswith("..") else rel


def push_bank(bank_dir, repo_id, *, name=None, extra_files=(), private=True):
    """Validate ``bank_dir``, upload it under ``<name>/…`` to the HF dataset repo ``repo_id``
    in one commit, and return the :func:`manifest_block` for a human to paste into
    banks-manifest.json (the trust anchor stays hand-committed).  ``extra_files`` (shard
    npzs, checkpoints, logs) are uploaded verbatim and unvalidated (the receiving end checks
    them) and left out of the block.  ``private`` applies only at repo creation, so a
    default-private push cannot un-publish a released repo."""
    bank_dir = str(bank_dir)
    name = name or os.path.basename(os.path.normpath(bank_dir))
    files = validate_bank(bank_dir)
    print(f"[push] {bank_dir}: {len(files)} bank file(s) validated.")
    extras = [str(p) for p in extra_files]
    for p in extras:
        if not os.path.isfile(p):
            raise ValueError(f"[push] extra {p} is not a file (directories are not "
                             f"expanded — pass the files).")
    hh = _hf()
    hashed = [(f, sha256_file(os.path.join(bank_dir, f))) for f in files]
    ops = [hh.CommitOperationAdd(path_in_repo=f"{name}/{f}",
                                 path_or_fileobj=os.path.join(bank_dir, f))
           for f in files]
    ops += [hh.CommitOperationAdd(path_in_repo=f"{name}/{_rel_in(bank_dir, p)}",
                                  path_or_fileobj=p) for p in extras]
    api = hh.HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.create_commit(repo_id, repo_type="dataset", operations=ops,
                             commit_message=f"push bank {name!r} ({len(files)} bank "
                                            f"file(s), {len(extras)} extra(s))")
    rev = str(info.oid)
    print(f"[push] {repo_id} @ {rev}: {len(ops)} file(s) in one commit.")
    return manifest_block(name, hashed, repo_id, rev)
