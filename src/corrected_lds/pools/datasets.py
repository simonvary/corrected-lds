"""Canonical dataset builds — row order is part of every artifact's identity.

A wrong row order silently corrupts every artifact, so every build and every load verifies
the sha256/16 content pins in :mod:`corrected_lds.banks.presets` (ImageNet's ~70 GB blobs
get a structural gate per load instead, see :func:`_blob_gate`).  CIFAR is always built from
the pinned HuggingFace parquet files (never torchvision/Toronto — different order); QNLI
label sequences are pinned because HF ``datasets`` may reshuffle between versions.  Module
import is torch-free; loaders import torch lazily, and cache builders need the ``[gpu]`` or
``[imagenet]`` extra.
"""

import hashlib
import io
import itertools
import os
import urllib.request

import numpy as np

from corrected_lds.banks.presets import DATA_HASHES, bank_hashes

__all__ = [
    "IMAGENET_CLASSES",
    "IMAGENET_MEAN255",
    "IMAGENET_STD255",
    "IMAGENET_TRAIN",
    "IMAGENET_VAL",
    "QNLI_TRAIN",
    "QNLI_VAL",
    "WIKITEXT2_REPO",
    "WIKITEXT2_REVISION",
    "build_cifar_cache",
    "build_imagenet_cache",
    "build_mnist_cache",
    "build_qnli_cache",
    "build_wikitext2_cache",
    "cache_dir",
    "imagenet_cache_pins",
    "load_cifar",
    "load_imagenet",
    "load_mnist",
    "load_qnli",
    "load_wikitext2",
    "smoke_imagenet_source",
]


# -- shared: cache location + the content-hash gate ------------------------- #
def cache_dir() -> str:
    d = os.environ.get("CORRECTED_LDS_DATA") or os.path.expanduser("~/.cache/corrected_lds")
    os.makedirs(d, exist_ok=True)
    return d


def _sha16(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _verify(dataset, arrays):
    """Assert every array matches its pin; raise with the full mismatch list."""
    pins = DATA_HASHES[dataset]
    got = {k: _sha16(arrays[k]) for k in pins}
    bad = [f"{k}: got {got[k]}, pinned {pins[k]}" for k in pins if got[k] != pins[k]]
    if bad:
        raise RuntimeError(
            f"[data] {dataset} content-hash mismatch — the row order or content is wrong, and "
            f"every mask/pools/tauq artifact would be silently corrupted: " + "; ".join(bad))


# -- CIFAR-10: raw uint8 cache in canonical HF-parquet row order ------------ #
_CIFAR_URL = "https://huggingface.co/datasets/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text"
_CIFAR_MEAN01 = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR_STD01 = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def build_cifar_cache() -> str:
    """Download the HF CIFAR-10 parquet, decode to ``Xtr (50000,3,32,32) u8 / ytr i8 /
    Xte / yte``, verify the pins and atomically write ``cifar10_raw.npz`` (idempotent)."""
    npz = os.path.join(cache_dir(), "cifar10_raw.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        _verify("cifar10", {k: z[k] for k in ("Xtr", "ytr", "Xte", "yte")})
        print(f"[data] {npz} already present and verified.", flush=True)
        return npz
    import pyarrow.parquet as pq
    from PIL import Image

    def fetch(split):
        dst = os.path.join(cache_dir(), f"cifar10_{split}.parquet")
        if not os.path.exists(dst) or os.path.getsize(dst) < 1_000_000:
            url = f"{_CIFAR_URL}/{split}/0000.parquet"
            print(f"[data] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, dst)
        return dst

    def decode(path):
        t = pq.read_table(path)
        lab = t.column("label").to_numpy().astype(np.int64)
        imgs = t.column("img").to_pylist()
        X = np.zeros((len(lab), 3, 32, 32), np.uint8)
        for i, d in enumerate(imgs):
            b = d["bytes"] if isinstance(d, dict) else d
            X[i] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.uint8).transpose(2, 0, 1)
        return X, lab

    Xtr, ytr = decode(fetch("train"))
    Xte, yte = decode(fetch("test"))
    arrays = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    _verify("cifar10", arrays)                             # gate BEFORE the cache exists
    tmp = npz + f".tmp.{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    print(f"[data] wrote {npz} ({os.path.getsize(npz)} bytes), hashes verified.", flush=True)
    return npz


def _load_cifar_raw():
    """The cached raw arrays as ``(Xtr u8, ytr i64, Xte, yte)``, hash-verified on every load."""
    npz = os.path.join(cache_dir(), "cifar10_raw.npz")
    if not os.path.exists(npz):
        raise RuntimeError(f"[data] {npz} missing; run `corrected-lds build-data --dataset "
                           f"cifar10` first (never load CIFAR from torchvision/Toronto — "
                           f"different row order).")
    z = np.load(npz)
    _verify("cifar10", {k: z[k] for k in ("Xtr", "ytr", "Xte", "yte")})
    return z["Xtr"], z["ytr"].astype(np.int64), z["Xte"], z["yte"].astype(np.int64)


def load_cifar(n_train=50_000, n_targets=2_000, seed=0, device="cuda") -> dict:
    """Loader contract: a dict of device tensors + ``meta``.

        X_train (n_train, 3, 32, 32), y_train (n_train,)
        X_query (n_targets, 3, 32, 32), y_query (n_targets,)   -- the LDS target set

    Normalized with the torchvision CIFAR constants x255 on raw-255 floats.  The seeded draw
    order is the contract (shared with the text loader): one ``default_rng(seed)`` draws the
    train subset first (only when ``n_train`` is below the split size), then the queries, so
    ``query_idx`` is a deterministic function of ``(seed, n_train, n_targets)``."""
    import torch
    Xtr, ytr, Xte, yte = _load_cifar_raw()                  # u8, canonical order, verified
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)   # exact u8 -> f32 0-255
    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(ytr))
    if n_train < len(ytr):
        keep = rng.choice(len(ytr), size=int(n_train), replace=False)
        train_idx = train_idx[keep]
        Xtr, ytr = Xtr[keep], ytr[keep]
    mean = (_CIFAR_MEAN01 * 255.0).reshape(1, 3, 1, 1)
    std = (_CIFAR_STD01 * 255.0).reshape(1, 3, 1, 1)
    Xtr = (Xtr - mean) / std
    q_idx = rng.choice(len(yte), size=min(int(n_targets), len(yte)), replace=False)
    Xq = (Xte[q_idx] - mean) / std
    yq = yte[q_idx]
    train_hash, query_hash = bank_hashes("cifar10")

    def to(a, d):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=d)

    return {
        "X_train": to(Xtr, torch.float32), "y_train": to(ytr, torch.long),
        "X_query": to(Xq, torch.float32), "y_query": to(yq, torch.long),
        "meta": {"n_train": len(ytr), "n_targets": len(yq), "n_classes": 10,
                 "in_ch": int(Xtr.shape[1]), "seed": int(seed), "dataset": "cifar10",
                 "train_idx": train_idx.astype(int).tolist(),
                 "query_idx": q_idx.astype(int).tolist(),
                 "train_hash": train_hash, "query_hash": query_hash,
                 "normalize": "torchvision_cifar10"},
    }


# -- MNIST: raw uint8 cache in a hash-pinned HF-parquet row order ----------- #
_MNIST_REPO = "ylecun/mnist"
_MNIST_FILES = {
    "train": "mnist/train-00000-of-00001.parquet",
    "test": "mnist/test-00000-of-00001.parquet",
}
_MNIST_MEAN, _MNIST_STD = np.float32(0.1307), np.float32(0.3081)


def build_mnist_cache() -> str:
    """Download and decode the pinned public MNIST parquet files to ``(N,1,28,28)`` uint8
    images and int64 labels; exact content hashes pin the upstream row order."""
    npz = os.path.join(cache_dir(), "mnist_raw.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        _verify("mnist", {k: z[k] for k in ("Xtr", "ytr", "Xte", "yte")})
        print(f"[data] {npz} already present and verified.", flush=True)
        return npz

    import pyarrow.parquet as pq
    from PIL import Image

    def fetch(split):
        dst = os.path.join(cache_dir(), f"mnist_{split}.parquet")
        if not os.path.exists(dst) or os.path.getsize(dst) < 1_000_000:
            name = _MNIST_FILES[split]
            url = f"https://huggingface.co/datasets/{_MNIST_REPO}/resolve/main/{name}"
            print(f"[data] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, dst)
        return dst

    def decode(path):
        table = pq.read_table(path, columns=["image", "label"])
        labels = table.column("label").to_numpy().astype(np.int64)
        images = np.empty((len(labels), 1, 28, 28), dtype=np.uint8)
        for i, value in enumerate(table.column("image").to_pylist()):
            raw = value["bytes"] if isinstance(value, dict) else value
            image = Image.open(io.BytesIO(raw)).convert("L")
            array = np.asarray(image, dtype=np.uint8)
            if array.shape != (28, 28):
                raise RuntimeError(f"[data] MNIST row {i} decoded as {array.shape}, expected 28x28")
            images[i, 0] = array
        return images, labels

    Xtr, ytr = decode(fetch("train"))
    Xte, yte = decode(fetch("test"))
    arrays = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    _verify("mnist", arrays)
    tmp = npz + f".tmp.{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    print(f"[data] wrote {npz} ({os.path.getsize(npz)} bytes), hashes verified.", flush=True)
    return npz


def load_mnist(n_train=6_144, n_targets=2_000, seed=0, device="cuda") -> dict:
    """Load the deterministic SOURCE-style MNIST subset and test queries."""
    import torch

    npz = os.path.join(cache_dir(), "mnist_raw.npz")
    if not os.path.exists(npz):
        raise RuntimeError(f"[data] {npz} missing; run `corrected-lds build-data --dataset "
                           f"mnist` first.")
    z = np.load(npz)
    arrays = {k: z[k] for k in ("Xtr", "ytr", "Xte", "yte")}
    _verify("mnist", arrays)
    Xtr, ytr, Xte, yte = (arrays["Xtr"], arrays["ytr"],
                           arrays["Xte"], arrays["yte"])
    n_train = min(int(n_train), len(ytr))
    n_targets = min(int(n_targets), len(yte))
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(ytr), size=n_train, replace=False)
    query_idx = rng.choice(len(yte), size=n_targets, replace=False)
    scale = np.float32(255.0)
    Xtr = (Xtr[train_idx].astype(np.float32) / scale - _MNIST_MEAN) / _MNIST_STD
    Xq = (Xte[query_idx].astype(np.float32) / scale - _MNIST_MEAN) / _MNIST_STD
    train_hash, query_hash = bank_hashes("mnist")

    def to(a, dtype):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dtype)

    return {
        "X_train": to(Xtr, torch.float32), "y_train": to(ytr[train_idx], torch.long),
        "X_query": to(Xq, torch.float32), "y_query": to(yte[query_idx], torch.long),
        "meta": {"n_train": n_train, "n_targets": n_targets, "n_classes": 10,
                 "in_ch": 1, "seed": int(seed), "dataset": "mnist",
                 "train_idx": train_idx.astype(int).tolist(),
                 "query_idx": query_idx.astype(int).tolist(),
                 "train_hash": train_hash, "query_hash": query_hash,
                 "normalize": "mnist_mean_0.1307_std_0.3081"},
    }


# -- QNLI (GLUE): tokenized uint16/uint8 cache ------------------------------ #
QNLI_TRAIN, QNLI_VAL = 104_743, 5_463          # GLUE QNLI split sizes (labelled)


def build_qnli_cache(seq_len=128, model_name="bert-base-cased") -> str:
    """Tokenize the full QNLI train (104,743) + validation (5,463) splits once at ``seq_len``,
    verify the pins on every array (tokens too: a drifted tokenizer cannot pass on labels
    alone) and atomically cache to npz (uint16 ids / uint8 masks).  Idempotent."""
    cache = os.path.join(cache_dir(), f"qnli_{model_name.replace('/', '_')}_seq{seq_len}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        _verify("qnli", {k: z[k] for k in z.files})
        print(f"[data] {cache} already present and verified.", flush=True)
        return cache
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("glue", "qnli")
    out = {}
    for split, key in (("train", "train"), ("val", "validation")):
        d = ds[key]
        enc = tok(list(d["question"]), list(d["sentence"]), max_length=seq_len,
                  truncation=True, padding="max_length")
        out[f"{split}_ids"] = np.asarray(enc["input_ids"], dtype=np.uint16)
        out[f"{split}_tok"] = np.asarray(enc["token_type_ids"], dtype=np.uint8)
        out[f"{split}_att"] = np.asarray(enc["attention_mask"], dtype=np.uint8)
        out[f"{split}_y"] = np.asarray(d["label"], dtype=np.int64)
    _verify("qnli", out)
    # per-process temp + rename: concurrent builders never read a half-written file
    tmp = f"{cache}.tmp.{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, cache)
    print(f"[data] wrote {cache}, label hashes verified.", flush=True)
    return cache


def load_qnli(n_train=50_000, n_targets=2_000, seed=0, device="cuda", seq_len=128,
              model_name="bert-base-cased") -> dict:
    """Text counterpart of :func:`load_cifar` with the same draw order (train subset first,
    then the validation queries) and every pin (tokens + labels) verified on every load:

        input_ids / token_type_ids / attention_mask (n_train, seq_len) long, labels
        q_* counterparts (the LDS query set), meta"""
    import torch
    cache = os.path.join(cache_dir(), f"qnli_{model_name.replace('/', '_')}_seq{seq_len}.npz")
    if not os.path.exists(cache):
        raise RuntimeError(f"[data] {cache} missing; run `corrected-lds build-data --dataset "
                           f"qnli` first.")
    z = np.load(cache)
    a = {k: z[k] for k in z.files}
    _verify("qnli", a)
    n_train = min(int(n_train), a["train_ids"].shape[0])
    n_targets = min(int(n_targets), a["val_ids"].shape[0])

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(a["train_ids"].shape[0], size=n_train, replace=False)
    q_idx = rng.choice(a["val_ids"].shape[0], size=n_targets, replace=False)
    train_hash, query_hash = bank_hashes("qnli")

    def to_long(arr, idx):
        return torch.as_tensor(np.ascontiguousarray(arr[idx]).astype(np.int64),
                               device=device, dtype=torch.long)

    return {
        "input_ids": to_long(a["train_ids"], train_idx),
        "token_type_ids": to_long(a["train_tok"], train_idx),
        "attention_mask": to_long(a["train_att"], train_idx),
        "labels": to_long(a["train_y"], train_idx),
        "q_input_ids": to_long(a["val_ids"], q_idx),
        "q_token_type_ids": to_long(a["val_tok"], q_idx),
        "q_attention_mask": to_long(a["val_att"], q_idx),
        "q_labels": to_long(a["val_y"], q_idx),
        "meta": {"modality": "text", "dataset": "qnli", "n_train": int(n_train),
                 "n_targets": int(n_targets), "n_classes": 2, "seq_len": int(seq_len),
                 "model": str(model_name), "seed": int(seed),
                 "train_idx": train_idx.astype(int).tolist(),
                 "query_idx": q_idx.astype(int).tolist(),
                 "train_hash": train_hash, "query_hash": query_hash},
    }


# -- ImageNet (imagenet-1k): sanitized-JPEG blob cache from the pinned HF parquet --- #
IMAGENET_TRAIN, IMAGENET_VAL, IMAGENET_CLASSES = 1_281_167, 50_000, 1_000
_IMAGENET_REPO = "ILSVRC/imagenet-1k"
# The parquet-native main branch (the 2025 Hub migration), pinned so a future reshard
# cannot silently reorder rows; content pins remain the ultimate gate.  Layout verified
# 2026-08-16 via the public tree API.
_IMAGENET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
_IMAGENET_SHARDS = {
    "train": [f"data/train-{i:05d}-of-00294.parquet" for i in range(294)],
    "val": [f"data/validation-{i:05d}-of-00014.parquet" for i in range(14)],
}
_IMAGENET_MAX_SIDE, _IMAGENET_JPEG_Q = 500, 90     # the reference writer's 500px / q90
_GATE_WINDOW = 16 * 2**20                          # structural blob gate window (16 MB)
IMAGENET_MEAN255 = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD255 = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _imagenet_files():
    d = cache_dir()
    files = {s: {"blob": os.path.join(d, f"imagenet_{s}_jpegs.bin"),
                 "offsets": os.path.join(d, f"imagenet_{s}_offsets.npy")}
             for s in ("train", "val")}
    files["meta"] = os.path.join(d, "imagenet_meta.npz")
    return files


def _hub_download(filename):
    """Download one gated file at the pinned revision; Hub errors (gated repo, 401/403) exit
    with the accept-the-license recipe, anything else (disk, misconfig) propagates."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        return hf_hub_download(_IMAGENET_REPO, filename, repo_type="dataset",
                               revision=_IMAGENET_REVISION)
    except (GatedRepoError, HfHubHTTPError) as e:
        raise SystemExit(
            f"[data] cannot download {filename} from {_IMAGENET_REPO}: "
            f"{str(e).splitlines()[0][:120]}\n"
            f"  ImageNet is license-gated: accept the terms once at "
            f"https://huggingface.co/datasets/{_IMAGENET_REPO},\n"
            f"  then set HF_TOKEN (or `hf auth login`) and re-run.") from e


def _iter_split_rows(split):
    """Yield ``(source_path, label, raw_image_bytes)`` in the CANONICAL row order:
    parquet shard 00000..end at the pinned revision, rows within each shard."""
    import pyarrow.parquet as pq
    for shard in _IMAGENET_SHARDS[split]:
        pf = pq.ParquetFile(_hub_download(shard))
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for img, lab in zip(batch.column("image").to_pylist(),
                                batch.column("label").to_pylist()):
                yield img["path"], int(lab), img["bytes"]


def smoke_imagenet_source(rows_per_split=512) -> dict:
    """Pre-build check with no cache writes: download one parquet shard per split (~1 GB of
    ~153 GB) and verify the builder's assumptions on real bytes (auth, pinned revision,
    ``image{bytes, path}`` / ``label`` schema, label range, :func:`_sanitize_jpeg`).  Returns
    the per-split report: per-image cost, extrapolated blob size and re-encode core-hours."""
    import time

    import pyarrow.parquet as pq
    report = {}
    for split, n_total in (("train", IMAGENET_TRAIN), ("val", IMAGENET_VAL)):
        pf = pq.ParquetFile(_hub_download(_IMAGENET_SHARDS[split][0]))
        names = set(pf.schema_arrow.names)
        if not {"image", "label"} <= names:
            raise RuntimeError(f"[data] imagenet {split} shard schema {sorted(names)} "
                               f"lacks image/label — the builder's assumptions do not "
                               f"hold; do NOT run the full build.")
        seen, out_bytes, labs = 0, 0, []
        t0 = time.monotonic()
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for img, lab in zip(batch.column("image").to_pylist(),
                                batch.column("label").to_pylist()):
                if not (isinstance(img.get("bytes"), (bytes, bytearray)) and img.get("path")):
                    raise RuntimeError(f"[data] imagenet {split} image struct lacks "
                                       f"bytes/path ({sorted(img)}); do NOT run the "
                                       f"full build.")
                out_bytes += len(_sanitize_jpeg(img["bytes"]))
                labs.append(int(lab))
                seen += 1
                if seen >= rows_per_split:
                    break
            if seen >= rows_per_split:
                break
        dt = time.monotonic() - t0
        if min(labs) < 0 or max(labs) >= IMAGENET_CLASSES:
            raise RuntimeError(f"[data] imagenet {split} labels outside "
                               f"[0, {IMAGENET_CLASSES}) in the smoke sample; do NOT "
                               f"run the full build.")
        report[split] = {
            "shard_rows": int(pf.metadata.num_rows), "rows_checked": seen,
            "ms_per_image": 1e3 * dt / seen,
            "est_blob_gb": out_bytes / seen * n_total / 1e9,
            "est_encode_core_hours": dt / seen * n_total / 3600,
        }
        r = report[split]
        print(f"[data] smoke {split}: shard 0 has {r['shard_rows']} rows; checked "
              f"{seen} — sanitize {r['ms_per_image']:.1f} ms/image (1 core), "
              f"est. blob {r['est_blob_gb']:.1f} GB, "
              f"est. re-encode {r['est_encode_core_hours']:.1f} core-hours", flush=True)
    print("[data] smoke OK — schema, labels, and sanitize verified on real shards; "
          "safe to run the full build.", flush=True)
    return report


def _sanitize_jpeg(raw: bytes) -> bytes:
    """Re-encode any image as a baseline-RGB JPEG at quality 90, downscaled to max side 500
    when larger (BOX ~ the reference writer's cv2 INTER_AREA).  RGB conversion absorbs
    ImageNet's grayscale/CMYK oddballs, so the blob is safe for batched nvjpeg decode."""
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > _IMAGENET_MAX_SIDE:
        scale = _IMAGENET_MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BOX)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=_IMAGENET_JPEG_Q)
    return out.getvalue()


def _chunked(it, n):
    it = iter(it)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def _assemble_split(split, rows, workers):
    """Re-encode ``rows`` in order into ``imagenet_<split>_jpegs.bin`` + its offsets file,
    atomically (``workers=0`` runs in-process).  Returns ``(labels int64, manifest sha256/16
    over the '|'-joined source paths)``."""
    import concurrent.futures as cf
    f = _imagenet_files()[split]
    tmp_blob = f["blob"] + f".tmp.{os.getpid()}"
    tmp_offsets = f["offsets"] + f".tmp.{os.getpid()}"
    labels, offsets, manifest = [], [0], hashlib.sha256()
    pool = cf.ProcessPoolExecutor(workers) if workers else None
    try:
        with open(tmp_blob, "wb") as out:
            for chunk in _chunked(rows, 512):
                paths, labs, raws = zip(*chunk)
                encoded = (list(pool.map(_sanitize_jpeg, raws, chunksize=16)) if pool
                           else [_sanitize_jpeg(r) for r in raws])
                for path, lab, enc in zip(paths, labs, encoded):
                    manifest.update(str(path).encode())
                    manifest.update(b"|")
                    out.write(enc)
                    offsets.append(offsets[-1] + len(enc))
                    labels.append(int(lab))
                if len(labels) % 102_400 < 512:
                    print(f"[data] imagenet {split}: {len(labels)} images re-encoded",
                          flush=True)
    finally:
        if pool:
            pool.shutdown()
    with open(tmp_offsets, "wb") as fh:
        np.save(fh, np.asarray(offsets, dtype=np.int64))
    os.replace(tmp_blob, f["blob"])
    os.replace(tmp_offsets, f["offsets"])
    return np.asarray(labels, dtype=np.int64), manifest.hexdigest()[:16]


def _sha16_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()[:16]


def _blob_gate(path) -> str:
    """Structural blob gate, checked on every load: sha256/16 over the exact byte size plus
    three fixed 16 MB windows (head / middle / tail).  It catches truncation or replacement in
    <1 s where a full ~70 GB hash takes minutes; the full hash is checked by ``build-data``."""
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as fh:
        for start in (0, max(0, size // 2 - _GATE_WINDOW // 2), max(0, size - _GATE_WINDOW)):
            fh.seek(start)
            h.update(fh.read(_GATE_WINDOW))
    return h.hexdigest()[:16]


def imagenet_cache_pins(full=False) -> dict:
    """The pin dict of the cache on disk, as the builder prints it for ``presets.py`` and every
    gate compares it: label digests, stored manifest digests, structural blob gates and, with
    ``full=True``, the full blob sha256/16s."""
    files = _imagenet_files()
    z = np.load(files["meta"], allow_pickle=False)
    got = {"train_y": _sha16(z["train_y"]), "val_y": _sha16(z["val_y"]),
           "train_manifest": str(z["train_manifest"]),
           "val_manifest": str(z["val_manifest"]),
           "train_blob_gate": _blob_gate(files["train"]["blob"]),
           "val_blob_gate": _blob_gate(files["val"]["blob"])}
    if full:
        got["train_blob"] = _sha16_file(files["train"]["blob"])
        got["val_blob"] = _sha16_file(files["val"]["blob"])
    return got


def _check_imagenet_pins(got):
    """Compare ``got`` against the recorded pins.  With none recorded, print the block for
    ``presets.DATA_HASHES`` and refuse: pins come from a real build, never placeholders."""
    pins = DATA_HASHES.get("imagenet")
    if pins is None:
        block = ",\n".join(f'                 "{k}": "{v}"' for k, v in sorted(got.items()))
        raise RuntimeError(
            "[data] imagenet content-hash pins are not recorded yet.  The cache is built; "
            "paste into corrected_lds/banks/presets.py DATA_HASHES:\n"
            '    "imagenet": {\n' + block + "},\n"
            "then re-run `corrected-lds build-data --dataset imagenet` to verify.")
    bad = [f"{k}: got {got[k]}, pinned {pins.get(k)}" for k in got if got[k] != pins.get(k)]
    if bad:
        raise RuntimeError(
            "[data] imagenet content-hash mismatch — the cache does not match the "
            "pinned hashes, and every mask/pool/tauq artifact would be silently "
            "corrupted: " + "; ".join(bad))


def build_imagenet_cache(workers=None) -> str:
    """Build, or fully re-verify, the canonical ImageNet cache: stream the pinned-revision,
    license-gated HF parquet shards, re-encode every image into the two JPEG blobs, check
    the row totals, exactly 1000 classes and <=1300 per class, and verify the pins (or print
    them on a first build).  A rerun on an existing cache recomputes every pin, including the
    full blob hashes.  Disk: ~153 GB of parquet in the hub cache + ~75 GB of blobs."""
    files = _imagenet_files()
    have = all(os.path.exists(p) for p in
               (files["meta"], files["train"]["blob"], files["train"]["offsets"],
                files["val"]["blob"], files["val"]["offsets"]))
    if have:
        _check_imagenet_pins(imagenet_cache_pins(full=True))
        print(f"[data] {files['meta']} already present; blobs fully re-verified.",
              flush=True)
        return files["meta"]
    if workers is None:
        workers = os.cpu_count() or 1
    side = {}
    for split, n_expect in (("train", IMAGENET_TRAIN), ("val", IMAGENET_VAL)):
        y, manifest = _assemble_split(split, _iter_split_rows(split), workers)
        if len(y) != n_expect:
            raise RuntimeError(f"[data] imagenet {split}: {len(y)} rows != the "
                               f"canonical {n_expect}; wrong source or revision.")
        side[split] = (y, manifest)
    train_y, val_y = side["train"][0], side["val"][0]
    counts = np.bincount(train_y, minlength=IMAGENET_CLASSES)
    if len(np.unique(train_y)) != IMAGENET_CLASSES or counts.max() > 1_300:
        raise RuntimeError(f"[data] imagenet train labels are off: "
                           f"{len(np.unique(train_y))} classes, max/class "
                           f"{counts.max()} (expect {IMAGENET_CLASSES} and <=1300).")
    if val_y.min() < 0 or val_y.max() >= IMAGENET_CLASSES:
        raise RuntimeError("[data] imagenet val labels outside [0, 1000).")
    import PIL
    tmp = files["meta"] + f".tmp.{os.getpid()}"
    np.savez(tmp, train_y=train_y, val_y=val_y,
             train_manifest=side["train"][1], val_manifest=side["val"][1],
             source_revision=_IMAGENET_REVISION, pillow_version=PIL.__version__)
    os.replace(tmp + ".npz", files["meta"])
    _check_imagenet_pins(imagenet_cache_pins(full=True))
    print(f"[data] wrote {files['meta']}; all content-hash pins verified.", flush=True)
    return files["meta"]


def _decode_center_crop(jpeg: bytes, val_res: int, crop_ratio: float) -> np.ndarray:
    """The reference val-decoder geometry (ffcv ``CenterCropRGBImageDecoder``): the centered
    square of side ``round(crop_ratio * min(h, w))`` resized to ``(val_res, val_res)``, not
    torchvision's resize-256/crop-224, with PIL bilinear (unlike the reference kernel).
    Returns ``(3, val_res, val_res)`` float32 on the raw-255 scale."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    s = max(1, round(crop_ratio * min(w, h)))
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((val_res, val_res),
                                                          Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)


def load_imagenet(n_train=IMAGENET_TRAIN, n_targets=2_000, seed=0, device="cuda",
                  val_res=256, crop_ratio=224 / 256) -> dict:
    """Streaming variant of the loader contract: the train side is a memmapped JPEG blob.

        jpegs (np.memmap u8), offsets (n+1,) i64   -- the sanitized train blob
        train_rows (n_train,) i64                  -- physical blob rows of the train
                                                      universe (identity at full n)
        y_train (n_train,) long on ``device``
        X_query (n_targets, 3, val_res, val_res) f32 on ``device``, normalized
        y_query, meta                              -- the LDS target set

    Same seeded draw order as every loader: train subset first (only when ``n_train`` is
    below the split size), then the queries.  Every load checks the label pins, manifest
    digests and blob gates against the recorded pins.  ``val_res`` / ``crop_ratio`` come from
    the recipe, so margins and TRAK scoring see byte-identical query tensors."""
    import torch
    files = _imagenet_files()
    if not os.path.exists(files["meta"]):
        raise RuntimeError(f"[data] {files['meta']} missing; run `corrected-lds "
                           f"build-data --dataset imagenet` first (license-gated HF "
                           f"download — the builder prints instructions).")
    if "imagenet" not in DATA_HASHES:
        raise RuntimeError("[data] imagenet content-hash pins are not recorded in presets.py; "
                           "finish `build-data --dataset imagenet` first (it prints "
                           "the block to paste).")
    _check_imagenet_pins(imagenet_cache_pins(full=False))
    z = np.load(files["meta"], allow_pickle=False)
    train_y_all = z["train_y"].astype(np.int64)
    val_y = z["val_y"].astype(np.int64)
    jpegs = np.memmap(files["train"]["blob"], dtype=np.uint8, mode="r")
    offsets = np.load(files["train"]["offsets"])
    val_jpegs = np.memmap(files["val"]["blob"], dtype=np.uint8, mode="r")
    val_offsets = np.load(files["val"]["offsets"])

    n_all = len(train_y_all)
    n_train = min(int(n_train), n_all)
    rng = np.random.default_rng(seed)
    train_rows = np.arange(n_all, dtype=np.int64)
    if n_train < n_all:                                 # train subset drawn FIRST
        train_rows = rng.choice(n_all, size=n_train, replace=False).astype(np.int64)
    q_idx = rng.choice(len(val_y), size=min(int(n_targets), len(val_y)), replace=False)

    Xq = np.stack([_decode_center_crop(
        bytes(val_jpegs[val_offsets[i]:val_offsets[i + 1]]), val_res, crop_ratio)
        for i in q_idx])
    Xq = (Xq - IMAGENET_MEAN255.reshape(1, 3, 1, 1)) / IMAGENET_STD255.reshape(1, 3, 1, 1)
    train_hash, query_hash = bank_hashes("imagenet")
    return {
        "jpegs": jpegs, "offsets": offsets, "train_rows": train_rows,
        "y_train": torch.as_tensor(train_y_all[train_rows], device=device,
                                   dtype=torch.long),
        "X_query": torch.as_tensor(np.ascontiguousarray(Xq), device=device,
                                   dtype=torch.float32),
        "y_query": torch.as_tensor(val_y[q_idx], device=device, dtype=torch.long),
        "meta": {"n_train": int(n_train), "n_targets": len(q_idx),
                 "n_classes": int(train_y_all.max()) + 1, "in_ch": 3,
                 "seed": int(seed), "dataset": "imagenet",
                 "train_idx": train_rows.astype(int).tolist(),
                 "query_idx": q_idx.astype(int).tolist(),
                 "train_hash": train_hash, "query_hash": query_hash,
                 "val_res": int(val_res), "crop_ratio": float(crop_ratio),
                 "normalize": "torchvision_imagenet"},
    }


# -- WikiText-2 (bergson's Bae-replication chunks): int64 token-id blocks --- #
# wikitext-2-raw-v1, GPT-2 tokenizer, HF run_clm group_texts into 512-token blocks, as
# hosted for bergson's examples/replicate_bae_approx_unrolling_source.  The REVISION is
# the identity: bergson's loader reads the same one (pools/bergson.py passes it through
# data.data_kwargs), so the pool's rows and these rows are the same bytes.
WIKITEXT2_REPO = "EleutherAI/bergson-wikitext-2-4656-chunks"
WIKITEXT2_REVISION = "971da0e6da181b9e147081087f1d3fd51669024c"
WIKITEXT2_TRAIN, WIKITEXT2_VAL, WIKITEXT2_SEQ = 4_656, 481, 512


def build_wikitext2_cache() -> str:
    """Cache the pinned train/validation ``input_ids`` as verified int64 ``(n, 512)`` arrays."""
    npz = os.path.join(cache_dir(), "wikitext2_ids.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        _verify("wikitext2", {k: z[k] for k in ("train_ids", "val_ids")})
        print(f"[data] {npz} already present and verified.", flush=True)
        return npz

    import pyarrow.parquet as pq

    def fetch(split):
        dst = os.path.join(cache_dir(), f"wikitext2_{split}.parquet")
        if not os.path.exists(dst):
            url = (f"https://huggingface.co/datasets/{WIKITEXT2_REPO}/resolve/"
                   f"{WIKITEXT2_REVISION}/data/{split}-00000-of-00001.parquet")
            print(f"[data] downloading {url}", flush=True)
            tmp = dst + f".tmp.{os.getpid()}"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
        return dst

    def ids(path, n):
        a = np.asarray(pq.read_table(path, columns=["input_ids"]).column("input_ids")
                       .to_pylist(), dtype=np.int64)
        if a.shape != (n, WIKITEXT2_SEQ):
            raise RuntimeError(f"[data] {path} decoded as {a.shape}, expected "
                               f"({n}, {WIKITEXT2_SEQ})")
        return a

    arrays = {"train_ids": ids(fetch("train"), WIKITEXT2_TRAIN),
              "val_ids": ids(fetch("validation"), WIKITEXT2_VAL)}
    _verify("wikitext2", arrays)
    tmp = npz + f".tmp.{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    print(f"[data] wrote {npz} ({os.path.getsize(npz)} bytes), hashes verified.", flush=True)
    return npz


def load_wikitext2(n_train=WIKITEXT2_TRAIN, n_targets=WIKITEXT2_VAL, seed=0,
                   device="cuda") -> dict:
    """The whole train and validation splits, in their hosted order, never subsampled.

    bergson's pool indexes documents by their row in the hosted split (``subsets.json``
    lists row ids; the queries are ``validation[0:Q]``), so any draw here would mis-pair
    every tauq with the pool; other sizes are refused.  ``seed`` is only recorded."""
    if (int(n_train), int(n_targets)) != (WIKITEXT2_TRAIN, WIKITEXT2_VAL):
        raise ValueError(f"[data] wikitext2 is used whole: n_train={WIKITEXT2_TRAIN}, "
                         f"n_targets={WIKITEXT2_VAL} (got {n_train}, {n_targets}).")
    import torch

    npz = os.path.join(cache_dir(), "wikitext2_ids.npz")
    if not os.path.exists(npz):
        raise RuntimeError(f"[data] {npz} missing; run `corrected-lds build-data --dataset "
                           f"wikitext2` first.")
    z = np.load(npz)
    arrays = {k: z[k] for k in ("train_ids", "val_ids")}
    _verify("wikitext2", arrays)
    train_hash, query_hash = bank_hashes("wikitext2")
    return {
        # NOT "input_ids": that key routes the attributors onto the BERT text path
        "lm_ids": torch.as_tensor(arrays["train_ids"], device=device, dtype=torch.long),
        "q_lm_ids": torch.as_tensor(arrays["val_ids"], device=device, dtype=torch.long),
        "meta": {"n_train": WIKITEXT2_TRAIN, "n_targets": WIKITEXT2_VAL,
                 "seq_len": WIKITEXT2_SEQ, "seed": int(seed), "dataset": "wikitext2",
                 "train_idx": list(range(WIKITEXT2_TRAIN)),
                 "query_idx": list(range(WIKITEXT2_VAL)),
                 "train_hash": train_hash, "query_hash": query_hash,
                 "revision": WIKITEXT2_REVISION},
    }
