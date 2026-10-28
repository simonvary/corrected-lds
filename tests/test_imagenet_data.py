"""ImageNet cache, network-free (needs PIL and torch): with the HF rows stubbed, the build and
load path runs for real on synthetic data; only ``build-data`` exercises the gated download.
"""

import io
import os

import numpy as np
import pytest

pytest.importorskip("PIL")
torch = pytest.importorskip("torch")

from PIL import Image

import corrected_lds.pools.datasets as ds
from corrected_lds.banks import presets

TRAIN_ROWS, VAL_ROWS, CLASSES = 8, 4, 2


def _jpeg(color, size=(600, 400), mode="RGB"):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    """Build the full cache from stubbed rows; record the printed pins; re-verify."""
    monkeypatch.setenv("CORRECTED_LDS_DATA", str(tmp_path))
    monkeypatch.setattr(ds, "IMAGENET_TRAIN", TRAIN_ROWS)
    monkeypatch.setattr(ds, "IMAGENET_VAL", VAL_ROWS)
    monkeypatch.setattr(ds, "IMAGENET_CLASSES", CLASSES)
    rows = {"train": [(f"t{i}.JPEG", i % CLASSES, _jpeg((10 * i, 20, 30)))
                      for i in range(TRAIN_ROWS)],
            "val": [(f"v{i}.JPEG", i % CLASSES, _jpeg((40, 10 * i, 60), size=(300, 200)))
                    for i in range(VAL_ROWS)]}
    monkeypatch.setattr(ds, "_iter_split_rows", lambda split: iter(rows[split]))
    # the committed imagenet pins do not fit the synthetic data
    monkeypatch.delitem(presets.DATA_HASHES, "imagenet", raising=False)
    # first build writes the cache, prints the pins in its error, and refuses
    with pytest.raises(RuntimeError, match="pins are not recorded"):
        ds.build_imagenet_cache(workers=0)
    pins = ds.imagenet_cache_pins(full=True)
    monkeypatch.setitem(presets.DATA_HASHES, "imagenet", pins)
    ds.build_imagenet_cache(workers=0)      # idempotent rerun = the full verify
    return rows


def test_build_writes_canonical_blobs(synthetic_cache):
    files = ds._imagenet_files()
    offsets = np.load(files["train"]["offsets"])
    assert offsets.shape == (TRAIN_ROWS + 1,) and offsets[0] == 0
    assert (np.diff(offsets) > 0).all()                    # every image non-empty
    z = np.load(files["meta"], allow_pickle=False)
    assert z["train_y"].tolist() == [i % CLASSES for i in range(TRAIN_ROWS)]
    # every blob entry is a decodable, sanitized JPEG: RGB, max side <= 500
    blob = np.memmap(files["train"]["blob"], dtype=np.uint8, mode="r")
    for i in range(TRAIN_ROWS):
        img = Image.open(io.BytesIO(bytes(blob[offsets[i]:offsets[i + 1]])))
        assert img.format == "JPEG" and img.mode == "RGB"
        assert max(img.size) <= ds._IMAGENET_MAX_SIDE
        assert img.size == (500, 333)                      # 600x400 -> BOX downscale


def test_decode_center_crop_geometry():
    # crop_ratio 0.5 selects exactly the white 50x50 box centred in a black 100x100 image
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    img.paste((255, 255, 255), (25, 25, 75, 75))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    out = ds._decode_center_crop(buf.getvalue(), val_res=32, crop_ratio=0.5)
    assert out.shape == (3, 32, 32) and out.dtype == np.float32
    assert out.mean() > 250                                # white, modulo jpeg fuzz


def test_load_contract_and_query_determinism(synthetic_cache):
    data = ds.load_imagenet(n_train=TRAIN_ROWS, n_targets=3, seed=0, device="cpu",
                            val_res=64, crop_ratio=0.875)
    m = data["meta"]
    assert m["dataset"] == "imagenet" and m["n_classes"] == CLASSES
    assert m["train_idx"] == list(range(TRAIN_ROWS))       # identity at full n
    assert data["X_query"].shape == (3, 3, 64, 64)
    assert data["X_query"].dtype == torch.float32
    assert data["y_query"].tolist() == [i % CLASSES for i in m["query_idx"]]
    assert data["y_train"].tolist() == [i % CLASSES for i in range(TRAIN_ROWS)]
    assert len(m["train_hash"]) == 16 and len(m["query_hash"]) == 16
    again = ds.load_imagenet(n_train=TRAIN_ROWS, n_targets=3, seed=0, device="cpu",
                             val_res=64, crop_ratio=0.875)
    assert again["meta"]["query_idx"] == m["query_idx"]    # seeded draw is the contract
    assert torch.equal(again["X_query"], data["X_query"])
    # the train subset is drawn first when n_train < full (the shared draw-order contract)
    sub = ds.load_imagenet(n_train=5, n_targets=3, seed=0, device="cpu",
                           val_res=64, crop_ratio=0.875)
    assert len(sub["meta"]["train_idx"]) == 5 == len(sub["train_rows"])
    assert sub["y_train"].tolist() == [i % CLASSES for i in sub["meta"]["train_idx"]]


def test_blob_tamper_is_refused_on_load(synthetic_cache):
    files = ds._imagenet_files()
    with open(files["val"]["blob"], "r+b") as fh:          # flip one byte mid-blob
        fh.seek(os.path.getsize(files["val"]["blob"]) // 2)
        b = fh.read(1)
        fh.seek(-1, os.SEEK_CUR)
        fh.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(RuntimeError, match="mismatch"):
        ds.load_imagenet(n_train=TRAIN_ROWS, n_targets=2, device="cpu", val_res=32)


def test_pin_mismatch_and_missing_pins_are_refused(synthetic_cache, monkeypatch):
    bad = dict(presets.DATA_HASHES["imagenet"], train_y="0" * 16)
    monkeypatch.setitem(presets.DATA_HASHES, "imagenet", bad)
    with pytest.raises(RuntimeError, match="mismatch"):
        ds.build_imagenet_cache(workers=0)
    monkeypatch.delitem(presets.DATA_HASHES, "imagenet")
    with pytest.raises(RuntimeError, match="pins are not recorded"):
        ds.load_imagenet(n_train=TRAIN_ROWS, n_targets=2, device="cpu", val_res=32)
