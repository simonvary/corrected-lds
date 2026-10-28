"""Release promises, checked offline: bank identity hashes are pinned, ``banks-manifest.json``
is well formed and lists only bank files, and the docs name only released files.  A moved
identity pin is a release-breaking bug, never a test to update.
"""

import json
import os
import re

import pytest

from corrected_lds.banks.presets import DATA_HASHES, PRESETS, bank_hashes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(REPO, "banks-manifest.json")) as fh:
    MANIFEST = json.load(fh)
BANKS = MANIFEST["banks"]
NAMES = {bank: set(files) for bank, files in BANKS.items()}

# (train_hash, query_hash) stamped in each released pool.npz.
RELEASED_HASHES = {
    "cifar10": ("0c79d6744e51f344", "72eeeba5111be6e3"),
    "mnist": ("87527ae5ec7c40f4", "050ef482aca3b534"),
    "qnli": ("1615f1b2274e6f81", "c6a11dfc0feaeb7f"),
    "wikitext2": ("03091dbab9d466e3", "d374d5b0d1ccc2a4"),
    "imagenet": ("a61a688ff7605f73", "c39190b20ebfb155"),
}
URL = re.compile(r"^https://huggingface\.co/datasets/corrected-lds/data/resolve/[0-9a-f]{40}$")
BANK_FILE = re.compile(r"^(pool|pool_[A-Za-z0-9_]+|tauq_[A-Za-z0-9_]+)\.npz$")


@pytest.mark.parametrize("dataset", sorted(RELEASED_HASHES))
def test_bank_hashes_are_pinned(dataset):
    assert bank_hashes(dataset) == RELEASED_HASHES[dataset]


def test_manifest_is_well_formed():
    assert MANIFEST["format"] == 2 and URL.match(MANIFEST["url"]), MANIFEST["url"]
    assert set(BANKS) == {"cifar10", "mnist", "qnli", "wikitext2_gpt2_bae", "imagenet"}
    for bank, files in BANKS.items():
        assert "pool.npz" in files, bank
        for name, sha256 in files.items():
            assert BANK_FILE.match(name), f"{bank}/{name}: not a bank file name"
            assert re.fullmatch(r"[0-9a-f]{64}", sha256), f"{bank}/{name}: sha256"


def test_every_bank_has_a_preset_with_pinned_data():
    # A dataset missing from DATA_HASHES would silently skip the loaders' hash check.
    for bank in BANKS:
        assert bank in PRESETS, bank
        assert PRESETS[bank]["dataset"] in DATA_HASHES, bank


def _expand(token):
    """``tauq_trak5_e{2,4}.npz`` -> {``tauq_trak5_e2.npz``, ``tauq_trak5_e4.npz``}."""
    m = re.search(r"\{([0-9,]+)\}", token)
    if not m:
        return {token}
    return set().union(*(_expand(token[:m.start()] + v + token[m.end():]) for v in m[1].split(",")))


def _card_trees():
    with open(os.path.join(REPO, "docs", "hf-dataset-card.md")) as fh:
        layout = fh.read().split("## Layout", 1)[1].split("```", 2)[1]
    trees, bank = {}, None
    for line in layout.splitlines():
        head = re.match(r"([A-Za-z0-9_]+)/(\s|$)", line)
        if head:
            bank = head[1]
            trees[bank] = set()
        elif bank and line.startswith(("├──", "└──")):
            for token in re.findall(r"[A-Za-z0-9_]*(?:\{[0-9,]+\}[A-Za-z0-9_]*)*\.npz", line):
                trees[bank] |= _expand(token)
    return trees


def test_dataset_card_lists_exactly_the_released_files():
    trees = _card_trees()
    assert set(trees) == set(BANKS)
    for bank in BANKS:
        assert trees[bank] == NAMES[bank], (
            f"{bank}: card-only {sorted(trees[bank] - NAMES[bank])}, "
            f"manifest-only {sorted(NAMES[bank] - trees[bank])}")


@pytest.mark.parametrize("doc", ["README.md", "docs/REPRODUCE.md", "docs/FORMATS.md"])
def test_docs_name_only_released_files(doc):
    with open(os.path.join(REPO, doc)) as fh:
        text = fh.read()
    for bank, key in re.findall(r"evaluate banks/([A-Za-z0-9_]+) --tauq ([A-Za-z0-9_]+)", text):
        assert f"tauq_{key}.npz" in NAMES[bank], f"{doc}: banks/{bank} has no tauq_{key}.npz"
    released = set().union(*NAMES.values())
    pattern = r"\b(?:tauq|pool_tda)_(?:trak\d+|ekfac|tracin|source_L\d+)(?:_e\d+)?\.npz"
    for name in re.findall(pattern, text):
        assert name in released, f"{doc} names {name}, which no bank releases"
