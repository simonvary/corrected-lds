"""compute-tau's provenance builders reproduce what the released files carry; the reuse checks
compare these strings verbatim, so changing one changes artifact identity.  The expected
strings, from the 43 released attribution files, are ``tests/data/release_provenance.json``.
Needs torch, traker and kronfluence; stamped versions are pinned to the release's.
"""

import json
import os
import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("trak")
pytest.importorskip("kronfluence")

from corrected_lds import cli
from corrected_lds.banks.presets import PRESETS
from corrected_lds.pools import runner
from corrected_lds.tda import ekfac, source, tracin, trak

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(REPO, "tests", "data", "release_provenance.json")) as fh:
    RELEASED = json.load(fh)

# Released files that no in-repo builder reproduces, and why.
EXCEPTIONS = {
    "cifar10/pool_tda_ekfac.npz": "pre-CLI EK-FAC format (a note field, no final_unit)",
    "cifar10/tauq_ekfac_e30.npz": "pre-CLI EK-FAC format (a note field, no final_unit)",
    "mnist/tauq_ekfac_e30.npz": "made by code that was never committed: older protocol "
                                "and aggregation names",
    "wikitext2_gpt2_bae/tauq_ekfac_bergson.npz": "bergson's own EK-FAC scores, built by "
                                                 "bergson (`bergson-ekfac`)",
    "wikitext2_gpt2_bae/tauq_source_L3_e1.npz": "SOURCE E=1 from an out-of-repo kit; no "
                                                "in-repo builder yet",
}
# Runs that overrode the preset's TRAK unit size (compute-tau --e-features).
E_FEATURES_OVERRIDE = {"cifar10/pool_tda_trak20.npz": 20}


@pytest.fixture(autouse=True)
def _release_versions(monkeypatch):
    monkeypatch.setattr(trak, "TRAKER_VERSION", "0.3.2")
    monkeypatch.setattr(ekfac, "KRONFLUENCE_VERSION", "1.0.1")


def _rebuild(name):
    """(the released provenance in today's spelling, what the CLI builds for that run).
    Old spellings: the TRAK family in the method name (``trak10``, ``trak1``, ...; now
    ``e_features``) and SOURCE without its L (``source``); merge-pool-tda also stamps the
    parts and member span of a merged pool_tda."""
    bank = name.split("/")[0]
    p, stored = PRESETS[bank], json.loads(RELEASED[name])
    recipe, method = p["recipe"], stored["method"]
    if re.fullmatch(r"trak\d*", method):
        stored = {k: v for k, v in stored.items() if k not in ("concat_of", "members")}
        stored["method"] = "trak"
        e_feat = E_FEATURES_OVERRIDE.get(name, p["trak"].get("e_features"))
        base = cli._trak_prov_base(p, recipe, p["trak"])
        return stored, cli._trak_provenance(base, stored["n_models"], stored["aggregation"],
                                            e_feat)
    if method.startswith("source"):
        segments = stored["source_args"]["segments"]
        stored["method"] = f"source_L{segments}"
        return stored, cli._source_setup(p, recipe, stored["n_models"], segments,
                                         p["n_train"])[-1]
    if method == "tracin":
        return stored, cli._tracin_setup(p, recipe, stored["n_models"])[-1]
    if method == "ekfac":
        return stored, cli._ekfac_setup(p, recipe, stored["n_models"])[-1]
    raise AssertionError(f"{name}: no builder for method {method!r}")


@pytest.mark.parametrize("name", sorted(set(RELEASED) - set(EXCEPTIONS)))
def test_builders_reproduce_released_provenance(name):
    stored, built = _rebuild(name)
    assert json.dumps(built, sort_keys=True, default=str) == \
        json.dumps(stored, sort_keys=True, default=str)


def test_every_released_attribution_file_is_covered():
    with open(os.path.join(REPO, "banks-manifest.json")) as fh:
        banks = json.load(fh)["banks"]
    released = {f"{bank}/{f}" for bank, files in banks.items() for f in files
                if f.startswith(("tauq_", "pool_tda_"))}
    assert set(RELEASED) == released
    assert set(EXCEPTIONS) <= released


def test_frozen_seeds_and_file_names():
    """Seeds and file names the released artifacts and their resumable work dirs depend on."""
    assert (runner.CELL_SEED_BASE, runner.MASK_SEED) == (50_000, 777)
    assert runner.cell_seed(3, 17) == 50_000 + 3 * 10_000 + 17
    assert (trak.ENS_SEED_BASE, trak.AUG_FEAT_SEED, tracin.TRACIN_SEED_BASE) == (10_000, 20_000, 30_000)
    assert os.path.basename(trak.pieces_path("w", 7)) == "m007.npz"
    assert os.path.basename(ekfac.partial_path("w", 7)) == "tauq_if_m_007.npz"
    assert os.path.basename(tracin.run_partial_path("w", 7)) == "tauq_tracin_r_007.npz"
    assert os.path.basename(source.run_partial_path("w", 7)) == "tauq_source_r_007.npz"
