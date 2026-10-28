"""Torch-free CLI refusals: bergson presets never reach ``build-pool``'s trainers, unknown
datasets raise (no fall-through to ResNet-9), and the ImageNet cache build dry-runs until ``--yes``.
"""

import pytest

from corrected_lds import cli
from corrected_lds.banks.presets import PRESETS


@pytest.mark.parametrize("extra", [[], ["--yes"], ["--merge", "2"]])
def test_build_pool_refuses_a_bergson_preset_before_creating_anything(tmp_path, extra):
    argv = ["build-pool", "--preset", "wikitext2_gpt2_bae", "--tag", "t",
            "--root", str(tmp_path), "--M", "4", "--R", "2", *extra]
    with pytest.raises(SystemExit, match="bergson bank"):
        cli.main(argv)
    assert not (tmp_path / "t").exists()


def test_an_unknown_dataset_raises_instead_of_falling_through(monkeypatch):
    monkeypatch.setitem(PRESETS, "toy", {"dataset": "toy", "recipe": None,
                                         "n_train": 4, "n_targets": 2})
    with pytest.raises(SystemExit, match="no loader for dataset 'toy'"):
        cli._load_preset_data("toy", "cpu")
    # wikitext2 has a loader, but its retrains are bergson's: no pool trainer here
    with pytest.raises(SystemExit, match="no pool trainer for dataset 'wikitext2'"):
        cli._make_task(PRESETS["wikitext2_gpt2_bae"], None, "cpu")


def test_build_data_imagenet_is_a_dry_run_until_yes(monkeypatch, capsys):
    from corrected_lds.pools import datasets
    calls = []
    monkeypatch.setattr(datasets, "build_imagenet_cache", lambda: calls.append("built"))
    assert cli.main(["build-data", "--dataset", "imagenet"]) == 2
    assert calls == [] and "--yes" in capsys.readouterr().out
    assert cli.main(["build-data", "--dataset", "imagenet", "--yes"]) == 0
    assert calls == ["built"]


def test_refusals_name_the_methods_a_preset_has():
    assert cli._methods_of("imagenet") == "trak"
    assert cli._methods_of("qnli") == "trak/tracin/ekfac"
    assert cli._methods_of("cifar10") == "trak/tracin/ekfac/source"
