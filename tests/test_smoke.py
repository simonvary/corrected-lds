import contextlib
import importlib.metadata
import io
import re
import subprocess
import sys

import pytest

import corrected_lds
from corrected_lds.cli import main


def test_version():
    assert corrected_lds.__version__ == importlib.metadata.version("corrected-lds")


def test_core_import_is_torch_free():
    # a fresh interpreter, because torch-gated test modules may have imported torch in this one
    code = ("import sys, corrected_lds, corrected_lds.cli, corrected_lds.banks.hub; "
            "print(sorted({'torch', 'transformers', 'trak', 'kronfluence'} & set(sys.modules)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "[]"


def _help(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 0
    return out.getvalue()


def test_every_subcommand_has_help():
    commands = re.search(r"\{([a-z0-9,-]+)\}", _help(["--help"]))[1].split(",")
    assert "compute-tau" in commands and "fetch-banks" in commands
    for c in commands:
        assert c in _help([c, "--help"])
