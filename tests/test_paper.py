"""Every runnable ``paper/*.py`` and ``slides/*.py`` (whose main calls ``run("<name>", ...)``)
must run from its directory's committed ``plotdata/<name>.json`` on a clean clone (no banks or
GPU, seconds); ``fig*.py`` must render ``<name>.pdf``.  ``--recompute`` is exercised manually
against real banks.
"""

import glob
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
SLIDES = os.path.join(REPO, "slides")
# The plotdata/PDF name is what a script passes to run(), not its file name (_rmse_corrected.py
# writes rmse_corrected.json); matching every script, not a prefix, gates any new figure.
RUN_NAME = re.compile(r'\brun\(\s*"([A-Za-z0-9_]+)"')


def _run_name(path):
    with open(path) as fh:
        m = RUN_NAME.search(fh.read())
    return m.group(1) if m else None


SCRIPTS = sorted((p, _run_name(p)) for d in (PAPER, SLIDES)
                 for p in glob.glob(os.path.join(d, "*.py")) if _run_name(p))


@pytest.mark.parametrize("script,name", SCRIPTS,
                         ids=[os.path.relpath(s, REPO) for s, _ in SCRIPTS])
def test_figure_renders_from_committed_plotdata(script, name, tmp_path):
    plotdata = os.path.join(os.path.dirname(script), "plotdata", f"{name}.json")
    assert os.path.exists(plotdata), (
        f"{os.path.relpath(script, REPO)} has no committed plotdata/{name}.json — every "
        f"figure ships with its numbers (the reproducibility anchor).")
    r = subprocess.run([sys.executable, script, "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=180, check=False)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    if os.path.basename(script).startswith("fig"):      # a stage only writes plotdata
        out = tmp_path / f"{name}.pdf"
        assert out.exists() and out.stat().st_size > 1000


def test_the_figure_scripts_are_the_papers_nine():
    # paper/figN.py builds paper Figure N; everything else is a _*.py stage or helper.
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PAPER, "fig*.py")))
    assert figs == [f"fig{i}.py" for i in range(1, 10)], figs


def test_every_figure_script_was_collected():
    names = {name for _, name in SCRIPTS}
    assert {f"fig{i}" for i in range(1, 10)} <= names, sorted(names)
