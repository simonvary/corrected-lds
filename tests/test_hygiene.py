"""Public-tree hygiene: no tracked file may carry host details (IPv4 addresses, ``.ssh/`` paths,
session URLs, dev work directories), personal names, pointers to planning notes outside the
repository or unresolvable commit ids, nor exceed 1 MB.  Skips outside a git checkout (sdist).
"""

import functools
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = "tests/test_hygiene.py"
MAX_BYTES = 1_000_000

RULES = {
    "ipv4": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "ssh": re.compile(r"\.ssh/"),
    "session-url": re.compile(r"claude\.ai/code/session_"),
    "dev-root": re.compile(r"/workspace/(?:qnli_pool|bergson-b-pool|pod_watchdog)|/root/\.claude"),
    # authorship lives in LICENSE and pyproject; literature citations ("Grosse et al.") are fine
    "name": re.compile(r"\b(?:Simon|Roger|Alireza|Vary|Mousavi|mousavih)\b|\bGrosse\b(?! et al)"),
    # sections and phases of planning notes that are not in the repository (JSON escapes §)
    "private-note": re.compile(r"plan (?:§|\\u00a7)|\bplan P\d|docs/plan\.md"
                               r"|plan-(?:imagenet|tracin|hub|ekfac-bert)"),
    # 7-40 hex digits with at least one letter and one digit: a commit or revision id
    "sha": re.compile(r"(?<![0-9A-Za-z_])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}"
                      r"(?![0-9A-Za-z_])"),
}

# rule -> files, or directories ending in "/", it does not apply to
EXEMPT = {
    "ipv4": ("uv.lock", "paper/plotdata/"),
    "name": ("LICENSE", "pyproject.toml", "CITATION.cff"),
    "private-note": (".gitignore",),        # it names the local notes so they stay untracked
    "sha": ("uv.lock", "paper/plotdata/", "banks-manifest.json", "tests/data/"),
}

# Ids a reader can resolve or place: upstream pins, the commits stamped into released pools
# (in the private development history, as FORMATS says), and the revision
# banks-manifest.json pins; 16 hex digits are hash pins.
PINS = (
    "0ad2e5b0ce09207d2af0e80383fe2d952dd8e747",    # bergson, the WikiText-2 trainer
    "08e50effa843f8350332371cd4eef57823778e94",    # bergson #511, the SOURCE fix
    "ea4f6941",                                     # #511 cherry-picked onto that pin
    "971da0e6da181b9e147081087f1d3fd51669024c",    # bergson-wikitext-2-4656-chunks
    "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",    # ImageNet-1k parquet
    "e95b405", "7a1a04f", "7135506", "798d4cf", "38ded77", "693841c",
)


@functools.cache
def _tracked():
    """Tracked files, or None outside a git checkout of this repository."""
    try:
        top = subprocess.run(["git", "-C", REPO, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        out = subprocess.run(["git", "-C", REPO, "ls-files", "-z"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if os.path.realpath(top) != os.path.realpath(REPO):
        return None
    return tuple(f for f in out.split("\0") if f)


@functools.cache
def _resolvable():
    with open(os.path.join(REPO, "banks-manifest.json"), encoding="utf-8") as fh:
        return PINS + tuple(re.findall(r"/resolve/([0-9a-f]{40})", fh.read()))


def _exempt(path, rule):
    return any(path == e or (e.endswith("/") and path.startswith(e))
               for e in EXEMPT.get(rule, ()))


def _line_hits(path, line):
    """``(rule, match)`` for every rule ``line`` of ``path`` breaks."""
    hits = []
    for rule, rx in RULES.items():
        if _exempt(path, rule):
            continue
        for m in rx.finditer(line):
            if rule == "ipv4" and line[max(0, m.start() - 2):m.start()] == "==":
                continue                                # a version pin such as pkg==12.8.4.1
            if rule == "sha" and (len(m.group()) == 16
                                  or any(p.startswith(m.group()) for p in _resolvable())):
                continue
            hits.append((rule, m.group()))
    return hits


@functools.cache
def _findings():
    """``(path, rule) -> ["line: match", ...]`` over every tracked file."""
    found = {}
    for path in _tracked() or ():
        full = os.path.join(REPO, path)
        if path == SELF or not os.path.isfile(full):
            continue
        if os.path.getsize(full) > MAX_BYTES:
            found.setdefault((path, "size"), []).append(f"{os.path.getsize(full):,} bytes")
        try:
            with open(full, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except UnicodeDecodeError:
            continue                                    # binary: only the size rule applies
        for n, line in enumerate(lines, 1):
            for rule, match in _line_hits(path, line):
                found.setdefault((path, rule), []).append(f"{n}: {match}")
    return found


def test_tracked_files_carry_no_private_details():
    if _tracked() is None:
        pytest.skip("not a git checkout of this repository")
    bad = _findings()
    assert not bad, "\n".join(f"{path} [{rule}] {', '.join(v[:5])}"
                              for (path, rule), v in sorted(bad.items()))
