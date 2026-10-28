"""The contract every paper and talk figure follows: ``compute()`` (banks -> numbers, the
expensive seeded part) and ``plot()`` (numbers -> PDF), wired by ``run()``:

    python paper/figN.py               # render from committed plotdata/figN.json
    python paper/figN.py --recompute   # regenerate plotdata from banks/ first

Stages (``paper/_*.py``) pass no ``plot()``.  ``compute()`` returns a JSON-able dict that
stamps its inputs (:func:`bank_stamp` per bank, every seed and parameter in ``"params"``),
so each committed number names the artifacts and settings behind it.
"""

import argparse
import json
import os

import numpy as np

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTDATA_DIR = os.path.join(PAPER_DIR, "plotdata")

SCHEMA = 1


def bank_stamp(pool) -> dict:
    """The identity of one pool, for a plotdata ``"banks"`` block."""
    return {"dataset": pool.dataset, "M": pool.M, "R": pool.R, "Q": pool.Q,
            "n": pool.n, "alpha": pool.alpha, "train_hash": pool.train_hash,
            "query_hash": pool.query_hash, "code_version": pool.code_version}


def to_jsonable(x):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts the dict."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def run(name, compute, plot=None, argv=None, root=PAPER_DIR):
    """Load-or-recompute ``<root>/plotdata/<name>.json``, then render ``<root>/figures/<name>.pdf``
    (``root``: ``paper/`` or ``slides/``); ``compute(banks_root) -> dict``,
    ``plot(data, out_path) -> None`` (None for a stage)."""
    ap = argparse.ArgumentParser(description=f"figure: {name}")
    ap.add_argument("--recompute", action="store_true",
                    help="regenerate plotdata from banks (default: use the committed file)")
    ap.add_argument("--banks-root", default=os.path.join(os.getcwd(), "banks"),
                    help="bank directories root, for --recompute (default: ./banks)")
    ap.add_argument("--out", default=os.path.join(root, "figures"),
                    help="output directory for the PDF")
    args = ap.parse_args(argv)

    plotdata_dir = os.path.join(root, "plotdata")
    pd_path = os.path.join(plotdata_dir, f"{name}.json")
    if args.recompute or not os.path.exists(pd_path):
        data = to_jsonable(compute(args.banks_root))
        data = {"schema": SCHEMA, "figure": name, **data}
        os.makedirs(plotdata_dir, exist_ok=True)
        with open(pd_path, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"[{name}] wrote {pd_path}")
    else:
        with open(pd_path) as fh:
            data = json.load(fh)
        if data.get("schema") != SCHEMA:
            raise SystemExit(f"[{name}] {pd_path} has schema {data.get('schema')}, "
                             f"expected {SCHEMA}; --recompute it.")

    if plot is None:
        return 0
    os.makedirs(args.out, exist_ok=True)
    # a fixed PDF date, so the same matplotlib re-renders a committed figure byte for byte
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    out = os.path.join(args.out, f"{name}.pdf")
    plot(data, out)
    print(f"[{name}] wrote {out}")
    return 0
