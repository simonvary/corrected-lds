"""Torch-free pool_tda tests: replica subsets combine to the exact plain mean, pairing extends
to the mask bind (byte-level ``subsets_sha16``), a ``pool_tda`` of partial tauqs scores like
the averaged tauq (one estimator, two routes), and ``evaluate`` attaches the replica block by
default, keyed by the tauq's provenance, not a filename, unless the routes are incomparable.
"""

import json

import numpy as np
import pytest

from corrected_lds.banks.artifacts import (
    combine_replicas,
    concat_pool_tda,
    load_pool,
    load_pool_tda,
    predictions_tda,
    subsets_digest,
    write_pool_tda,
    write_tauq,
)
from corrected_lds.banks.presets import Resnet9Recipe, bank_hashes
from corrected_lds.cli import main as cli_main
from corrected_lds.evaluate import evaluate_pool, evaluate_pool_tda
from corrected_lds.pools.runner import PoolTask, build_pool

N, Q, M, R = 40, 3, 4, 2


def _meta(n=N, q=Q):
    th, qh = bank_hashes("cifar10")
    return {"n_train": n, "n_targets": q, "dataset": "cifar10",
            "train_idx": list(range(n)), "query_idx": [5, 1, 9][:q],
            "train_hash": th, "query_hash": qh}


def _pool(tmp_path, name="p"):
    recipe = Resnet9Recipe(epochs=1)
    task = PoolTask(recipe=recipe,
                    train_cell=lambda d, sub, seed: (int(seed), np.asarray(sub)),
                    readout=lambda m, d: np.arange(Q, dtype=float) + m[0] * 1e-3)
    data = {"meta": _meta()}
    path = build_pool(str(tmp_path / name), task, data, M=M, R=R, verbose=False)
    return load_pool(path), data["meta"]


def _partials(k, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(N, Q)) for _ in range(k)]


def _pool_tda(tmp_path, pool, meta, parts, name="pool_tda_x.npz", **kw):
    S64 = pool.subsets.astype(np.float64)
    g = np.stack([S64 @ t for t in parts])
    kw.setdefault("method", "ekfac")
    kw.setdefault("inner", "single-member")
    kw.setdefault("e_features", 1)
    kw.setdefault("seed_base", 10_000)
    kw.setdefault("provenance", {"method": kw["method"]})
    out = write_pool_tda(str(tmp_path / name), g, meta, subsets=pool.subsets, **kw)
    return load_pool_tda(out), g


# -- schema: write/load roundtrip + refusals -------------------------------- #
def test_write_load_roundtrip_and_refusals(tmp_path):
    pool, meta = _pool(tmp_path)
    ptda, g = _pool_tda(tmp_path, pool, meta, _partials(3))
    assert ptda.R_tda == 3 and ptda.M == M and ptda.Q == Q
    assert ptda.subsets_sha16 == subsets_digest(pool.subsets)
    assert ptda.e_features == 1 and ptda.inner == "single-member"
    np.testing.assert_allclose(ptda.g, g, atol=0)

    S = pool.subsets
    with pytest.raises(ValueError, match="g shape"):          # M mismatch
        write_pool_tda(str(tmp_path / "b1.npz"), g[:, :M - 1], meta, subsets=S,
                       method="ekfac", inner="single-member", e_features=1,
                       seed_base=0, provenance={})
    with pytest.raises(ValueError, match="inner"):
        write_pool_tda(str(tmp_path / "b2.npz"), g, meta, subsets=S, method="x",
                       inner="exotic", e_features=1, seed_base=0, provenance={})
    with pytest.raises(ValueError, match="non-finite"):
        bad = g.copy()
        bad[0, 0, 0] = np.inf
        write_pool_tda(str(tmp_path / "b3.npz"), bad, meta, subsets=S, method="ekfac",
                       inner="single-member", e_features=1, seed_base=0,
                       provenance={})
    with pytest.raises(ValueError, match="e_features"):
        write_pool_tda(str(tmp_path / "b4.npz"), g, meta, subsets=S, method="trak",
                       inner="separated-avg-eq15", e_features=0, seed_base=0,
                       provenance={})
    np.savez(str(tmp_path / "raw"), g=g)
    with pytest.raises(ValueError, match="missing required key"):
        load_pool_tda(str(tmp_path / "raw.npz"))


# -- combine: exact plain mean over any replica subset ---------------------- #
def test_combine_replicas_exact_and_refusals(tmp_path):
    pool, meta = _pool(tmp_path)
    ptda, g = _pool_tda(tmp_path, pool, meta, _partials(4))
    np.testing.assert_allclose(combine_replicas(ptda), g.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(combine_replicas(ptda, [2]), g[2], atol=0)
    np.testing.assert_allclose(combine_replicas(ptda, [0, 3]),
                               (g[0] + g[3]) / 2, atol=1e-15)
    # prefix = the canonical smaller ensemble
    np.testing.assert_allclose(combine_replicas(ptda, range(2)), g[:2].mean(axis=0),
                               atol=1e-15)
    with pytest.raises(ValueError, match="empty"):
        combine_replicas(ptda, [])
    with pytest.raises(ValueError, match="duplicate"):
        combine_replicas(ptda, [1, 1])
    with pytest.raises(ValueError, match="outside"):
        combine_replicas(ptda, [4])


# -- pairing: data identity and the mask bind ------------------------------- #
def test_predictions_tda_pairing_enforced(tmp_path):
    pool, meta = _pool(tmp_path)
    ptda, g = _pool_tda(tmp_path, pool, meta, _partials(3))
    np.testing.assert_allclose(predictions_tda(pool, ptda), g.mean(axis=0), atol=1e-15)

    # a different query draw is refused even though every shape matches
    meta2 = dict(meta)
    meta2["query_idx"] = [7, 2, 0]
    bad_q, _ = _pool_tda(tmp_path, pool, meta2, _partials(3), name="bad_q.npz")
    with pytest.raises(ValueError, match="query_idx"):
        predictions_tda(pool, bad_q)

    # a different mask draw (same M and data identity) is refused via the byte digest
    other = np.array(pool.subsets)
    other[0, :2] = 1 - other[0, :2]
    fake, _ = _pool_tda(tmp_path, pool, meta, _partials(3), name="bad_m.npz")
    z = dict(np.load(fake.path, allow_pickle=False))
    z["subsets_sha16"] = subsets_digest(other)
    np.savez(str(tmp_path / "bad_m2"), **z)
    with pytest.raises(ValueError, match="mask draw"):
        predictions_tda(pool, load_pool_tda(str(tmp_path / "bad_m2.npz")))


# -- the two artifact routes are one estimator ------------------------------ #
def test_evaluate_pool_tda_matches_tauq_path(tmp_path):
    pool, meta = _pool(tmp_path)
    parts = _partials(4, seed=7)
    ptda, _ = _pool_tda(tmp_path, pool, meta, parts)
    tau_path = write_tauq(str(tmp_path / "tauq_mean.npz"),
                          np.mean(parts, axis=0).astype(np.float32), meta,
                          n_models=4, provenance={"method": "mean-of-partials"})
    from corrected_lds.banks.artifacts import load_tauq
    res_tau = evaluate_pool(pool, load_tauq(tau_path))
    res_tda = evaluate_pool_tda(pool, ptda)
    a, b = res_tau["results"], res_tda["results"]
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, float):
            # tauq stores f32 and the mask sums associate differently -> tiny fp drift
            np.testing.assert_allclose(vb, va, rtol=5e-4, atol=5e-4, err_msg=key)
    src = res_tda["tauq"]
    assert src["method"] == "ekfac" and src["E"] == 4
    assert src["replicas"] == [0, 1, 2, 3] and src["e_features"] == 1
    with pytest.raises(TypeError):                 # Eq. 16 is not expressible on slices
        evaluate_pool_tda(pool, ptda, soft_threshold_k=100)


# -- CLI: the replica block is attached by default, from provenance --------- #
def _cli_bank(tmp_path):
    """A bank dir with a pool, a provenance-carrying tauq, and its pool_tda."""
    pool, meta = _pool(tmp_path, name="bank")
    bank = tmp_path / "bank"
    parts = _partials(3, seed=11)
    write_tauq(str(bank / "tauq_ekfac_e3.npz"), np.mean(parts, axis=0), meta,
               n_models=3, provenance={"method": "ekfac", "aggregation": "plain-mean"})
    ptda, _ = _pool_tda(bank, pool, meta, parts, name="pool_tda_ekfac.npz")
    return pool, bank, ptda


def _run_evaluate(tmp_path, bank, *argv):
    out = tmp_path / "res.json"
    rc = cli_main(["evaluate", str(bank), "--tauq", "ekfac_e3",
                   "--json", str(out), *argv])
    assert rc == 0
    with open(out) as fh:
        return json.load(fh)


def test_cli_evaluate_attaches_pool_tda_by_default(tmp_path):
    pool, bank, ptda = _cli_bank(tmp_path)
    res = _run_evaluate(tmp_path, bank)
    block = res["pool_tda"]
    # same seed/params path; json round-trip normalizes the int dict keys
    direct = json.loads(json.dumps(evaluate_pool_tda(pool, ptda)))
    assert block["results"] == direct["results"]
    assert block["source"]["method"] == "ekfac" and block["source"]["E"] == 3
    assert len(block["replica_rho_R"]) == 3
    assert block["replica_rho_R_sd"] > 0

    assert "pool_tda" not in _run_evaluate(tmp_path, bank, "--pool-tda", "none")
    # Eq. 16 thresholding makes the routes incomparable -> block skipped
    assert "pool_tda" not in _run_evaluate(tmp_path, bank, "--soft-threshold-k", "5")
    # an explicit name resolves like --tauq does; a missing one fails loudly
    assert "pool_tda" in _run_evaluate(tmp_path, bank, "--pool-tda", "ekfac")
    with pytest.raises(SystemExit, match="not found"):
        _run_evaluate(tmp_path, bank, "--pool-tda", "nope")


def test_pool_tda_auto_key_provenance_mapping():
    from corrected_lds.cli import _pool_tda_auto_key

    class _T:
        def __init__(self, prov):
            self.provenance = prov

    # TRAK keys on e_features; a method that names the knob is its own key, whatever the size,
    # so renaming a bank is a data change, not a code change.
    assert _pool_tda_auto_key(
        _T({"method": "trak10", "e_features": 10})) == "trak10"
    assert _pool_tda_auto_key(_T({"method": "trak20", "e_features": 20})) == "trak20"
    assert _pool_tda_auto_key(_T({"method": "trak1", "e_features": 1})) == "trak1"
    assert _pool_tda_auto_key(_T({"method": "trak5", "e_features": 5})) == "trak5"
    assert _pool_tda_auto_key(_T({"method": "trak77"})) == "trak77"
    # a bare "trak" keys on the stamped knob, whatever its size
    assert _pool_tda_auto_key(_T({"method": "trak", "e_features": 5})) == "trak5"
    assert _pool_tda_auto_key(_T({"method": "trak", "e_features": 10})) == "trak10"
    # a bare "trak" without its e_features stamp names no estimator: refuse
    assert _pool_tda_auto_key(_T({"method": "trak"})) is None
    assert _pool_tda_auto_key(
        _T({"method": "trak", "aggregation": "separated-avg-eq15"})) is None
    src = {"method": "source", "source_args": {"segments": 3}}
    assert _pool_tda_auto_key(_T(src)) == "source_L3"          # L is part of the estimator
    assert _pool_tda_auto_key(_T({"method": "source"})) is None
    assert _pool_tda_auto_key(_T({"method": "tracin"})) == "tracin"
    assert _pool_tda_auto_key(_T({"method": "mymethod"})) is None
    assert _pool_tda_auto_key(_T({})) is None


def test_concat_pool_tda_is_exact_and_refuses_mismatches(tmp_path):
    """Parts append exactly along the replica axis, in argument order; a part that disagrees on
    the estimator recipe or the pairing/mask bind is refused (it would mix estimators)."""
    pool, meta = _pool(tmp_path)
    parts = _partials(5)
    a, ga = _pool_tda(tmp_path, pool, meta, parts[:2], name="a.npz")
    b, gb = _pool_tda(tmp_path, pool, meta, parts[2:], name="b.npz")

    out = concat_pool_tda(str(tmp_path / "ab.npz"), [a.path, b.path], pool)
    merged = load_pool_tda(out)
    assert merged.R_tda == 5
    assert np.array_equal(merged.g, np.concatenate([ga, gb], axis=0))   # exact, ordered
    assert (merged.e_features, merged.seed_base) == (a.e_features, a.seed_base)
    assert merged.subsets_sha16 == subsets_digest(pool.subsets)
    # every replica subset of the merge is still an exact smaller-ensemble estimate
    assert np.allclose(combine_replicas(merged, [0, 3]),
                       0.5 * (ga[0] + gb[1]))
    assert [x["R_tda"] for x in merged.provenance["concat_of"]] == [2, 3]

    # a part built at another inner size is a different estimator
    c, _ = _pool_tda(tmp_path, pool, meta, parts[:1], name="c.npz",
                     e_features=2, inner="separated-avg-eq15", method="trak")
    with pytest.raises(ValueError, match="e_features"):
        concat_pool_tda(str(tmp_path / "bad.npz"), [a.path, c.path], pool)

    # ...and a pool whose masks are not the ones the parts were projected on
    other, _ = _pool(tmp_path, name="other")
    object.__setattr__(other, "subsets", 1 - pool.subsets)
    with pytest.raises(ValueError, match="refusing to rebind"):
        concat_pool_tda(str(tmp_path / "bad2.npz"), [a.path], other)
