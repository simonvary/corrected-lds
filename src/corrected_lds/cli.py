"""The corrected-lds command line — one argparse subcommand tree.

Installed as ``corrected-lds`` and runnable as ``python -m corrected_lds``.  The module
imports no torch (GPU subcommands load ``pools/`` lazily), so ``--help`` and the data
commands need only core deps.  Expensive steps (``build-pool``, ``compute-tau``, imagenet
``build-data``) print the resume state and a cost estimate, then require ``--yes``.
"""

import argparse
import json
import os

import numpy as np

from corrected_lds.banks.presets import PRESETS

# rough per-cell training minutes on one modern GPU, for the estimate: qnli assumes bf16
# (fp32 costs ~4x); imagenet is unmeasured (Table D.2: ~29 min per TRAK model at alpha=0.5).
_CELL_MINUTES = {"cifar10": 1.5, "mnist": 0.15,
                 "qnli": 45.0, "imagenet": 30.0, "wikitext2": 4.5}
# rough per-member TRAK featurize+score cost (chunked; independent of ensemble size)
_FEAT_MINUTES = {"cifar10": 3.0, "mnist": 0.5,
                 "qnli": 25.0, "imagenet": 10.0, "wikitext2": 2.0}  # wikitext2: measured


def _cmd_build_data(args):
    from corrected_lds.pools import datasets
    if args.smoke:
        if args.dataset != "imagenet":
            raise SystemExit("[data] --smoke is imagenet-only (the other caches are "
                             "small enough to just build).")
        datasets.smoke_imagenet_source()   # one shard per split, no cache writes
        return 0
    if args.dataset == "cifar10":
        datasets.build_cifar_cache()
    elif args.dataset == "qnli":
        datasets.build_qnli_cache()
    elif args.dataset == "mnist":
        datasets.build_mnist_cache()
    elif args.dataset == "wikitext2":
        datasets.build_wikitext2_cache()
    elif args.dataset == "imagenet":
        if not args.yes:
            print("[data] imagenet downloads ~153 GB of license-gated parquet and writes a "
                  "~75 GB cache (a rerun re-verifies the cache in full); --smoke checks "
                  "access first for ~1 GB.")
            print("[data] dry run only — pass --yes to start (never auto-run).")
            return 2
        datasets.build_imagenet_cache()    # rerun on an existing cache = full verify
    else:
        raise SystemExit(f"[data] no cache builder for dataset {args.dataset!r}")
    return 0


def _load_preset_data(preset, device):
    """The preset's data dict + recipe, on ``device``."""
    p = PRESETS[preset]
    recipe = p["recipe"]
    if p["dataset"] == "qnli":
        from corrected_lds.pools.datasets import load_qnli
        data = load_qnli(p["n_train"], p["n_targets"], seed=0, device=device,
                         seq_len=recipe.seq_len, model_name=recipe.model)
    elif p["dataset"] == "mnist":
        from corrected_lds.pools.datasets import load_mnist
        data = load_mnist(p["n_train"], p["n_targets"], seed=0, device=device)
    elif p["dataset"] == "wikitext2":  # used whole, in the hosted row order
        from corrected_lds.pools.datasets import load_wikitext2
        data = load_wikitext2(p["n_train"], p["n_targets"], seed=0, device=device)
    elif p["dataset"] == "imagenet":   # query decode geometry comes from the recipe
        from corrected_lds.pools.datasets import load_imagenet
        data = load_imagenet(p["n_train"], p["n_targets"], seed=0, device=device,
                             val_res=recipe.val_res, crop_ratio=recipe.crop_ratio)
    elif p["dataset"] == "cifar10":
        from corrected_lds.pools.datasets import load_cifar
        data = load_cifar(p["n_train"], p["n_targets"], seed=0, device=device)
    else:
        raise SystemExit(f"[data] no loader for dataset {p['dataset']!r}")
    return p, recipe, data


def _make_task(preset_cfg, recipe, device):
    from corrected_lds.pools.runner import PoolTask
    if preset_cfg["dataset"] == "qnli":
        from corrected_lds.pools.tasks.qnli_bert import margins, train_text_model
        return PoolTask(
            recipe=recipe,
            train_cell=lambda data, sub, seed: train_text_model(data, sub, seed, recipe,
                                                                device, verbose=True),
            readout=lambda model, data: margins(
                model, data, device, eval_batch=recipe.eval_batch,
                precision=recipe.precision))
    if preset_cfg["dataset"] == "imagenet":
        from corrected_lds.pools.tasks.imagenet_resnet18 import (
            margins,
            train_imagenet_model,
        )
        return PoolTask(
            recipe=recipe,
            train_cell=lambda data, sub, seed: train_imagenet_model(data, sub, seed,
                                                                    recipe, device,
                                                                    verbose=True),
            readout=lambda model, data: margins(model, data, device))
    if preset_cfg["dataset"] == "mnist":
        from corrected_lds.pools.tasks.mnist_mlp import margins, train_mnist_model
        return PoolTask(
            recipe=recipe,
            train_cell=lambda data, sub, seed: train_mnist_model(
                data, sub, seed, recipe, device, verbose=True),
            readout=lambda model, data: margins(model, data, device))
    if preset_cfg["dataset"] != "cifar10":
        raise SystemExit(f"[pool] no pool trainer for dataset {preset_cfg['dataset']!r}")
    from corrected_lds.pools.tasks.cifar_resnet9 import margins, train_paper_model
    return PoolTask(
        recipe=recipe,
        train_cell=lambda data, sub, seed: train_paper_model(data, sub, seed, recipe,
                                                             device, verbose=True),
        readout=lambda model, data: margins(model, data, device))


def _methods_of(preset):
    """``preset``'s compute-tau methods, e.g. ``"trak/tracin/ekfac"``: EK-FAC needs its own
    block (its damping) and scores the TracIn trajectories."""
    p = PRESETS[preset]
    return "/".join(m for m in ("trak", "tracin", "ekfac", "source")
                    if p.get(m) and (m != "ekfac" or p.get("tracin")))


def _cmd_build_pool(args):
    if PRESETS[args.preset].get("harness") == "bergson":     # before anything is created
        raise SystemExit(f"[pool] preset {args.preset!r} is a bergson bank: its retrains run "
                         f"on bergson's own harness, so build it with `corrected-lds "
                         f"bergson-pool` (docs/REPRODUCE.md).")
    from corrected_lds.pools import runner
    p, recipe, data = _load_preset_data(args.preset, args.device)
    pool_dir = os.path.join(args.root, args.tag)
    n, Q = data["meta"]["n_train"], data["meta"]["n_targets"]

    if args.merge is not None:
        runner.merge_shards(pool_dir, recipe, data, args.M, args.R, args.merge,
                            allow_mixed_code=args.allow_mixed_code)
        print(f"[pool] merged {args.merge} shards -> {os.path.join(pool_dir, 'pool.npz')}")
        return 0

    shard = tuple(args.shard) if args.shard else None
    filename = "pool.npz" if shard is None else f"pool_shard_{shard[0]}.npz"
    margins, _, _, path, _ = runner.init_pool(
        pool_dir, n, Q, args.M, args.R, recipe, p["dataset"], filename=filename,
        ident=runner.identity_from_meta(data["meta"]), shard=shard)
    if shard is None:
        todo = runner.remaining_cells(margins)
    else:
        cells = runner.shard_cells(args.M, args.R, shard[0], shard[1])
        todo = [c for c in cells if np.isnan(margins[c[0], c[1]]).any()]
    est_h = len(todo) * _CELL_MINUTES[p["dataset"]] / 60.0
    print(f"[pool] {path}: {args.R * args.M - len(runner.remaining_cells(margins))}/"
          f"{args.R * args.M} cells done; this run would train {len(todo)} cell(s), "
          f"~{est_h:.1f} GPU-hours at ~{_CELL_MINUTES[p['dataset']]} min/cell.")
    if not args.yes:
        print("[pool] dry run only — pass --yes to start training (never auto-run).")
        return 2
    runner.build_pool(pool_dir, _make_task(p, recipe, args.device), data, args.M, args.R,
                      save_every=args.save_every, shard=shard)
    return 0


def _parse_members(spec, n_models):
    """'0-4' or '0,2,5' -> a sorted list of member indices, validated against n_models."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = sorted(m for m in out if not 0 <= m < n_models)
    if bad or not out:
        raise SystemExit(f"[tau] --members {spec!r} selects {sorted(out) or 'nothing'}; "
                         f"valid indices are 0..{n_models - 1}.")
    return sorted(out)


_TAU_AGGS = {"eq15": ("separated-avg-eq15",), "plain-mean": ("plain-mean",),
             "both": ("separated-avg-eq15", "plain-mean")}
# file key per aggregation; "trak" gets the preset's e_features appended (trak10, trak20, ...)
_TAU_FILE_KEY = {"separated-avg-eq15": "trak", "plain-mean": "trak1"}


def _pool_for_tda(pool_dir, data):
    """The pool whose masks ``pool_tda_*`` slices bind to, or None (reason printed); a pool
    whose data identity differs from ``data`` is an error, as its masks index other rows."""
    from corrected_lds.banks.artifacts import load_pool
    path = os.path.join(pool_dir, "pool.npz")
    if not os.path.exists(path):
        print(f"[tau] {path} not found -> pool_tda_* not emitted (replica-resolved "
              f"predictions bind to the pool's masks; build the pool, then re-run "
              f"compute-tau — ekfac/tracin partials are kept for that backfill).")
        return None
    pool = load_pool(path)
    meta = data["meta"]
    bad = []
    if pool.dataset != str(meta["dataset"]):
        bad.append("dataset")
    if not np.array_equal(pool.train_idx, np.asarray(meta["train_idx"], dtype=np.int64)):
        bad.append("train_idx")
    if not np.array_equal(pool.query_idx, np.asarray(meta["query_idx"], dtype=np.int64)):
        bad.append("query_idx")
    if pool.train_hash != str(meta["train_hash"]) \
            or pool.query_hash != str(meta["query_hash"]):
        bad.append("data hashes")
    if bad:
        raise SystemExit(f"[tau] {path} does not match the loaded data "
                         f"({', '.join(bad)} differ); refusing to bind pool_tda "
                         f"slices to its masks.")
    return pool


def _emit_pool_tda_partials(out_path, work_dir, pool, meta, n_units, partial_fn, *,
                            method, inner, seed_base, provenance):
    """Project the per-member/per-run partial tauqs in ``work_dir`` onto the pool masks and
    write ``out_path``; False if a partial is missing (the caller keeps ``work_dir``)."""
    paths = [partial_fn(work_dir, r) for r in range(n_units)]
    if not all(os.path.exists(pp) for pp in paths):
        return False
    if os.path.exists(out_path):           # never silently replace a larger or different file
        with np.load(out_path, allow_pickle=False) as z:
            old = json.loads(str(z["provenance"]))
        new = json.loads(json.dumps(provenance, sort_keys=True, default=str))
        if int(old.get("n_models", 0)) > n_units or (
                {k: v for k, v in old.items() if k != "n_models"}
                != {k: v for k, v in new.items() if k != "n_models"}):
            raise SystemExit(f"[tau] {out_path} exists with {old.get('n_models')} replicas or "
                             f"different provenance; refusing to replace it with {n_units}. "
                             f"Move it away to rebuild it.")
    S64 = pool.subsets.astype(np.float64)
    g = np.stack([S64 @ np.load(pp, allow_pickle=False)["tauq"].astype(np.float64)
                  for pp in paths])
    from corrected_lds.banks.artifacts import write_pool_tda
    write_pool_tda(out_path, g, meta, subsets=pool.subsets, method=method, inner=inner,
                   e_features=1, seed_base=seed_base, provenance=provenance)
    print(f"[tau] wrote {out_path} (pool_tda g {g.shape}, {method}, "
          f"replica = one {inner.split('-', 1)[1]}).")
    return True


# -- provenance: what each compute-tau artifact stamps ---------------------- #
# Reuse checks and tests/test_provenance.py (against the released files) compare it
# verbatim.
def _unit_size(aggregation, e_feat):
    """The e_features of the artifact ``aggregation`` produces: plain-mean is always 1, as
    Eq. 15 over a one-member unit reduces to that member's own estimate."""
    return int(e_feat) if aggregation == "separated-avg-eq15" else 1


def _trak_prov_base(p, recipe, cfg):
    """What every TRAK artifact of one run shares: the resolved TRAK block, recipe, traker."""
    from corrected_lds.tda.trak import TRAKER_VERSION
    prov_base = {"method": "trak", "recipe": recipe.to_dict(), "alpha": recipe.alpha,
                 "dataset": p["dataset"],
                 "proj_dim": int(cfg["proj_dim"]),
                 "featurize_augment": bool(cfg.get("featurize_augment", False)),
                 "use_half_precision": bool(cfg.get("use_half_precision", False)),
                 "traker_version": TRAKER_VERSION}
    if "soft_threshold_k" in cfg:     # the Eq. 16 evaluation protocol this tauq expects
        prov_base["soft_threshold_k"] = int(cfg["soft_threshold_k"])
    if cfg.get("tracked"):                              # a parameter subset is identity
        prov_base["tracked"] = list(cfg["tracked"])
    if p.get("harness") == "bergson":   # who trained the members; "harness" instead says
        prov_base["member_harness"] = p["harness"]   # who scored a tauq (_pool_tda_auto_key)
    return prov_base


def _trak_provenance(prov_base, n_models, aggregation, e_feat):
    """One TRAK artifact; e_features (estimator identity, pool_tda key) follows ``aggregation``."""
    return {**prov_base, "n_models": int(n_models), "aggregation": aggregation,
            "e_features": _unit_size(aggregation, e_feat)}


def _tracin_setup(p, recipe, n_models):
    """TracIn's args, scored checkpoint epochs and provenance for an n_models run."""
    from corrected_lds.tda import tracin as tracinmod
    from corrected_lds.tda.trak import TRAKER_VERSION
    cfg = p["tracin"]
    targs = tracinmod.TracinArgs(ckpt_every=int(cfg["ckpt_every"]),
                                 label_smoothing=getattr(recipe, "label_smoothing", 0.0),
                                 use_half_precision=bool(cfg.get("use_half_precision",
                                                                 False)),
                                 checkpoint_epochs=cfg.get("checkpoint_epochs", ()),
                                 units_per_epoch=int(cfg.get("units_per_epoch", 1)),
                                 tracked=tuple(cfg.get("tracked", ())),
                                 projection_per_checkpoint=bool(
                                     cfg.get("projection_per_checkpoint", False)))
    cps = tracinmod.resolve_ckpt_epochs(recipe.epochs, targs.ckpt_every,
                                        targs.checkpoint_epochs, targs.units_per_epoch)
    prov = {"method": "tracin", "recipe": recipe.to_dict(), "n_models": int(n_models),
            "dataset": p["dataset"], "proj_dim": int(cfg["proj_dim"]),
            "tracin_args": targs.to_dict(), "ckpt_epochs": [int(e) for e in cps],
            "seed_base": tracinmod.TRACIN_SEED_BASE, "protocol": "full-set-trajectory",
            "aggregation": "plain-mean-trajectories", "traker_version": TRAKER_VERSION}
    return targs, cps, prov


def _source_setup(p, recipe, n_models, n_seg, n_train):
    """SOURCE's args, scored checkpoint epochs and provenance (incl. per-segment lr x steps)."""
    from corrected_lds.tda import source as srcmod
    from corrected_lds.tda import tracin as tracinmod
    from corrected_lds.tda.ekfac import KRONFLUENCE_VERSION
    cfg = p["source"]
    sargs = srcmod.SourceArgs(segments=n_seg,
                              ckpt_every=int(cfg["ckpt_every"]),
                              label_smoothing=recipe.label_smoothing,
                              factor_batch=int(cfg.get("factor_batch", 64)),
                              grad_batch=int(cfg.get("grad_batch", 64)),
                              query_chunk=int(cfg.get("query_chunk", 500)),
                              grad_dtype=str(cfg.get("grad_dtype", "float64")),
                              checkpoint_epochs=cfg.get("checkpoint_epochs", ()))
    cps = tracinmod.resolve_ckpt_epochs(recipe.epochs, sargs.ckpt_every,
                                        sargs.checkpoint_epochs)
    lr_k = srcmod.segment_lr_times_steps(recipe, n_train, sargs)
    # L is estimator identity, so it is part of the method, as e_features is for trak
    prov = {"method": f"source_L{sargs.segments}", "recipe": recipe.to_dict(),
            "n_models": int(n_models),
            "dataset": p["dataset"], "source_args": sargs.to_dict(),
            "ckpt_epochs": [int(e) for e in cps], "lr_times_steps": [float(v) for v in lr_k],
            "seed_base": tracinmod.TRACIN_SEED_BASE, "protocol": "full-set-trajectory",
            "aggregation": "plain-mean-trajectories",
            "kronfluence_version": KRONFLUENCE_VERSION}
    return sargs, cps, prov


def _ekfac_setup(p, recipe, n_models):
    """EK-FAC's args, the trajectory layout it scores (TracIn's) and its provenance."""
    from corrected_lds.tda import ekfac as ekfacmod
    from corrected_lds.tda import tracin as tracinmod
    tcfg = p["tracin"]
    k = int(tcfg.get("units_per_epoch", 1))
    cps = tracinmod.resolve_ckpt_epochs(recipe.epochs, int(tcfg["ckpt_every"]), (), k)
    # the preset block is EkfacArgs kwargs; a text bank must set its own chunking (the ResNet-9
    # query_batch=500 needs 159 GB at bert-base).  Batch sizes are not provenance; low_rank is.
    eargs = ekfacmod.EkfacArgs(**{"label_smoothing": getattr(recipe, "label_smoothing", 0.0),
                                  **{k: v for k, v in (p.get("ekfac") or {}).items()
                                     if k != "ensemble_sizes"}})
    prov = {"method": "ekfac", "recipe": recipe.to_dict(), "n_models": int(n_models),
            "dataset": p["dataset"], "ekfac_args": eargs.to_dict(),
            "protocol": "full-set-final-checkpoint", "final_unit": int(cps[-1]),
            "units_per_epoch": k, "seed_base": tracinmod.TRACIN_SEED_BASE,
            "aggregation": "plain-mean", "kronfluence_version": ekfacmod.KRONFLUENCE_VERSION}
    return eargs, cps, k, prov


def _pin_projector(p):
    """Pin traker's CUDA projector to the SM count of the GPU that made the release
    (``projector_num_sms``), so TRAK and TracIn draw the released projection on any GPU."""
    n = p.get("projector_num_sms")
    if n is not None:
        from corrected_lds.tda.trak import pin_projector_sms
        pin_projector_sms(n)
        print(f"[tau] CUDA projector pinned to {n} SMs, the release GPU's count.")


def _cmd_compute_tau(args):
    if args.method != "trak" and args.aggregation != "eq15":
        raise SystemExit("[tau] --aggregation is a TRAK knob (Eq. 15 vs plain-mean): "
                         "EK-FAC, TracIn and SOURCE are plain-mean natively — each "
                         "member/run yields a self-contained influence estimate with "
                         "nothing to separate.")
    if args.method != "trak" and getattr(args, "e_features", None):
        raise SystemExit("[tau] --e-features is a TRAK knob (the inner Eq. 15 unit "
                         "size): EK-FAC, TracIn and SOURCE replicas are single "
                         "members/runs — e_features = 1 by construction.")
    if args.method != "trak" and (args.members is not None
                                  or getattr(args, "block", None) is not None):
        raise SystemExit("[tau] --members / --block are TRAK knobs (its pieces cache); "
                         "EK-FAC, TracIn and SOURCE split by trajectory run with --runs.")
    if args.method in ("trak", "tracin"):
        _pin_projector(PRESETS[args.preset])
    if args.method == "tracin":     # its own flow: trajectory ckpts, per-run partials
        return _cmd_compute_tau_tracin(args)
    if args.method == "source":     # same trajectories as tracin, segment-walked
        return _cmd_compute_tau_source(args)
    if args.method == "ekfac" and PRESETS[args.preset]["dataset"] == "imagenet":
        raise SystemExit("[tau] the EK-FAC path has not been exercised on imagenet: it "
                         "fits factors over the WHOLE train set in one pass and scores "
                         "every (query, train) pair, neither of which streams at 1.28M "
                         "images; use --method trak here.")
    if args.method == "ekfac":      # full-set runs' final checkpoints, never TRAK members
        return _cmd_compute_tau_ekfac_trajectories(args)
    from corrected_lds.tda import trak as trakmod
    p, recipe, data = _load_preset_data(args.preset, args.device)
    cfg = dict(p["trak"])
    bergson_members = p.get("harness") == "bergson"  # imported, never trained here
    n_models = args.n_models or max(cfg["ensemble_sizes"])
    members = None
    if args.members is not None:                     # pieces-only mode (multi-GPU)
        members = _parse_members(args.members, n_models)
    block = None
    if getattr(args, "block", None) is not None:     # one block of the roster -> slices
        if members is not None:
            raise SystemExit("[tau] --block and --members are exclusive: cache the "
                             "block's pieces with --members first, then assemble it "
                             "with --block.")
        block = _parse_members(args.block, n_models)
        if block != list(range(block[0], block[-1] + 1)):
            raise SystemExit(f"[tau] --block {args.block!r} must be ONE contiguous range "
                             f"A-B: slice order is replica order.")
    trakmod.check_proj_dim(cfg["proj_dim"], data["meta"]["n_train"])
    override = getattr(args, "e_features", None)     # --e-features beats the preset
    e_feat = int(override or cfg["e_features"])
    pool_dir = os.path.join(args.root, args.tag)
    ckpt_dir = os.path.join(pool_dir, "ckpts")
    aggs = _TAU_AGGS[args.aggregation]
    # prefix snapshots: smaller released sizes come from the same featurize pass, since the
    # first E seed-determined members are the canonical E ensemble (= a --n-models E run).
    snap_at = tuple(e for e in sorted(set(cfg["ensemble_sizes"])) if e < n_models)
    es = [*snap_at, n_models]
    # every size, snapshots included, must be whole Eq. 15 units: a partial unit is a
    # different estimator under the family's name.
    if block is not None and len(block) % e_feat:
        raise SystemExit(f"[tau] --block {args.block!r} holds {len(block)} members, not a "
                         f"multiple of e_features={e_feat}: its last Eq. 15 unit would be "
                         f"partial.")
    bad = [e for e in es if e % e_feat]
    if bad:
        raise SystemExit(f"[tau] ensemble size(s) {bad} are not multiples of the "
                         f"preset's e_features={e_feat} — each E replica is one "
                         f"COMPLETE Eq. 15 estimator over e_features members.")
    def key(a):
        k = _TAU_FILE_KEY[a]
        return f"{k}{int(e_feat)}" if k == "trak" else k

    def ef_of(a):
        return _unit_size(a, e_feat)

    def E_of(a, n):
        """``E`` counts replicas (averaged estimator units), not models: n_models / e_features,
        the axis of the 1/E attenuation law (tauq_trak5_e4 = four units of five models)."""
        return int(n) // ef_of(a)

    outs = {(a, e): os.path.join(pool_dir, f"tauq_{key(a)}_e{E_of(a, e)}.npz")
            for a in aggs for e in es}
    out_names = ", ".join(outs[a, e] for a in aggs for e in es)

    hp = bool(cfg.get("use_half_precision", False))       # TRAK numeric precision: tauq
    prov_base = _trak_prov_base(p, recipe, cfg)
    provs = {(a, e): _trak_provenance(prov_base, e, a, e_feat) for a in aggs for e in es}
    acc_prov = {**prov_base, "n_models": int(n_models),   # the shared-pass stamp
                "aggregation": "+".join(aggs),
                **({"snapshot_at": [int(e) for e in snap_at]} if snap_at else {})}
    existing = [k for k in outs if os.path.exists(outs[k])] if block is None else []
    if existing:
        ok = len(existing) == len(outs)
        for k in existing:
            z = np.load(outs[k], allow_pickle=False)
            stored = str(z["provenance"]) if "provenance" in z.files else None
            ok = ok and stored == json.dumps(provs[k], sort_keys=True, default=str)
        if ok:
            print(f"[tau] {out_names} present (provenance ok) -> reuse.")
            return 0
        raise SystemExit(f"[tau] existing tauq file(s) do not match this run "
                         f"({', '.join(outs[k] for k in existing)}); refusing to "
                         f"overwrite. Delete them or use a fresh tag.")

    scope = (block if block is not None else members if members is not None
             else list(range(n_models)))
    have = sum(os.path.exists(os.path.join(ckpt_dir, f"member_{m:03d}.pt")) for m in scope)
    n_train_here = 0 if bergson_members else len(scope) - have
    est_h = (n_train_here * _CELL_MINUTES[p["dataset"]] + len(scope) * _FEAT_MINUTES[p["dataset"]]) / 60
    what = (f"assemble pool_tda slices for block {block[0]}-{block[-1]} of {n_models} "
            f"from its cached pieces (no tauq written)" if block is not None
            else f"featurize/score {n_models}" if members is None
            else f"cache pieces for members {members} of {n_models} (no tauq written)")
    print(f"[tau] {out_names}: {have}/{len(scope)} ensemble checkpoints present; this run "
          f"would train {n_train_here} member(s) + {what}, ~{est_h:.1f} GPU-hours.")
    if bergson_members and have < len(scope):
        print(f"[tau] {len(scope) - have} member(s) are not imported yet; bergson trains "
              f"them (`corrected-lds import-bergson-member`; see docs/REPRODUCE.md).")
    if not args.yes:
        print("[tau] dry run only — pass --yes to start (never auto-run).")
        return 2

    pool = _pool_for_tda(pool_dir, data)          # None -> pool_tda skipped (printed)
    if block is not None and pool is None:
        raise SystemExit("[tau] --block writes pool_tda slices, which bind the pool's "
                         "masks: build or fetch the pool first.")
    want_units = pool is not None and "separated-avg-eq15" in aggs
    want_pm = pool is not None and "plain-mean" in aggs
    if bergson_members:
        ckpts = _bergson_members(p, ckpt_dir, n_models, scope)
    else:
        task = _make_task(p, recipe, args.device)             # reuse the pool trainer
        ckpts = trakmod.train_ensemble(ckpt_dir, data, recipe, n_models, task.train_cell,
                                       members=members if block is None else block)
    work_dir = os.path.join(pool_dir, f"tau_work_trak_e{n_models}")
    scratch = args.scratch or work_dir
    slice_kw = {"unit_size": e_feat if want_units else None,
                "subsets": pool.subsets if (want_units or want_pm) else None,
                "member_slices": want_pm, "snapshot_at": snap_at}
    if block is not None:   # the block's members, global ids, pieces from the shared cache
        slice_kw.update(snapshot_at=(), member_ids=block, pieces_dir=work_dir)
        ckpts = [ckpts[m] for m in block]
        work_dir = os.path.join(pool_dir, f"tau_work_trak_e{n_models}"
                                          f"_b{block[0]}-{block[-1]}")
        scratch = args.scratch or work_dir
    if p["dataset"] == "qnli":
        from corrected_lds.pools.tasks.qnli_bert import make_text_model
        mctor = lambda: make_text_model(recipe, data["meta"]["n_classes"], args.device,
                                        pretrained=False)
        tauqs = trakmod.compute_tauq_text(
            ckpts, data, mctor, proj_dim=cfg["proj_dim"], device=args.device,
            batch=int(cfg.get("trak_batch", 32)), work_dir=work_dir, scratch_dir=scratch,
            use_half_precision=hp, provenance=acc_prov, aggregation=aggs,
            verbose=True, require_cuda_projector=cfg.get("require_cuda_projector", True),
            members=members, **slice_kw)
    elif p["dataset"] == "wikitext2":
        from corrected_lds.pools.tasks.wikitext_gpt2 import make_model
        mctor = lambda: make_model(recipe, device=args.device)
        tauqs = trakmod.compute_tauq_lm(
            ckpts, data, mctor, tracked=tuple(cfg.get("tracked") or ()),
            proj_dim=cfg["proj_dim"],
            device=args.device, batch=int(cfg.get("trak_batch", 8)), work_dir=work_dir,
            scratch_dir=scratch, use_half_precision=hp, provenance=acc_prov,
            aggregation=aggs, verbose=True,
            require_cuda_projector=cfg.get("require_cuda_projector", True),
            members=members, **slice_kw)
    elif p["dataset"] == "imagenet":
        from corrected_lds.pools.tasks.imagenet_resnet18 import featurize_batches, make_model
        mctor = lambda: make_model(recipe, num_classes=data["meta"]["n_classes"],
                                   device=args.device)
        # decode from the blob at the val geometry; save_every=5 paces ~10 GB accumulator writes
        train_batches = lambda: featurize_batches(data, int(cfg.get("trak_batch", 64)),
                                                  args.device, recipe.val_res,
                                                  recipe.crop_ratio)
        tauqs = trakmod.compute_tauq(
            ckpts, data, mctor, proj_dim=cfg["proj_dim"], device=args.device,
            batch=int(cfg.get("trak_batch", 64)), work_dir=work_dir, scratch_dir=scratch,
            use_half_precision=hp, provenance=acc_prov, aggregation=aggs, verbose=True,
            require_cuda_projector=cfg.get("require_cuda_projector", True),
            train_batches=train_batches, save_every=5, members=members, **slice_kw)
    elif p["dataset"] == "mnist":
        from corrected_lds.pools.tasks.mnist_mlp import make_model
        mctor = lambda: make_model(recipe, data["meta"]["n_classes"], device=args.device)
        tauqs = trakmod.compute_tauq(
            ckpts, data, mctor, proj_dim=cfg["proj_dim"], device=args.device,
            batch=int(cfg.get("trak_batch", 256)), work_dir=work_dir,
            scratch_dir=scratch, use_half_precision=hp, provenance=acc_prov,
            aggregation=aggs, verbose=True,
            require_cuda_projector=cfg.get("require_cuda_projector", False),
            members=members, **slice_kw)
    elif p["dataset"] == "cifar10":
        from corrected_lds.pools.tasks.cifar_resnet9 import make_model
        mctor = lambda: make_model(num_classes=data["meta"]["n_classes"],
                                   in_ch=data["meta"]["in_ch"], device=args.device)
        feat_aug = (trakmod.make_feat_aug(recipe)
                    if cfg.get("featurize_augment") else None)
        tauqs = trakmod.compute_tauq(
            ckpts, data, mctor, proj_dim=cfg["proj_dim"], device=args.device,
            batch=int(cfg.get("trak_batch", 256)), work_dir=work_dir,
            scratch_dir=scratch, use_half_precision=hp, provenance=acc_prov,
            aggregation=aggs, verbose=True,
            require_cuda_projector=cfg.get("require_cuda_projector", False),
            featurize_augment=feat_aug, members=members, **slice_kw)
    else:
        raise SystemExit(f"[tau] no TRAK model for dataset {p['dataset']!r}")
    if members is not None:                       # pieces-only: nothing to write yet
        print(f"[tau] pieces cached for members {tauqs['pieces']} under "
              f"{trakmod.pieces_path(work_dir, 0).rsplit(os.sep, 1)[0]}; re-run "
              f"without --members (after every member is cached) to accumulate "
              f"{out_names}.")
        if scratch != work_dir:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)
        return 0
    from corrected_lds.banks.artifacts import write_pool_tda, write_tauq
    if block is not None:   # slices only: a pooled tauq needs the whole roster's sums
        import shutil
        span = {"members": [block[0], block[-1] + 1], "n_models": len(block)}
        tag = f"b{block[0]}-{block[-1]}"
        for kind, ok, agg, ef, g in (
                ("units", want_units, "separated-avg-eq15", e_feat, tauqs.get("g_units")),
                ("members", want_pm, "plain-mean", 1, tauqs.get("g_members_pm"))):
            if not ok:
                continue
            path = os.path.join(pool_dir, f"pool_tda_trak{int(ef)}_{tag}.npz")
            write_pool_tda(path, g, data["meta"], subsets=pool.subsets,
                           method=f"trak{int(ef)}",
                           inner="separated-avg-eq15" if kind == "units" else "single-member",
                           e_features=int(ef), seed_base=trakmod.ENS_SEED_BASE,
                           provenance={**provs[agg, n_models], **span})
            print(f"[tau] wrote {path} (pool_tda g {g.shape}, members "
                  f"{block[0]}-{block[-1]}); append it with `corrected-lds merge-pool-tda`.")
        shutil.rmtree(work_dir, ignore_errors=True)   # the block's accumulator; pieces stay
        if scratch != work_dir:
            shutil.rmtree(scratch, ignore_errors=True)
        return 0
    snapshots = tauqs.get("snapshots", {})
    for (a, e), path in outs.items():
        t = tauqs[a] if e == n_models else snapshots[e][a]
        write_tauq(path, t, data["meta"], n_models=int(e),
                   proj_dim=int(cfg["proj_dim"]),
                   provenance=provs[a, e])
        print(f"[tau] wrote {path} (tauq {t.shape}, trak, "
              f"aggregation={a}, n_models={e}).")
    if pool is not None:
        # named for the inner knob, pool_tda_trak<e_features>, so plain-mean is pool_tda_trak1
        if want_units and len(tauqs.get("g_units", ())):
            u_path = os.path.join(pool_dir, f"pool_tda_trak{int(e_feat)}.npz")
            write_pool_tda(
                u_path, tauqs["g_units"],
                data["meta"], subsets=pool.subsets, method=f"trak{int(e_feat)}",
                inner="separated-avg-eq15", e_features=e_feat,
                seed_base=trakmod.ENS_SEED_BASE,
                provenance=provs["separated-avg-eq15", n_models])
            print(f"[tau] wrote {u_path} "
                  f"(pool_tda g {tauqs['g_units'].shape}, replica = one Eq. 15 unit "
                  f"of e_features={e_feat} members).")
        if want_pm and len(tauqs.get("g_members_pm", ())):
            pm_path = os.path.join(pool_dir, "pool_tda_trak1.npz")
            write_pool_tda(
                pm_path, tauqs["g_members_pm"],
                data["meta"], subsets=pool.subsets, method="trak1",
                inner="single-member", e_features=1,
                seed_base=trakmod.ENS_SEED_BASE, provenance=provs["plain-mean", n_models])
            print(f"[tau] wrote {pm_path} "
                  f"(pool_tda g {tauqs['g_members_pm'].shape}, replica = one member, "
                  f"i.e. the e_features=1 member of the TRAK family).")
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)               # accumulator no longer needed
    if scratch != work_dir:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def _cmd_merge_pool_tda(args):
    """Join ``pool_tda`` parts along the replica axis (CPU): the released file plus the
    blocks that extend it, or the blocks of one sharded featurize pass."""
    from corrected_lds.banks.artifacts import concat_pool_tda, load_pool, load_pool_tda
    pool = load_pool(args.pool)
    total = 0
    for part in args.parts:
        t = load_pool_tda(part)
        total += t.R_tda
        print(f"[merge] {part}: R_tda={t.R_tda} e_features={t.e_features} "
              f"inner={t.inner} members={t.provenance.get('members', 'whole roster')}")
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"[merge] {args.out} exists; pass --force to overwrite.")
    out = concat_pool_tda(args.out, args.parts, pool)
    t = load_pool_tda(out)
    if t.R_tda != total:
        raise SystemExit(f"[merge] {out} has R_tda={t.R_tda}, expected {total}.")
    print(f"[merge] wrote {out} (g {t.g.shape}, R_tda={t.R_tda}, "
          f"e_features={t.e_features}, members={t.provenance.get('members', 'whole roster')})"
          f" -> measured E up to {t.R_tda}.")
    return 0


def _cmd_compute_tau_tracin(args):
    """The TracIn flow: full-set trajectory checkpoints under ``tracin_ckpts/`` (not the
    TRAK members' ``ckpts/``), per-run partials, plain-mean combine."""
    from corrected_lds.tda import tracin as tracinmod
    p = PRESETS[args.preset]
    cfg = p.get("tracin")
    if cfg is None:
        raise SystemExit(f"[tau] preset {args.preset!r} has no tracin block; its methods: "
                         f"{_methods_of(args.preset)}.")
    _, recipe, data = _load_preset_data(args.preset, args.device)
    n_models = args.n_models or max(cfg["ensemble_sizes"])
    targs, cps, prov = _tracin_setup(p, recipe, n_models)
    pool_dir = os.path.join(args.root, args.tag)
    traj_dir = os.path.join(pool_dir, "tracin_ckpts")
    out = os.path.join(pool_dir, f"tauq_tracin_e{n_models}.npz")
    if os.path.exists(out):
        z = np.load(out, allow_pickle=False)
        stored = str(z["provenance"]) if "provenance" in z.files else None
        want = json.dumps(prov, sort_keys=True, default=str)
        if stored == want:
            print(f"[tau] {out} present (provenance ok) -> reuse.")
            ptda = os.path.join(pool_dir, "pool_tda_tracin.npz")
            wd = os.path.join(pool_dir, f"tau_work_tracin_e{n_models}")
            if not os.path.exists(ptda) and os.path.isdir(wd):  # backfill from partials
                pool = _pool_for_tda(pool_dir, data)
                if pool is not None:
                    _emit_pool_tda_partials(
                        ptda, wd, pool, data["meta"], n_models,
                        tracinmod.run_partial_path, method="tracin",
                        inner="single-trajectory",
                        seed_base=tracinmod.TRACIN_SEED_BASE, provenance=prov)
            return 0
        raise SystemExit(f"[tau] {out} exists with DIFFERENT provenance; refusing to "
                         f"overwrite. Delete it or use a fresh tag.")

    k = targs.units_per_epoch
    have = sum(all(os.path.exists(q)
                   for q in tracinmod.trajectory_paths(traj_dir, r, cps, k))
               for r in range(n_models))
    # a full-set run costs ~2 alpha=0.5 cells; featurize ~ one grad pass per checkpoint
    est_h = ((n_models - have) * 2 * _CELL_MINUTES[p["dataset"]]
             + n_models * len(cps) * _FEAT_MINUTES[p["dataset"]]) / 60
    if p.get("harness") == "bergson":     # nothing is trained here: see the branch below
        est_h = n_models * len(cps) * _FEAT_MINUTES[p["dataset"]] / 60
        print(f"[tau] {out}: {have}/{n_models} trajectory run(s) imported from bergson; "
              f"this run would score {n_models} x {len(cps)} checkpoints, "
              f"~{est_h:.1f} GPU-hours.")
    else:
        print(f"[tau] {out}: {have}/{n_models} trajectory run(s) present; this run would "
              f"train {n_models - have} full-set run(s) + score {n_models} x {len(cps)} "
              f"checkpoints, ~{est_h:.1f} GPU-hours.")
    if not args.yes:
        print("[tau] dry run only — pass --yes to start (never auto-run).")
        return 2

    want = None if args.runs is None else _parse_runs(args.runs, n_models)
    trajs, mctor = _trajectories(p, recipe, data, traj_dir, n_models, cps, k,
                                 targs.ckpt_every, args.device, runs=want)
    work_dir = os.path.join(pool_dir, f"tau_work_tracin_e{n_models}")
    tauq = tracinmod.compute_tauq_tracin(
        trajs, data, mctor, recipe=recipe, args=targs, proj_dim=int(cfg["proj_dim"]),
        device=args.device, batch=int(cfg.get("batch", 256)), work_dir=work_dir,
        provenance=prov, verbose=True, runs=want,
        require_cuda_projector=cfg.get("require_cuda_projector", False))
    if want is not None:
        print(f"[tau] wrote partials for runs {want} into {work_dir}. Re-run WITHOUT "
              f"--runs once every worker has finished to combine and write {out}.")
        return 0
    from corrected_lds.banks.artifacts import write_tauq
    write_tauq(out, tauq, data["meta"], n_models=n_models, proj_dim=int(cfg["proj_dim"]),
               provenance=prov)
    print(f"[tau] wrote {out} (tauq {tauq.shape}, tracin, n_models={n_models} x T={len(cps)}).")
    pool = _pool_for_tda(pool_dir, data)
    emitted = pool is not None and _emit_pool_tda_partials(
        os.path.join(pool_dir, "pool_tda_tracin.npz"), work_dir, pool, data["meta"],
        n_models, tracinmod.run_partial_path, method="tracin",
        inner="single-trajectory", seed_base=tracinmod.TRACIN_SEED_BASE,
        provenance=prov)
    import shutil
    if emitted and not getattr(args, "keep_partials", False):
        shutil.rmtree(work_dir, ignore_errors=True)           # partials no longer needed
    else:
        print(f"[tau] keeping {work_dir} (per-run partials) so pool_tda_tracin can be "
              f"emitted once the pool exists — re-run compute-tau then.")
    return 0


def _cmd_compute_tau_source(args):
    """The SOURCE flow: TracIn's full-set trajectories walked over L segments, thinned to the
    preset's ``ckpt_every`` by file selection (``_provenance.json`` omits it), so SOURCE trains
    nothing new where TracIn already checkpoints every epoch (``cifar10``)."""
    from corrected_lds.tda import source as srcmod
    from corrected_lds.tda import tracin as tracinmod
    p = PRESETS[args.preset]
    cfg = p.get("source")
    if cfg is None:
        raise SystemExit(f"[tau] preset {args.preset!r} has no source block; its methods: "
                         f"{_methods_of(args.preset)}.")
    _, recipe, data = _load_preset_data(args.preset, args.device)
    n_models = args.n_models or max(cfg["ensemble_sizes"])
    n_seg = int(cfg["segments"])
    sargs, cps, prov = _source_setup(p, recipe, n_models, n_seg, data["meta"]["n_train"])
    pool_dir = os.path.join(args.root, args.tag)
    traj_dir = os.path.join(pool_dir, "tracin_ckpts")        # shared with tracin
    # L is in the artifact name, so L=1 and L=3 coexist in one tag on the same runs
    tag_l = f"L{sargs.segments}"
    out = os.path.join(pool_dir, f"tauq_source_{tag_l}_e{n_models}.npz")
    if os.path.exists(out):
        z = np.load(out, allow_pickle=False)
        stored = str(z["provenance"]) if "provenance" in z.files else None
        if stored == json.dumps(prov, sort_keys=True, default=str):
            print(f"[tau] {out} present (provenance ok) -> reuse.")
            ptda = os.path.join(pool_dir, f"pool_tda_source_{tag_l}.npz")
            wd = os.path.join(pool_dir, f"tau_work_source_{tag_l}_e{n_models}")
            if not os.path.exists(ptda) and os.path.isdir(wd):   # backfill from partials
                pool = _pool_for_tda(pool_dir, data)
                if pool is not None:
                    _emit_pool_tda_partials(
                        ptda, wd, pool, data["meta"], n_models,
                        srcmod.run_partial_path, method=f"source_{tag_l}",
                        inner="single-trajectory",
                        seed_base=tracinmod.TRACIN_SEED_BASE, provenance=prov)
            return 0
        raise SystemExit(f"[tau] {out} exists with DIFFERENT provenance; refusing to "
                         f"overwrite. Delete it or use a fresh tag.")

    have = sum(all(os.path.exists(q) for q in tracinmod.trajectory_paths(traj_dir, r, cps))
               for r in range(n_models))
    # ~2 alpha=0.5 cells per full-set run; per checkpoint, a factor fit + one pass per query chunk
    n_chunks = max(1, -(-int(data["meta"]["n_targets"]) // sargs.query_chunk))
    est_h = ((n_models - have) * 2 * _CELL_MINUTES[p["dataset"]]
             + n_models * len(cps) * (1 + n_chunks) * _FEAT_MINUTES[p["dataset"]]) / 60
    print(f"[tau] {out}: {have}/{n_models} trajectory run(s) present; this run would train "
          f"{n_models - have} full-set run(s), then fit factors at {len(cps)} checkpoint(s) "
          f"in L={sargs.segments} segment(s) and score {n_chunks} query chunk(s) per run, "
          f"~{est_h:.1f} GPU-hours.")
    if not args.yes:
        print("[tau] dry run only — pass --yes to start (never auto-run).")
        return 2

    if p["dataset"] == "mnist":
        from corrected_lds.pools.tasks.mnist_mlp import make_model, train_mnist_model
        hooked = lambda data_, sub, seed, hook: train_mnist_model(
            data_, sub, seed, recipe, args.device, verbose=True, on_epoch_end=hook)
        mctor = lambda: make_model(recipe, data["meta"]["n_classes"],
                                   device=args.device)
    elif p["dataset"] == "cifar10":
        from corrected_lds.pools.tasks.cifar_resnet9 import make_model, train_paper_model
        hooked = lambda data_, sub, seed, hook: train_paper_model(
            data_, sub, seed, recipe, args.device, verbose=True, on_epoch_end=hook)
        mctor = lambda: make_model(num_classes=data["meta"]["n_classes"],
                                   in_ch=data["meta"]["in_ch"], device=args.device)
    else:
        raise SystemExit(f"[tau] no SOURCE trainer for dataset {p['dataset']!r}")
    # --runs must shard the training too, or every worker retrains every trajectory
    tracinmod.train_trajectories(traj_dir, data, recipe, n_models, hooked,
                                 ckpt_every=_traj_ckpt_every(p, sargs),
                                 runs=(None if args.runs is None
                                       else _parse_runs(args.runs, n_models)))
    trajs = [tracinmod.trajectory_paths(traj_dir, r, cps) for r in range(n_models)]
    work_dir = os.path.join(pool_dir, f"tau_work_source_{tag_l}_e{n_models}")
    if args.runs is not None:        # one worker of a multi-GPU split: partials only
        want = _parse_runs(args.runs, n_models)
        srcmod.compute_source_partials(trajs, data, mctor, recipe=recipe, args=sargs,
                                       device=args.device, work_dir=work_dir,
                                       provenance=prov, runs=want, verbose=True)
        print(f"[tau] wrote partials for runs {want} into {work_dir}. Re-run WITHOUT "
              f"--runs once every worker has finished to combine and write {out}.")
        return 0
    tauq = srcmod.compute_tauq_source(trajs, data, mctor, recipe=recipe, args=sargs,
                                      device=args.device, work_dir=work_dir,
                                      provenance=prov, verbose=True)
    from corrected_lds.banks.artifacts import write_tauq
    write_tauq(out, tauq, data["meta"], n_models=n_models, provenance=prov)
    print(f"[tau] wrote {out} (tauq {tauq.shape}, source, n_models={n_models} x "
          f"C={len(cps)}, L={sargs.segments}).")
    pool = _pool_for_tda(pool_dir, data)
    emitted = pool is not None and _emit_pool_tda_partials(
        os.path.join(pool_dir, f"pool_tda_source_{tag_l}.npz"), work_dir, pool,
        data["meta"], n_models, srcmod.run_partial_path, method=f"source_{tag_l}",
        inner="single-trajectory", seed_base=tracinmod.TRACIN_SEED_BASE, provenance=prov)
    import shutil
    if emitted and not getattr(args, "keep_partials", False):
        shutil.rmtree(work_dir, ignore_errors=True)
    elif emitted:
        print(f"[tau] keeping {work_dir} (--keep-partials): combine_source_partials can "
              f"rebuild any single-run or subset tauq without re-scoring.")
    else:
        print(f"[tau] keeping {work_dir} (per-run partials) so pool_tda_source can be "
              f"emitted once the pool exists — re-run compute-tau then.")
    return 0


# -- harness="bergson" banks ------------------------------------------------ #
# The attributors score full-set runs that bergson trained, imported into
# tracin_ckpts/.
def _bergson_trajectories(p, traj_dir, n_runs, cps, k, device, runs=None):
    """``(per-run checkpoint lists, model_ctor)`` for a bergson bank's imported runs; a
    foreign directory stamp or a missing checkpoint is an error, as nothing is retrained."""
    from corrected_lds.pools import bergson
    from corrected_lds.pools.tasks.wikitext_gpt2 import make_model
    from corrected_lds.tda import tracin as tracinmod
    recipe = p["recipe"]
    bergson.check_trajectory_stamp(traj_dir, bergson.trajectory_stamp(
        recipe, p["n_train"], k, tracinmod.TRACIN_SEED_BASE))
    trajs = [tracinmod.trajectory_paths(traj_dir, r, cps, k) for r in range(n_runs)]
    missing = [r for r in (range(n_runs) if runs is None else runs)
               if not all(os.path.exists(q) for q in trajs[r])]
    if missing:
        raise SystemExit(f"[tau] bergson trajectory run(s) {missing} are not imported "
                         f"into {traj_dir}; train and import them first "
                         f"(`corrected-lds import-bergson-trajectory`; see "
                         f"docs/REPRODUCE.md).")
    return trajs, (lambda: make_model(recipe, device=device))


def _trajectories(p, recipe, data, traj_dir, n_runs, cps, k, ckpt_every, device, runs=None):
    """``(per-run checkpoint lists, model_ctor)`` of the full-set runs TracIn and EK-FAC share:
    imported for a bergson bank, else trained here (only ``runs`` if given)."""
    if p.get("harness") == "bergson":     # trajectories are imported, never trained here
        return _bergson_trajectories(p, traj_dir, n_runs, cps, k, device, runs=runs)
    from corrected_lds.tda import tracin as tracinmod
    if p["dataset"] == "qnli":
        from corrected_lds.pools.tasks.qnli_bert import make_text_model, train_text_model
        hooked = lambda data_, sub, seed, hook: train_text_model(
            data_, sub, seed, recipe, device, verbose=True, on_epoch_end=hook)
        mctor = lambda: make_text_model(recipe, data["meta"]["n_classes"], device,
                                        pretrained=False)
    elif p["dataset"] == "mnist":
        from corrected_lds.pools.tasks.mnist_mlp import make_model, train_mnist_model
        hooked = lambda data_, sub, seed, hook: train_mnist_model(
            data_, sub, seed, recipe, device, verbose=True, on_epoch_end=hook)
        mctor = lambda: make_model(recipe, data["meta"]["n_classes"], device=device)
    elif p["dataset"] == "cifar10":
        from corrected_lds.pools.tasks.cifar_resnet9 import make_model, train_paper_model
        hooked = lambda data_, sub, seed, hook: train_paper_model(
            data_, sub, seed, recipe, device, verbose=True, on_epoch_end=hook)
        mctor = lambda: make_model(num_classes=data["meta"]["n_classes"],
                                   in_ch=data["meta"]["in_ch"], device=device)
    else:
        raise SystemExit(f"[tau] no trajectory trainer for dataset {p['dataset']!r}")
    # --runs shards training as well as scoring, or every worker retrains every trajectory
    tracinmod.train_trajectories(traj_dir, data, recipe, n_runs, hooked,
                                 ckpt_every=ckpt_every, runs=runs)
    # the scored grid: a pinned checkpoint_epochs schedule scores a subset of the saved epochs
    return [tracinmod.trajectory_paths(traj_dir, r, cps, k) for r in range(n_runs)], mctor


def _cmd_compute_tau_ekfac_trajectories(args):
    """EK-FAC at the final checkpoint of each full-set run, damping from the preset: Bae et
    al.'s IF protocol, used for every released ``pool_tda_ekfac``.  The runs are TracIn's
    trajectories (``tracin_ckpts/``, seeds TRACIN_SEED_BASE + r), never TRAK's alpha=0.5 subset
    members: IF wants full-set models."""
    p = PRESETS[args.preset]
    tcfg, ecfg = p.get("tracin"), p.get("ekfac")
    if tcfg is None:
        raise SystemExit(f"[tau] preset {args.preset!r} has no tracin block, i.e. no "
                         f"full-set trajectory layout for EK-FAC to score.")
    if ecfg is None or "damping" not in ecfg:
        raise SystemExit(f"[tau] preset {args.preset!r} has no ekfac block with an explicit "
                         f"damping (1e-8 in the released banks; None asks for kronfluence's "
                         f"heuristic).")
    from corrected_lds.tda import ekfac as ekfacmod
    from corrected_lds.tda import tracin as tracinmod
    _, recipe, data = _load_preset_data(args.preset, args.device)
    n_models = args.n_models or max(ecfg.get("ensemble_sizes", tcfg["ensemble_sizes"]))
    eargs, cps, k, prov = _ekfac_setup(p, recipe, n_models)
    pool_dir = os.path.join(args.root, args.tag)
    traj_dir = os.path.join(pool_dir, "tracin_ckpts")
    out = os.path.join(pool_dir, f"tauq_ekfac_e{n_models}.npz")
    work_dir = os.path.join(pool_dir, f"tau_work_ekfac_e{n_models}")
    ptda = os.path.join(pool_dir, "pool_tda_ekfac.npz")
    if os.path.exists(out):
        z = np.load(out, allow_pickle=False)
        stored = str(z["provenance"]) if "provenance" in z.files else None
        if stored != json.dumps(prov, sort_keys=True, default=str):
            raise SystemExit(f"[tau] {out} exists with DIFFERENT provenance; refusing to "
                             f"overwrite. Delete it or use a fresh tag.")
        print(f"[tau] {out} present (provenance ok) -> reuse.")
        if not os.path.exists(ptda) and os.path.isdir(work_dir):   # backfill from partials
            pool = _pool_for_tda(pool_dir, data)
            if pool is not None:
                _emit_pool_tda_partials(ptda, work_dir, pool, data["meta"], n_models,
                                        ekfacmod.partial_path, method="ekfac",
                                        inner="single-member",
                                        seed_base=tracinmod.TRACIN_SEED_BASE,
                                        provenance=prov)
        return 0

    have = sum(all(os.path.exists(q) for q in tracinmod.trajectory_paths(traj_dir, r, cps, k))
               for r in range(n_models))
    # a full-set run costs ~2 alpha=0.5 cells (bergson's are imported, never trained here)
    n_train = 0 if p.get("harness") == "bergson" else n_models - have
    est_h = (n_train * 2 * _CELL_MINUTES[p["dataset"]]
             + n_models * 2 * _FEAT_MINUTES[p["dataset"]]) / 60
    print(f"[tau] {out}: {have}/{n_models} full-set run(s) present; this run would train "
          f"{n_train} + score EK-FAC at the final checkpoint of {n_models}, "
          f"~{est_h:.1f} GPU-hours.")
    if not args.yes:
        print("[tau] dry run only — pass --yes to start (never auto-run).")
        return 2
    want = None if args.runs is None else _parse_runs(args.runs, n_models)
    trajs, mctor = _trajectories(p, recipe, data, traj_dir, n_models, cps, k,
                                 int(tcfg["ckpt_every"]), args.device, runs=want)
    tauq = ekfacmod.compute_tauq_ekfac([t[-1] for t in trajs], data, mctor, args=eargs,
                                       device=args.device, work_dir=work_dir,
                                       provenance=prov, verbose=True, members=want)
    if want is not None:
        print(f"[tau] wrote partials for runs {want} into {work_dir}. Re-run WITHOUT "
              f"--runs once every worker has finished to combine and write {out}.")
        return 0
    from corrected_lds.banks.artifacts import write_tauq
    write_tauq(out, tauq, data["meta"], n_models=n_models, provenance=prov)
    print(f"[tau] wrote {out} (tauq {tauq.shape}, ekfac, {n_models} full-set runs).")
    pool = _pool_for_tda(pool_dir, data)
    emitted = pool is not None and _emit_pool_tda_partials(
        ptda, work_dir, pool, data["meta"], n_models, ekfacmod.partial_path,
        method="ekfac", inner="single-member", seed_base=tracinmod.TRACIN_SEED_BASE,
        provenance=prov)
    if emitted and not getattr(args, "keep_partials", False):
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)       # partials no longer needed
    else:
        print(f"[tau] keeping {work_dir} (per-run partials) so pool_tda_ekfac can be "
              f"emitted once the pool exists — re-run compute-tau then.")
    return 0


_IMPORT_GATE_NATS = 0.02     # |mean query loss - pool baseline mean|; seed sd is ~0.002


def _bank_baselines(pool_dir):
    """``(baselines, path)`` of the bank's in-loop readout, or ``(None, None)``; only that
    readout has full-data baselines (``pool_dropout_readout.npz`` if ``pool.npz`` is eval)."""
    for name in ("pool.npz", "pool_dropout_readout.npz"):
        path = os.path.join(pool_dir, name)
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                if "baselines" in z.files:
                    return z["baselines"], path
    return None, None


def _cmd_import_bergson_trajectory(args):
    """One bergson-trained attributor run -> ``tracin_ckpts/run_RR/``, with two gates: every
    checkpoint loads into our GPT-2 with ``strict=True``, and the final checkpoint, read as
    the pool read its full-data baselines (same recipe, other seeds, dropout active), must
    land within seed spread of them, which a wrong transpose or dataset revision cannot."""
    import torch

    from corrected_lds.pools import bergson
    from corrected_lds.pools.tasks.wikitext_gpt2 import make_model, query_losses
    from corrected_lds.tda import tracin as tracinmod
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson":
        raise SystemExit(f"[import] preset {args.preset!r} is not a bergson bank.")
    cfg, recipe = p["tracin"], p["recipe"]
    n_runs, k = max(cfg["ensemble_sizes"]), int(cfg.get("units_per_epoch", 1))
    if not 0 <= args.run < n_runs:
        raise SystemExit(f"[import] --run {args.run} outside 0..{n_runs - 1}.")
    run_dir = os.path.dirname(os.path.abspath(args.export_dir.rstrip("/")))
    seed = bergson.run_seed(run_dir)
    if seed is not None and seed != tracinmod.TRACIN_SEED_BASE + args.run:
        raise SystemExit(f"[import] {run_dir} trained with seed {seed}, but run {args.run} "
                         f"must be seed {tracinmod.TRACIN_SEED_BASE + args.run}.")
    pool_dir = os.path.join(args.root, args.tag)
    traj_dir = os.path.join(pool_dir, "tracin_ckpts")
    bergson.check_trajectory_stamp(traj_dir, bergson.trajectory_stamp(
        recipe, p["n_train"], k, tracinmod.TRACIN_SEED_BASE), write=True)
    steps = bergson.trajectory_steps(recipe, p["n_train"], k)
    paths = tracinmod.trajectory_paths(traj_dir, args.run, range(1, len(steps) + 1), k)
    model = make_model(recipe, device=args.device)
    bergson.import_trajectory(args.export_dir, paths, steps, model)
    model.load_state_dict(torch.load(paths[-1], map_location=args.device,
                                     weights_only=True))
    _, _, data = _load_preset_data(args.preset, args.device)
    # read as the pool read its baselines (dropout under train_mode); eval mode for the record
    q = query_losses(model, data, args.device, dropout=bool(recipe.train_mode))
    q_eval = query_losses(model, data, args.device)
    msg = (f"[import] run {args.run} (seed {seed}): final mean query loss {q.mean():.4f} "
           f"as the pool reads it (dropout={bool(recipe.train_mode)}), "
           f"{q_eval.mean():.4f} in eval mode")
    base, pool_path = _bank_baselines(pool_dir)
    if base is not None:
        gap = abs(float(q.mean()) - float(base.mean()))
        print(f"{msg}; pool baselines {base.mean():.4f} (per-seed sd "
              f"{base.mean(axis=1).std(ddof=1):.4f}, {os.path.basename(pool_path)}), "
              f"gap {gap:.4f}")
        if gap > _IMPORT_GATE_NATS:
            import shutil  # the scorers must find this run missing, not plausible
            shutil.rmtree(os.path.dirname(paths[0]), ignore_errors=True)
            raise SystemExit(f"[import] gate FAILED: gap {gap:.4f} > {_IMPORT_GATE_NATS} "
                             f"nats -- the imported model does not read the pool's "
                             f"baselines. Run {args.run}'s checkpoints were removed.")
    else:
        print(f"{msg}; no pool with baselines in {pool_dir}, baseline gate skipped.")
    print(f"[import] wrote {len(paths)} checkpoint(s) under {os.path.dirname(paths[0])}")
    return 0


def _bergson_members(p, ckpt_dir, n_models, scope):
    """Index-ordered member checkpoint paths of a bergson bank's TRAK ensemble: the directory
    must carry the bergson member stamp and every member in ``scope`` must be imported."""
    from corrected_lds.pools import bergson
    from corrected_lds.tda import trak as trakmod
    bergson.check_trajectory_stamp(
        ckpt_dir, bergson.member_stamp(p["recipe"], p["n_train"], trakmod.ENS_SEED_BASE),
        importer="import-bergson-member")
    paths = [os.path.join(ckpt_dir, f"member_{m:03d}.pt") for m in range(int(n_models))]
    missing = [m for m in scope if not os.path.exists(paths[m])]
    if missing:
        raise SystemExit(f"[tau] TRAK member(s) {missing} are not imported into "
                         f"{ckpt_dir}; train them with bergson and run `corrected-lds "
                         f"import-bergson-member` (see docs/REPRODUCE.md).")
    return paths


def _cmd_import_bergson_member(args):
    """One bergson-trained TRAK member -> ``ckpts/member_MMM.pt``, gated on its seed
    (``ENS_SEED_BASE + m``), its data (member ``m``'s half of our rows, by row list and hash)
    and a ``strict=True`` load; a half-data model has no baseline: its eval loss is reported."""
    import torch

    from corrected_lds.pools import bergson
    from corrected_lds.pools.tasks.wikitext_gpt2 import make_model, query_losses
    from corrected_lds.tda import trak as trakmod
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson" or "trak" not in p:
        raise SystemExit(f"[import] preset {args.preset!r} is not a bergson bank with a "
                         f"trak block.")
    recipe, m = p["recipe"], int(args.member)
    if m < 0:
        raise SystemExit(f"[import] --member {m} must be >= 0.")
    run_dir = os.path.dirname(os.path.abspath(args.export_dir.rstrip("/")))
    seed = bergson.run_seed(run_dir)
    if seed != trakmod.ENS_SEED_BASE + m:
        raise SystemExit(f"[import] {run_dir} trained with seed {seed}, but member {m} "
                         f"must be seed {trakmod.ENS_SEED_BASE + m}.")
    _, _, data = _load_preset_data(args.preset, args.device)
    with open(args.data_meta) as fh:
        meta = json.load(fh)
    try:
        bergson.check_member_data(meta, m, p["n_train"], recipe.alpha,
                                  data["lm_ids"].cpu().numpy())
    except ValueError as e:
        raise SystemExit(f"[import] {e}") from e
    ckpt_dir = os.path.join(args.root, args.tag, "ckpts")
    bergson.check_trajectory_stamp(
        ckpt_dir, bergson.member_stamp(recipe, p["n_train"], trakmod.ENS_SEED_BASE),
        write=True, importer="import-bergson-member")
    out = os.path.join(ckpt_dir, f"member_{m:03d}.pt")
    model = make_model(recipe, device=args.device)
    bergson.import_trajectory(args.export_dir, [out],
                              [bergson.member_final_step(recipe, p["n_train"])], model)
    model.load_state_dict(torch.load(out, map_location=args.device, weights_only=True))
    q = query_losses(model, data, args.device)
    msg = (f"[import] member {m} (seed {seed}, {len(meta['rows'])} rows): eval-mode mean "
           f"query loss {q.mean():.4f}")
    pool_path = os.path.join(args.root, args.tag, "pool.npz")
    if os.path.exists(pool_path):
        with np.load(pool_path, allow_pickle=False) as z:
            if "losses" in z.files:
                msg += (f"; the pool's half-data retrains read {z['losses'].mean():.4f} "
                        f"({os.path.basename(pool_path)}, weight-zeroed, full length)")
    print(msg)
    print(f"[import] wrote {out}")
    return 0


def _cmd_bergson_source_check(args):
    """Baseline-loss gate for one SOURCE attributor run: ``import-bergson-trajectory``'s
    final-checkpoint gate, without importing anything, plus the seed bergson recorded."""
    import torch
    from safetensors.torch import load_file

    from corrected_lds.pools import bergson
    from corrected_lds.pools.tasks.wikitext_gpt2 import hf_to_ours, make_model, query_losses
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson":
        raise SystemExit(f"[check] preset {args.preset!r} is not a bergson bank.")
    recipe = p["recipe"]
    run_dir = os.path.dirname(os.path.abspath(args.export_dir.rstrip("/")))
    seed = bergson.run_seed(run_dir)
    if args.expect_seed is not None and seed != args.expect_seed:
        raise SystemExit(f"[check] {run_dir} was trained with seed {seed}, expected "
                         f"{args.expect_seed}")
    k = int(p["tracin"].get("units_per_epoch", 1))
    last = bergson.trajectory_steps(recipe, p["n_train"], k)[-1]
    src = os.path.join(args.export_dir, f"checkpoint-{last}", "model.safetensors")
    model = make_model(recipe, device=args.device)
    model.load_state_dict(hf_to_ours(load_file(src)), strict=True)
    _, _, data = _load_preset_data(args.preset, args.device)
    with torch.no_grad():
        q = query_losses(model, data, args.device, dropout=bool(recipe.train_mode))
    base, path = _bank_baselines(args.baselines_dir)
    if base is None:
        raise SystemExit(f"[check] no pool with baselines in {args.baselines_dir}")
    gap = abs(float(q.mean()) - float(base.mean()))
    print(f"[check] {run_dir} (seed {seed}): final mean query loss {q.mean():.4f} "
          f"(dropout={bool(recipe.train_mode)}); baselines {base.mean():.4f} "
          f"({os.path.basename(path)}), gap {gap:.4f}")
    if gap > _IMPORT_GATE_NATS:
        raise SystemExit(f"[check] gate FAILED: gap {gap:.4f} > {_IMPORT_GATE_NATS} nats")
    return 0


def _cmd_bergson_source_config(args):
    """Write one pod's SOURCE driver config + its bootstrap script."""
    from corrected_lds.pools import bergson_source as bs
    gpus = _parse_runs(args.gpus, 1024)
    cfg = bs.SourceConfig(
        preset=args.preset, runs=_parse_runs(args.runs, 1024) if args.runs else [],
        lanes=bs._parse_lanes(gpus, args.shm_lanes, args.shm_dir, args.disk_dir),
        root=args.root, pool=args.pool, baselines_dir=args.baselines_dir,
        venv_bergson=args.venv_bergson, venv_ours=args.venv_ours, site_fix=args.site_fix,
        data_dir=args.data_dir, check_run=args.check_run,
        check_ref=args.check_ref, query_batch=args.query_batch, piece_gb=args.piece_gb)
    if (cfg.check_run is None) != (cfg.check_ref is None):
        raise SystemExit("[source] --check-run and --check-ref go together")
    if not cfg.runs and cfg.check_run is None:
        raise SystemExit("[source] nothing to do: give --runs and/or --check-run")
    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "source_config.json")
    cfg.dump(cfg_path)
    boot = os.path.join(args.out, "bootstrap_source.sh")
    with open(boot, "w") as fh:
        fh.write(bs.render_source_bootstrap(venv_ours=args.venv_ours,
                                            venv_bergson=args.venv_bergson,
                                            site_fix=args.site_fix, data_dir=args.data_dir))
    os.chmod(boot, 0o755)
    print(f"[source] runs {cfg.runs} on {len(cfg.lanes)} lanes "
          f"({sum(1 for _, d in cfg.lanes if d == args.shm_dir)} on {args.shm_dir}); "
          f"check run {cfg.check_run}\n[source] wrote {cfg_path} and {boot}")
    return 0


def _cmd_bergson_source_run(args):
    from corrected_lds.pools import bergson_source as bs
    return bs.SourceDriver(bs.SourceConfig.load(args.config)).run()


def _cmd_bergson_source_combine(args):
    """Single-run SOURCE tauqs -> tauq_<method>_e<E>.npz + pool_tda_<method>.npz, scored."""
    from corrected_lds.banks.artifacts import load_pool
    from corrected_lds.evaluate import evaluate_pool
    from corrected_lds.pools import bergson_source as bs
    pool = load_pool(args.pool)
    tq, pt = bs.combine_source_runs(args.parts, pool, args.out_dir)
    from corrected_lds.banks.artifacts import load_tauq
    r = evaluate_pool(pool, load_tauq(tq))["results"]
    print(f"[combine] {tq} + {pt}: naive {r['spearman']:.4f} | corrected "
          f"{r['corrected_rho_inf']:.4f} (SE {r['corrected_se_jackknife']:.4f}) on {args.pool}")
    return 0


def _cmd_read_bergson_retrained(args):
    """Eval-mode query losses of one chunk of ``validate --save_models`` checkpoints -> npz."""
    from corrected_lds.pools import bergson
    from corrected_lds.pools.tasks.wikitext_gpt2 import make_model
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson":
        raise SystemExit(f"[read] preset {args.preset!r} is not a bergson bank.")
    _, recipe, data = _load_preset_data(args.preset, args.device)
    losses = bergson.read_retrained(args.dir, args.lo, args.hi,
                                    make_model(recipe, device=args.device), data,
                                    args.device)
    tmp = args.out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, losses=losses, lo=int(args.lo), hi=int(args.hi), readout="eval")
    os.replace(tmp, args.out)
    print(f"[read] masks {args.lo}..{args.hi - 1}: eval-mode mean query loss "
          f"{losses.mean():.4f} -> {args.out}")
    return 0


def _cmd_bergson_pool(args):
    """A bergson bank's pool in the release's layout: ``setup`` writes the inputs and scripts
    into --dir, ``check`` compares the eval readouts done so far with --ref, and ``assemble``
    writes both readouts' pools into --out."""
    import sys

    from corrected_lds.pools import bergson
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson":
        raise SystemExit(f"[pool] preset {args.preset!r} is not a bergson bank.")
    seeds, spans, Q = p["seeds"], p["pool_spans"], p["n_targets"]
    M = spans[-1][1]
    if args.action == "setup":
        from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION, cache_dir
        bergson.setup_replication_inputs(args.dir, M, p["n_train"], p["recipe"].alpha, Q)
        with open(os.path.join(args.dir, "bootstrap.sh"), "w") as f:
            f.write(bergson.render_bootstrap(args.dir, args.venv_bergson))
        units = bergson.replication_units([(s, spans) for s in seeds], p["pool_chunk"])
        sh = bergson.write_replication_scripts(
            args.dir, venv_bergson=args.venv_bergson, venv_ours=sys.prefix, preset=args.preset,
            recipe=p["recipe"], M=M, Q=Q, repo=WIKITEXT2_REPO, revision=WIKITEXT2_REVISION,
            data_dir=cache_dir(), units=units)
        print(f"[pool] {len(units)} units over {len(seeds)} seeds x {M} masks: run "
              f"{args.dir}/bootstrap.sh once, then `{sh} <w> <gpu>` on each GPU")
    elif args.action == "check":
        with np.load(args.ref, allow_pickle=False) as z:
            ref_seeds, ref = [int(s) for s in z["seeds"]], z["losses"]
        gap = {}
        for s in seeds:
            d = os.path.join(args.dir, f"r{s}")
            for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if name.startswith("eval_") and name.endswith(".npz"):
                    with np.load(os.path.join(d, name), allow_pickle=False) as z:
                        lo, hi = int(z["lo"]), int(z["hi"])
                        gap[s, lo, hi] = np.abs(z["losses"] - ref[ref_seeds.index(s), lo:hi]).max()
        if not gap:
            raise SystemExit(f"[check] no eval-mode readouts under {args.dir} yet")
        worst = max(gap, key=gap.get)
        msg = (f"{len(gap)} unit(s) against {args.ref}: max |d| {gap[worst]:.1e} (seed "
               f"{worst[0]}, masks {worst[1]}..{worst[2] - 1})")
        if gap[worst] > 1e-4:     # the release's own gate; RTX 5090s matched it bit for bit
            raise SystemExit(f"[check] FAILED, {msg}")
        print(f"[check] OK, {msg}")
    else:
        paths = [os.path.join(args.out, n) for n in ("pool.npz", "pool_dropout_readout.npz")]
        if any(os.path.exists(x) for x in paths):
            raise SystemExit(f"[pool] {args.out} already holds a pool; refusing to overwrite it.")
        os.makedirs(args.out, exist_ok=True)
        masks = np.load(os.path.join(args.dir, "masks.npy"), allow_pickle=False)
        dirs = [os.path.join(args.dir, f"r{s}") for s in seeds]
        kw = {"alpha": p["recipe"].alpha, "recipe": p["recipe"].to_dict(), "Q": Q}
        bergson.build_eval_pool(paths[0], dirs, seeds, masks, **kw)
        bergson.build_pool(paths[1], dirs, seeds, masks, **kw)
        print(f"[pool] wrote {paths[0]} (eval readout) and {paths[1]}")
    return 0


def _cmd_bergson_ekfac(args):
    """bergson's own EK-FAC on one full-data model, seeded with the pool's first seed:
    ``setup`` writes bootstrap.sh, the configs and run_ekfac.sh into --dir, and ``import``, the
    script's last step, converts bergson's score store into the tauq --out."""
    import sys

    from corrected_lds.pools import bergson
    p = PRESETS[args.preset]
    if p.get("harness") != "bergson":
        raise SystemExit(f"[ekfac] preset {args.preset!r} is not a bergson bank.")
    n, Q = p["n_train"], p["n_targets"]
    if args.action == "setup":
        from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION
        sh = bergson.write_ekfac_scripts(
            args.dir, recipe=p["recipe"], seed=p["seeds"][0], n_train=n,
            units_per_epoch=p["tracin"]["units_per_epoch"], repo=WIKITEXT2_REPO,
            revision=WIKITEXT2_REVISION, venv_bergson=args.venv_bergson, venv_ours=sys.prefix,
            preset=args.preset, out=os.path.abspath(args.out))
        with open(os.path.join(args.dir, "bootstrap.sh"), "w") as f:
            f.write(bergson.render_bootstrap(args.dir, args.venv_bergson))
        print(f"[ekfac] run {args.dir}/bootstrap.sh once, then {sh} on one GPU")
    else:
        from corrected_lds.banks.artifacts import write_tauq
        from corrected_lds.banks.presets import bank_hashes
        if os.path.exists(args.out):
            raise SystemExit(f"[ekfac] {args.out} exists; refusing to overwrite it.")
        train_hash, query_hash = bank_hashes(p["dataset"])
        ident = {"dataset": p["dataset"], "n_train": n, "n_targets": Q,
                 "train_idx": np.arange(n), "query_idx": np.arange(Q),
                 "train_hash": train_hash, "query_hash": query_hash}
        store = os.path.join(args.dir, "ekfac", "scores")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_tauq(args.out, bergson.read_score_store(store), ident, provenance=dict(
            bergson.EKFAC_PROVENANCE, queries=f"full {Q}-doc validation split"))
        print(f"[ekfac] {store} -> {args.out}")
    return 0


def _parse_runs(spec, n_models):
    """'0-4' or '0,2,5' -> a sorted list of run indices, validated against n_models."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = sorted(r for r in out if not 0 <= r < n_models)
    if bad or not out:
        raise SystemExit(f"[tau] --runs {spec!r} selects {sorted(out) or 'nothing'}; "
                         f"valid indices are 0..{n_models - 1}.")
    return sorted(out)


def _traj_ckpt_every(preset, sargs):
    """The trajectory ``ckpt_every`` for SOURCE: the tracin block's, which must divide
    SOURCE's so both share one trajectory bank; otherwise ``train_trajectories`` would wipe
    and retrain the whole directory."""
    tracin_every = int(preset.get("tracin", {}).get("ckpt_every", sargs.ckpt_every))
    if sargs.ckpt_every % tracin_every == 0:
        return tracin_every                      # denser or equal: reuse tracin's bank
    raise SystemExit(f"[tau] source ckpt_every={sargs.ckpt_every} is not a multiple of "
                     f"the preset's tracin ckpt_every={tracin_every}; sharing "
                     f"tracin_ckpts/ would force a full retrain. Align them in presets.py.")


def _pool_tda_auto_key(tau):
    """The ``pool_tda_<key>.npz`` key a tauq's provenance (not its filename) implies, or None."""
    method = tau.provenance.get("method")
    if tau.provenance.get("harness") == "bergson":
        # scored by bergson's pipeline (wikitext2_gpt2_bae's tauq_ekfac_bergson); the
        # pool_tda_ekfac.npz beside it is our EK-FAC's, so attaching it would mix estimators
        return None
    if isinstance(method, str) and method.startswith("trak"):
        # one family keyed by e_features (plain-mean is trak1); trak<N> methods are their key
        if method != "trak":
            return method
        ef = tau.provenance.get("e_features")
        return f"trak{int(ef)}" if ef is not None else None
    if isinstance(method, str) and method.startswith("source"):
        # source_L<L> is its own key; an older bare "source" takes L from the writer's source_args
        if method != "source":
            return method
        seg = (tau.provenance.get("source_args") or {}).get("segments")
        return f"source_L{int(seg)}" if seg is not None else None
    if method in ("ekfac", "tracin"):
        return method
    return None


def _cmd_evaluate(args):
    import sys

    from corrected_lds.banks.artifacts import load_pool, load_pool_tda, load_tauq
    from corrected_lds.evaluate import evaluate_pool, pool_tda_summary
    if os.path.isdir(args.bank):
        bank_dir, pool_path = args.bank, os.path.join(args.bank, "pool.npz")
    else:
        bank_dir, pool_path = os.path.dirname(args.bank), args.bank
    tau_path = (args.tauq if os.path.exists(args.tauq)
                else os.path.join(bank_dir, f"tauq_{args.tauq}.npz"))
    pool = load_pool(pool_path)
    tau = load_tauq(tau_path)
    res = evaluate_pool(pool, tau, soft_threshold_k=args.soft_threshold_k)

    ptda_path = None                        # replica-resolved block (on by default)
    if args.pool_tda == "auto":
        key = _pool_tda_auto_key(tau)
        cand = (os.path.join(bank_dir, f"pool_tda_{key}.npz") if key else None)
        ptda_path = cand if cand and os.path.exists(cand) else None
        if ptda_path is None:         # a missing block must never read as "no block"
            why = (f"{os.path.basename(cand)} is not in {bank_dir}" if cand else
                   f"provenance does not determine one (method="
                   f"{tau.provenance.get('method')!r}, e_features="
                   f"{tau.provenance.get('e_features')!r})")
            print(f"[evaluate] no replica-resolved pool_tda block: {why}. Pass "
                  f"--pool-tda <key|path> to attach one explicitly, or --pool-tda none "
                  f"to ask for no block.", file=sys.stderr)
    elif args.pool_tda != "none":
        ptda_path = (args.pool_tda if os.path.exists(args.pool_tda)
                     else os.path.join(bank_dir, f"pool_tda_{args.pool_tda}.npz"))
        if not os.path.exists(ptda_path):
            raise SystemExit(f"[evaluate] --pool-tda {args.pool_tda!r}: {ptda_path} "
                             f"not found.")
    if ptda_path is not None and args.soft_threshold_k is not None:
        print("[evaluate] pool_tda block skipped: --soft-threshold-k scores a "
              "thresholded tauq, and Eq. 16 is not expressible on pool_tda slices — "
              "the two routes would not be comparable.", file=sys.stderr)
    elif ptda_path is not None:             # pairing/mask-bind mismatch raises here
        res["pool_tda"] = pool_tda_summary(pool, load_pool_tda(ptda_path))
    js = json.dumps(res, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(js + "\n")
        print(f"[evaluate] wrote {args.json}")
    else:
        print(js)
    return 0


def _cmd_compare(args):
    """A rebuilt bank file against the released one, or (``--consistency``) every
    plain-mean tauq of a bank against its pool_tda.  Exit 1 on a mismatch."""
    from corrected_lds.banks import compare as cmp
    if args.consistency:
        rows = cmp.consistency(args.new)
        for r in rows:
            print(f"[consistency] {r['tauq']} vs the first {r['E']} replicas of "
                  f"{r['pool_tda']}: max rel {r['max_rel']:.1e} {'ok' if r['ok'] else 'MISMATCH'}")
        return 0 if all(r["ok"] for r in rows) else 1
    if not args.ref:
        raise SystemExit("[compare] give NEW and REF files, or a bank with --consistency")
    res = cmp.compare_files(args.new, args.ref)
    if res["mismatch"]:
        print(f"[compare] MISMATCH ({res['kind']}): {'; '.join(res['mismatch'])}")
        return 1
    print(f"[compare] {res['kind']} {res['shape']}: bitwise {'yes' if res['bitwise'] else 'no'}, "
          f"max abs {res['max_abs']:.3g}, max rel {res['max_rel']:.3g}, corr {res['corr']:.8f}")
    return 0


def _cmd_fetch_banks(args):
    from corrected_lds.banks.hub import fetch_banks
    return fetch_banks(args.manifest, args.root, names=args.bank,
                       token=os.environ.get("HF_TOKEN"))    # env-only: flags leak into history


def _cmd_push_bank(args):
    from corrected_lds.banks.hub import push_bank
    block = push_bank(args.bank, args.repo, name=args.name,
                      extra_files=tuple(args.file or ()), private=not args.public)
    # print the update for a human to commit (outer braces stripped to paste under "banks")
    entry = "\n".join(json.dumps(block["banks"], indent=1).splitlines()[1:-1])
    print(f'[push] banks-manifest.json: set "url" to {block["url"]} and paste under "banks":')
    print(entry)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="corrected-lds",
                                 description="Debiased, disattenuated LDS evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="build + hash-verify the canonical dataset caches")
    b.add_argument("--dataset", required=True,
                   choices=("cifar10", "mnist", "qnli", "imagenet",
                            "wikitext2"))
    b.add_argument("--smoke", action="store_true",
                   help="imagenet only: one-shard source reconnaissance (auth, schema, "
                        "sanitize timing, size estimates) before committing to the "
                        "~153 GB build; writes nothing")
    b.add_argument("--yes", action="store_true",
                   help="imagenet: start the ~153 GB download and cache build (the "
                        "other datasets are small and never ask)")
    b.set_defaults(func=_cmd_build_data)

    g = sub.add_parser("build-pool", help="build/resume/extend a retrain pool (GPU)")
    g.add_argument("--preset", required=True, choices=sorted(PRESETS))
    g.add_argument("--tag", required=True, help="pool directory name under --root")
    g.add_argument("--M", required=True, type=int, help="number of subset masks")
    g.add_argument("--R", required=True, type=int, help="seeds per mask")
    g.add_argument("--root", default="pools", help="root directory for built pools")
    g.add_argument("--device", default="cuda")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"),
                   help="train only shard I of N into pool_shard_I.npz "
                        "(multi-GPU: one shard per CUDA_VISIBLE_DEVICES)")
    g.add_argument("--merge", type=int, metavar="N",
                   help="merge N completed shards into pool.npz (no training)")
    g.add_argument("--allow-mixed-code", action="store_true",
                   help="with --merge: accept shards written by different code versions "
                        "(the pool records them all in code_versions)")
    g.add_argument("--yes", action="store_true", help="actually start training")
    g.set_defaults(func=_cmd_build_pool)

    t = sub.add_parser("compute-tau",
                       help="train the member ensemble + compute tauq scores (GPU)")
    t.add_argument("--preset", required=True, choices=sorted(PRESETS))
    t.add_argument("--tag", required=True, help="pool directory name under --root")
    t.add_argument("--method", default="trak",
                   choices=("trak", "ekfac", "tracin", "source"))
    t.add_argument("--n-models", type=int, default=None,
                   help="number of independently trained + featurized models entering "
                        "the estimator (the paper's M; a true ensemble of influence "
                        "estimates only under plain-mean aggregation; default: the "
                        "preset's largest ensemble_sizes entry)")
    t.add_argument("--e-features", type=int, default=None,
                   help="TRAK only: override the preset's e_features (the FROZEN inner "
                        "Eq. 15 unit size for pool_tda unit tracking; n_models must "
                        "stay a multiple; see docs/FORMATS.md)")
    t.add_argument("--aggregation", default="eq15",
                   choices=("eq15", "plain-mean", "both"),
                   help="TRAK only: Eq. 15 separated average (default, the paper), the "
                        "EK-FAC-style plain mean of per-member products, or both from "
                        "ONE featurize pass (-> tauq_trak<e_features>_e*, tauq_trak1_e*)")
    t.add_argument("--root", default="pools")
    t.add_argument("--device", default="cuda")
    t.add_argument("--scratch", default=None,
                   help="dir for the large per-model feature store (default: work dir)")
    t.add_argument("--block", default=None,
                   help="TRAK only: assemble the pool_tda slices of ONE contiguous block "
                        "of the roster, e.g. '340-499', from pieces cached by --members "
                        "(GLOBAL member ids: the projector seeds and augmentation a full "
                        "pass would use).  Writes pool_tda_trak<e>_b<A>-<B>.npz (+ trak1 "
                        "with --aggregation both) and NO tauq; append the block onto a "
                        "released file with merge-pool-tda.")
    t.add_argument("--members", default=None,
                   help="TRAK only: train + featurize ONLY these ensemble members, cache "
                        "their pieces under the tag's tau_work dir, and stop, e.g. '0-4' "
                        "or '0,2,5'. Launch one process per GPU on disjoint subsets "
                        "(isolate with CUDA_VISIBLE_DEVICES, not a cuda:N device), then "
                        "re-run with no --members: the accumulator loads the cached "
                        "pieces instead of featurizing — bit-identical to one "
                        "sequential pass.")
    t.add_argument("--runs", default=None,
                   help="SOURCE/TracIn/EK-FAC: train and score ONLY these trajectory "
                        "runs, then stop, "
                        "e.g. '0-4' or '0,2,5'. Run indices are independent, so launch one "
                        "process per GPU on disjoint subsets (isolate with "
                        "CUDA_VISIBLE_DEVICES, not a cuda:N device string), then re-run "
                        "with no --runs to combine and write the tauq.")
    t.add_argument("--keep-partials", action="store_true",
                   help="keep the per-member/per-run work dir after pool_tda is emitted, "
                        "so single-replica tauqs stay recombinable (E=1 from an E=2 run).")
    t.add_argument("--yes", action="store_true", help="actually start")
    t.set_defaults(func=_cmd_compute_tau)

    ib = sub.add_parser("import-bergson-trajectory",
                        help="import one bergson-trained attributor run (CPU/GPU)")
    ib.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ib.add_argument("--tag", required=True, help="bank directory name under --root")
    ib.add_argument("--root", default="pools")
    ib.add_argument("--run", required=True, type=int, help="attributor run index r")
    ib.add_argument("--export-dir", required=True,
                    help="the run's exported/ dir (bergson export_checkpoints output)")
    ib.add_argument("--device", default="cuda",
                    help="where the baseline gate reads the 481 queries (CPU: ~8 min)")
    ib.set_defaults(func=_cmd_import_bergson_trajectory)

    im = sub.add_parser("import-bergson-member",
                        help="import one bergson-trained TRAK member (CPU/GPU)")
    im.add_argument("--preset", required=True, choices=sorted(PRESETS))
    im.add_argument("--tag", required=True, help="bank directory name under --root")
    im.add_argument("--root", default="pools")
    im.add_argument("--member", required=True, type=int, help="member index m")
    im.add_argument("--export-dir", required=True,
                    help="the member run's exported/ dir (bergson export_checkpoints output)")
    im.add_argument("--data-meta", required=True,
                    help="the <member dataset>.json make_member_data.py wrote (rows + hash)")
    im.add_argument("--device", default="cuda")
    im.set_defaults(func=_cmd_import_bergson_member)

    rb = sub.add_parser("read-bergson-retrained",
                        help="eval-mode query losses of bergson's saved retrains (GPU)")
    rb.add_argument("--preset", required=True, choices=sorted(PRESETS))
    rb.add_argument("--dir", required=True, help="validate's <run_path>/retrained")
    rb.add_argument("--lo", required=True, type=int)
    rb.add_argument("--hi", required=True, type=int)
    rb.add_argument("--out", required=True, help="npz with losses (hi-lo, Q)")
    rb.add_argument("--device", default="cuda")
    rb.set_defaults(func=_cmd_read_bergson_retrained)

    bp = sub.add_parser("bergson-pool",
                        help="a bergson bank's pool in the release's layout (GPU pod)")
    bp.add_argument("action", choices=["setup", "check", "assemble"],
                    help="setup: masks, score store, bootstrap.sh and run_replication.sh in "
                         "--dir; check: compare the eval readouts done so far with --ref; "
                         "assemble: pool.npz (eval readout) and pool_dropout_readout.npz")
    bp.add_argument("--dir", required=True, help="work directory on the pod")
    bp.add_argument("--preset", default="wikitext2_gpt2_bae", choices=sorted(PRESETS))
    bp.add_argument("--venv-bergson", default="/root/bv",
                    help="setup: where bootstrap.sh installs bergson")
    bp.add_argument("--ref", default="banks/wikitext2_gpt2_bae/pool.npz",
                    help="check: the released pool")
    bp.add_argument("--out", default="pools/wikitext2_gpt2_bae",
                    help="assemble: the bank directory to write")
    bp.set_defaults(func=_cmd_bergson_pool)

    be = sub.add_parser("bergson-ekfac",
                        help="bergson's own EK-FAC on one full-data model (GPU pod)")
    be.add_argument("action", choices=["setup", "import"],
                    help="setup: bootstrap.sh, train.yaml, ekfac.yaml and run_ekfac.sh in "
                         "--dir; import: bergson's score store under --dir -> --out")
    be.add_argument("--dir", required=True, help="work directory on the pod (~330 GB free)")
    be.add_argument("--preset", default="wikitext2_gpt2_bae", choices=sorted(PRESETS))
    be.add_argument("--venv-bergson", default="/root/bv",
                    help="setup: where bootstrap.sh installs bergson")
    be.add_argument("--out", default="pools/wikitext2_gpt2_bae/tauq_ekfac_bergson.npz",
                    help="the tauq that import writes")
    be.set_defaults(func=_cmd_bergson_ekfac)

    sc = sub.add_parser("bergson-source-config",
                        help="write a pod's multi-seed SOURCE driver config + bootstrap")
    sc.add_argument("--preset", default="wikitext2_gpt2_bae", choices=sorted(PRESETS))
    sc.add_argument("--runs", default=None,
                    help="attributor run indices, e.g. 1-9 or 1,2,3 (omit for a check-only pod)")
    sc.add_argument("--gpus", required=True, help="GPU ids, e.g. 0-7 (one lane each)")
    sc.add_argument("--shm-lanes", type=int, default=3,
                    help="the first N lanes scratch on --shm-dir (RAM), the rest on --disk-dir")
    sc.add_argument("--shm-dir", default="/dev/shm/source_scratch")
    sc.add_argument("--disk-dir", default="/root/source_scratch")
    sc.add_argument("--root", default="/root/source_e10", help="work root on local disk")
    sc.add_argument("--pool", required=True, help="eval-readout pool.npz (gates, identity)")
    sc.add_argument("--baselines-dir", required=True,
                    help="bank dir whose pool carries baselines (the train gate)")
    sc.add_argument("--check-run", type=int, default=None,
                    help="re-score this run's first piece and compare with --check-ref")
    sc.add_argument("--check-ref", default=None, help="that piece's earlier score store")
    sc.add_argument("--query-batch", type=int, default=64,
                    help="queries per piece (bergson's own batch is 64; scratch scales with it)")
    sc.add_argument("--piece-gb", type=float, default=None,
                    help="scratch one fresh piece may grow to (default: from --query-batch)")
    sc.add_argument("--venv-bergson", default="/root/bv")
    sc.add_argument("--venv-ours", default="/root/venv")
    sc.add_argument("--site-fix", default="/root/bergson_fix_site")
    sc.add_argument("--data-dir", default="/workspace/cldata")
    sc.add_argument("--out", required=True, help="directory for the config + bootstrap")
    sc.set_defaults(func=_cmd_bergson_source_config)

    sr = sub.add_parser("bergson-source-run", help="run the multi-seed SOURCE driver (GPU pod)")
    sr.add_argument("--config", required=True)
    sr.set_defaults(func=_cmd_bergson_source_run)

    sk = sub.add_parser("bergson-source-check",
                        help="baseline-loss gate for one SOURCE attributor run (GPU)")
    sk.add_argument("--preset", required=True, choices=sorted(PRESETS))
    sk.add_argument("--export-dir", required=True)
    sk.add_argument("--baselines-dir", required=True)
    sk.add_argument("--expect-seed", type=int, default=None)
    sk.add_argument("--device", default="cuda")
    sk.set_defaults(func=_cmd_bergson_source_check)

    sm = sub.add_parser("bergson-source-combine",
                        help="single-run SOURCE tauqs -> E-ensemble tauq + pool_tda (CPU)")
    sm.add_argument("parts", nargs="+", help="tauq npz files, one per attributor run")
    sm.add_argument("--pool", required=True, help="the pool.npz they pair with")
    sm.add_argument("--out-dir", required=True)
    sm.set_defaults(func=_cmd_bergson_source_combine)

    mp = sub.add_parser("merge-pool-tda",
                        help="concatenate pool_tda parts along the replica axis (CPU)")
    mp.add_argument("parts", nargs="+", help="pool_tda_*.npz parts, IN REPLICA ORDER")
    mp.add_argument("--pool", required=True, help="the pool.npz the slices bind to")
    mp.add_argument("--out", required=True, help="output pool_tda path")
    mp.add_argument("--force", action="store_true", help="overwrite an existing --out")
    mp.set_defaults(func=_cmd_merge_pool_tda)

    e = sub.add_parser("evaluate", help="score one bank: pool x tauq -> result JSON (CPU)")
    e.add_argument("bank", help="bank directory (containing pool.npz) or a pool.npz path")
    e.add_argument("--tauq", required=True,
                   help="tauq name (e.g. trak10_e10 -> <bank>/tauq_trak10_e10.npz) or a path")
    e.add_argument("--pool-tda", default="auto",
                   help="attach the replica-resolved block to the result: 'auto' "
                        "(default — use the pool_tda_<method>.npz implied by the "
                        "tauq's provenance when it exists next to the pool), 'none' "
                        "to disable, or an explicit pool_tda name/path")
    e.add_argument("--soft-threshold-k", type=int, default=None,
                   help="apply Eq. 16 fixed-k soft-thresholding to the tauq before "
                        "scoring (the paper's ImageNet protocol: k=1000; the tauq's "
                        "provenance records the recommended k)")
    e.add_argument("--json", default="", help="write the result JSON here instead of stdout")
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("compare", help="compare a rebuilt bank file with the released one, "
                                       "or check a bank's tauqs against its pool_tdas (CPU)")
    c.add_argument("new", help="the rebuilt pool.npz, tauq_*.npz or pool_tda_*.npz; with "
                               "--consistency, a bank directory")
    c.add_argument("ref", nargs="?", help="the released file of the same kind")
    c.add_argument("--consistency", action="store_true",
                   help="check that S @ tauq_<m>_e<E> is the mean of the first E replicas "
                        "of pool_tda_<m>, for every plain-mean tauq of the bank")
    c.set_defaults(func=_cmd_compare)

    f = sub.add_parser("fetch-banks", help="download the released banks, verified against "
                                           "the manifest committed in this repo")
    f.add_argument("--bank", action="append", help="bank name (repeatable; default: all)")
    f.add_argument("--manifest", default="banks-manifest.json")
    f.add_argument("--root", default="banks")
    f.set_defaults(func=_cmd_fetch_banks)

    u = sub.add_parser("push-bank", help="validate a bank directory and upload it to a "
                                         "HF dataset repo in ONE commit; prints the "
                                         "manifest entry to paste (needs the [hub] "
                                         "extra and hf auth login / HF_TOKEN)")
    u.add_argument("bank", help="bank directory (pool.npz + tauq_*.npz)")
    u.add_argument("--repo", required=True,
                   help="HF dataset repo id, e.g. your-org/corrected-lds-banks")
    u.add_argument("--name", default=None,
                   help="bank name inside the repo (default: the directory's name)")
    u.add_argument("--public", action="store_true",
                   help="create the repo public (default: private; applies at CREATION "
                        "only — an existing repo's visibility is never changed)")
    u.add_argument("--file", action="append", metavar="EXTRA",
                   help="extra file to upload verbatim, e.g. shards/ckpts (repeatable; "
                        "unvalidated, excluded from the manifest entry)")
    u.set_defaults(func=_cmd_push_bank)

    args = ap.parse_args(argv)
    return args.func(args)
