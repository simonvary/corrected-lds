"""Build a pool on EleutherAI's bergson retrain harness, run as a subprocess (a reimplemented
trainer would confound the protocol being measured).

bergson removes data by zeroing a document's loss weight and shares one seed across a
replica's masks (unlike ``runner.cell_seed``), so batching and data order match across masks
and ``y[r, j, q] = mu(S_j) + c(r) + eps(r, j)``, where ``c(r)`` cancels in a correlation across
masks: the floor is :func:`~corrected_lds.estimator.seed_noise_variance_centred` (``_stacked``
over-reports it), and pools written here are stamped ``seed_mode="shared-per-replica"`` to pick it.

We store the raw per-query cross-entropy, negated (higher = better): bergson's ``diff`` derives
from it, and chunking is safe as each chunk's own baseline cancels in ``baseline_loss - diff``.

Install bergson from :data:`BERGSON_COMMIT` with ``--force-reinstall``: PyPI 0.26.1 calls
``per_doc_query_losses`` without ``grad_accum_steps`` (one forward, 24.5 GiB of logits at
batch 256: OOM), and git HEAD still reports 0.26.1, so ``pip install git+...`` silently no-ops.
"""

import json
import os

import numpy as np

from corrected_lds.banks.masks import sample_subsets

__all__ = [
    "BERGSON_COMMIT",
    "BERGSON_TRANSFORMERS",
    "EKFAC_PROVENANCE",
    "build_eval_pool",
    "build_pool",
    "check_member_data",
    "check_trajectory_stamp",
    "collect",
    "import_trajectory",
    "member_final_step",
    "member_rows",
    "member_stamp",
    "read_retrained",
    "read_score_store",
    "render_bootstrap",
    "render_member_data_script",
    "render_member_script",
    "render_member_yaml",
    "render_replication_worker",
    "render_trajectory_script",
    "render_trajectory_yaml",
    "replication_units",
    "run_seed",
    "setup_replication_inputs",
    "trajectory_stamp",
    "trajectory_steps",
    "write_ekfac_scripts",
    "write_member_scripts",
    "write_replication_scripts",
    "write_score_store",
    "write_subsets_json",
    "write_trajectory_scripts",
]

# The pinned bergson commit; the bergson files cited in this module refer to it.
BERGSON_COMMIT = "0ad2e5b0ce09207d2af0e80383fe2d952dd8e747"
# The bergson venv's transformers: it reproduced the released wikitext pool; bergson pins none.
BERGSON_TRANSFORMERS = "5.18.0"


def write_subsets_json(masks, path):
    """Our ``(M, n)`` int8 masks (1 = kept) -> bergson's ``subsets.json``, the doc ids to remove
    (weight ``subset_weight=0.0``); :func:`read_subsets_json` inverts it exactly."""
    masks = np.asarray(masks)
    removed = [np.nonzero(masks[j] == 0)[0].tolist() for j in range(masks.shape[0])]
    with open(path, "w") as f:
        json.dump(removed, f)
    return removed


def read_subsets_json(path, n):
    """``subsets.json`` -> ``(M, n)`` int8 masks with 1 = kept."""
    with open(path) as f:
        removed = json.load(f)
    masks = np.ones((len(removed), int(n)), dtype=np.int8)
    for j, ids in enumerate(removed):
        masks[j, ids] = 0
    return masks


def write_score_store(path, n_train, n_queries, scores=None):
    """Write a bergson score store (``info.json`` + ``scores.bin``), zeros if ``scores`` is None:
    ``bergson validate`` requires one (``cli/commands.py``: ``assert self.scores``), and on a
    pool-only build its ``nan`` Spearman is ignored while the query losses are unaffected.  The
    layout is bergson's ``_score_struct_dtype`` for float32, a ``(score_i, written_i)`` pair per
    8 bytes; bergson reads it back from ``info.json``."""
    n_train, n_queries = int(n_train), int(n_queries)
    os.makedirs(path, exist_ok=True)
    struct_json = {"names": [f"{k}_{i}" for i in range(n_queries) for k in ("score", "written")],
                   "formats": ["float32", "bool"] * n_queries,
                   "offsets": [8 * i + o for i in range(n_queries) for o in (0, 4)],
                   "itemsize": 8 * n_queries}
    mm = np.memmap(os.path.join(path, "scores.bin"), dtype=np.dtype(struct_json),
                   mode="w+", shape=(n_train,))
    for i in range(n_queries):
        mm[f"score_{i}"] = 0.0 if scores is None else scores[:, i].astype("float32")
        mm[f"written_{i}"] = True
    mm.flush()
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump({"attribute_tokens": False, "num_items": n_train,
                   "num_rows": n_train, "num_scores": n_queries,
                   "dtype": struct_json}, f, indent=2)


def read_score_store(path):
    """A bergson score store -> ``(n_train, n_queries)`` float64 as stored, refusing unwritten
    columns.  Signs are not normalised: EK-FAC/TRAK scores are influence-signed (higher =
    proponent, our tauq convention), SOURCE/MAGIC loss-signed; the caller states which."""
    with open(os.path.join(path, "info.json")) as f:
        info = json.load(f)
    dt = np.dtype({k: info["dtype"][k] for k in ("names", "formats", "offsets", "itemsize")})
    raw = np.memmap(os.path.join(path, "scores.bin"), dtype=dt, mode="r")
    n_q = int(info["num_scores"])
    unwritten = [q for q in range(n_q) if not raw[f"written_{q}"].all()]
    if unwritten:
        raise ValueError(f"[bergson] {path}: {len(unwritten)} of {n_q} score columns are "
                         f"not fully written (first: {unwritten[0]})")
    return np.stack([raw[f"score_{q}"].astype(np.float64) for q in range(n_q)], axis=1)


def _read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def _refuse_overlap(run_dir, name, owner, subsets):
    """Record that chunk file ``name`` holds ``subsets``; raise if another file of ``run_dir``
    already did, as overlapping chunk layouts would let the last-sorted file silently win."""
    for j in subsets:
        if owner[j] is not None:
            raise ValueError(f"[bergson] {run_dir}: subset {j} is in both {owner[j]} and "
                             f"{name}; overlapping chunk files -- remove the stale one.")
        owner[j] = name


def collect(run_dir, M, Q):
    """One replica's ``validation*.csv`` + ``summary*.csv`` -> ``(loss, baseline, seen)``:
    ``(M, Q)`` raw per-query cross-entropy ``baseline_loss - diff``, the ``(Q,)`` baseline of
    the last chunk read, and an ``(M,)`` bool of masks present (a gap is never zero-filled).
    Each chunk (``validation_{lo}_{hi}.csv``) has its own baseline, which cancels."""
    M, Q = int(M), int(Q)
    loss = np.full((M, Q), np.nan)
    seen = np.zeros(M, dtype=bool)
    baseline = np.full(Q, np.nan)
    names = sorted(n for n in os.listdir(run_dir)
                   if n.startswith("validation") and n.endswith(".csv"))
    if not names:
        raise FileNotFoundError(f"[bergson] no validation*.csv under {run_dir}")
    owner = [None] * M
    for name in names:
        suffix = name[len("validation"):-len(".csv")]
        spath = os.path.join(run_dir, f"summary{suffix}.csv")
        if not os.path.exists(spath):
            continue                              # chunk still running: no summary yet
        v, s = _read_csv(os.path.join(run_dir, name)), _read_csv(spath)
        base = np.full(Q, np.nan)
        base[s["query"].astype(int)] = s["baseline_loss"]
        sub = v["subset"].astype(int)
        _refuse_overlap(run_dir, name, owner, np.unique(sub).tolist())
        qs = v["query"].astype(int)
        loss[sub, qs] = base[qs] - v["diff"]
        seen[np.unique(sub)] = True
        baseline = base
    return loss, baseline, seen


def build_pool(out_path, run_dirs, seeds, masks, *, alpha, recipe, Q,
               dataset="wikitext2", code_version="bergson-pool-1"):
    """Assemble collected replicas into a ``pool.npz`` that ``load_pool`` accepts:
    ``margins = -loss``, plus the per-replica ``baselines`` and raw ``losses`` as extra keys.
    Side digests come from :func:`~corrected_lds.banks.presets.bank_hashes`, so a drifted dataset
    revision fails ``load_pool``'s pin check; stamped ``seed_mode="shared-per-replica"``."""
    M = np.asarray(masks).shape[0]
    losses, baselines = [], []
    for rd in run_dirs:
        loss, base, seen = collect(rd, M, Q)
        if not seen.all():
            raise ValueError(f"[bergson] {rd} is INCOMPLETE: {int((~seen).sum())} of {M} "
                             f"masks missing; finish the run before building a pool.")
        losses.append(loss)
        baselines.append(base)
    # bergson's in-loop readout never calls .eval(), so under train_mode dropout is on
    readout = "dropout" if dict(recipe).get("train_mode") else "in-loop"
    return _save_pool(out_path, np.stack(losses), masks, seeds, alpha=alpha, recipe=recipe,
                      Q=Q, dataset=dataset, code_version=code_version,
                      baselines=np.stack(baselines), readout=readout)


def _save_pool(out_path, y, masks, seeds, *, alpha, recipe, Q, dataset, code_version,
               **extra):
    """Write raw losses ``y (R, M, Q)`` as a shared-per-replica ``pool.npz``; ``extra`` keys
    ride along (``load_pool`` checks presence, not exclusivity)."""
    from corrected_lds.banks.presets import bank_hashes

    masks = np.asarray(masks, dtype=np.int8)
    M, n = masks.shape
    recipe = dict(recipe)
    if float(recipe.get("alpha", alpha)) != float(alpha):
        raise ValueError(f"[bergson] alpha {alpha} != recipe alpha {recipe.get('alpha')}; "
                         f"load_pool refuses an internally inconsistent pool.")
    recipe["alpha"] = float(alpha)
    y = np.asarray(y, dtype=np.float64)
    if y.shape[1:] != (M, int(Q)):
        raise ValueError(f"[bergson] losses {y.shape} do not match (R, {M}, {Q})")
    train_hash, query_hash = bank_hashes(dataset)
    np.savez(
        out_path,
        margins=-y, losses=y,
        subsets=masks, M=M, R=y.shape[0], alpha=float(alpha),
        recipe=json.dumps(recipe), dataset=dataset,
        seed_mode="shared-per-replica", subsets_source="sample_subsets:777",
        schema=3, code_version=code_version,
        completed_cells=y.shape[0] * M, done_rows=y.shape[0],
        seeds=np.asarray(seeds, dtype=np.int64),
        train_idx=np.arange(n), query_idx=np.arange(Q),
        train_hash=train_hash, query_hash=query_hash, **extra,
    )
    return out_path


# -- pod-side scripts, rendered here so this module is their source of truth --- #
def render_bootstrap(root="/workspace/p1", venv="/workspace/bv"):
    """One-shot bootstrap for a fresh pod: verify the GPU, install bergson pinned."""
    return f"""#!/usr/bin/env bash
# Bootstrap a fresh pod for the bergson wikitext pool.  Idempotent.
set -euo pipefail

# 1. The GPU must actually run this model.  A 5090 is Blackwell (sm_120) and needs a
#    torch built for CUDA 12.8+; an older docker torch fails with a cryptic kernel
#    error rather than a clean message, so check FIRST.  Kernels for a lower minor version
#    of the same major run too (sm_86 on an sm_89 L40S or RTX 4090).
python -c "
import torch
cap = torch.cuda.get_device_capability(0)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', cap)
assert [a for a in torch.cuda.get_arch_list() if a[3:].isdigit() and int(a[3:]) // 10 == cap[0]
        and int(a[3:]) % 10 <= cap[1]], 'torch has no kernels for this GPU'
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(10): b = a @ a
torch.cuda.synchronize(); print('matmul OK')
"

# 2. venv INHERITS the docker CUDA wheels; only pure-python deps are installed.
python -m venv --system-site-packages {venv}
{venv}/bin/pip install -q --upgrade pip
{venv}/bin/pip install -q bergson "transformers=={BERGSON_TRANSFORMERS}"

# 3. Pin to the analysed commit.  MANDATORY: git HEAD still self-reports 0.26.1, so a
#    plain install no-ops over the PyPI build, whose per_doc_query_losses does not chunk
#    the query forward (24.5 GiB alloc -> OOM).
{venv}/bin/pip install -q --force-reinstall --no-deps \\
  "git+https://github.com/EleutherAI/bergson.git@{BERGSON_COMMIT}"
test "$({venv}/bin/python -c "import bergson,os;print(os.path.dirname(bergson.__file__))")" != ""
grep -c "run_cfg.grad_accum_steps" {venv}/lib/python*/site-packages/bergson/validate.py

mkdir -p {root}
echo "bootstrap OK"
"""


# -- attributor trajectories: full-set bergson runs for our TracIn and EK-FAC --- #
def trajectory_steps(recipe, n_train, units_per_epoch) -> list:
    """The global steps whose checkpoints are kept, every ``1/k`` epoch: on the Bae recipe
    (``4656 / 8 = 582`` steps per epoch, ``k = 2``) Bae et al.'s six, ``[291, ..., 1746]``.
    Refuses an inexact split, on which bergson's checkpoint steps and our units would disagree."""
    k = int(units_per_epoch)
    spe, rem = divmod(int(n_train), int(recipe.batch))
    if rem or spe % k:
        raise ValueError(f"[bergson] {n_train} docs / batch {recipe.batch} do not split "
                         f"into {k} equal units per epoch")
    return [(spe // k) * u for u in range(1, int(recipe.epochs) * k + 1)]


def trajectory_stamp(recipe, n_train, units_per_epoch, seed_base,
                     dataset="wikitext2") -> str:
    """The ``tracin_ckpts/_provenance.json`` of an imported trajectory bank: the
    ``tracin.train_trajectories`` stamp (whose recipe pins the bergson commit and dataset
    revision) plus ``harness`` and ``units_per_epoch``, so our and bergson's banks never mix."""
    return json.dumps({"protocol": "full-set-trajectory", "harness": "bergson",
                       "seed_base": int(seed_base), "recipe": recipe.to_dict(),
                       "dataset": dataset, "n_train": int(n_train),
                       "units_per_epoch": int(units_per_epoch)}, sort_keys=True)


def check_trajectory_stamp(traj_dir, stamp, *, write=False,
                           importer="import-bergson-trajectory"):
    """Refuse a ``traj_dir`` stamped for anything else or holding unstamped checkpoints;
    ``write=True`` stamps a fresh directory (the importer), ``False`` requires the stamp (the
    scorers).  Also guards the TRAK members' ``ckpts/`` (``importer`` names its command)."""
    path = os.path.join(traj_dir, "_provenance.json")
    if os.path.exists(path):
        with open(path) as fh:
            stored = fh.read()
        if stored != stamp:
            raise RuntimeError(f"[bergson] {traj_dir} holds trajectories from a DIFFERENT "
                               f"protocol/recipe; refusing to mix.\n  stored : {stored}\n"
                               f"  current: {stamp}")
        return
    if not write:
        raise FileNotFoundError(f"[bergson] {traj_dir} has no imported checkpoints; train "
                                f"them with bergson and run `corrected-lds "
                                f"{importer}` first (see docs/REPRODUCE.md).")
    os.makedirs(traj_dir, exist_ok=True)
    loose = [f for _, _, fs in os.walk(traj_dir) for f in fs if f.endswith(".pt")]
    if loose:
        raise RuntimeError(f"[bergson] {traj_dir} holds {len(loose)} UNSTAMPED "
                           f"checkpoint(s); refusing to reuse them.")
    with open(path, "w") as fh:
        fh.write(stamp)


def import_trajectory(export_dir, out_paths, steps, model):
    """Convert bergson's exported ``checkpoint-<step>/`` dirs into our checkpoints; returns
    ``out_paths``.  Each state dict is loaded into ``model`` (a ``wikitext_gpt2.make_model``)
    with ``strict=True`` first, so a changed layout fails here, not inside a score."""
    from safetensors.torch import load_file

    from corrected_lds.pools.tasks.wikitext_gpt2 import hf_to_ours
    from corrected_lds.tda.trak import _save_ckpt_atomic

    if len(out_paths) != len(steps):
        raise ValueError(f"[bergson] {len(steps)} steps for {len(out_paths)} paths")
    for step, out in zip(steps, out_paths):
        src = os.path.join(export_dir, f"checkpoint-{int(step)}", "model.safetensors")
        if not os.path.exists(src):
            raise FileNotFoundError(f"[bergson] {src} missing; export with "
                                    f"bergson.utils.trainer_export.export_checkpoints"
                                    f"(run_path, steps={list(steps)})")
        model.load_state_dict(hf_to_ours(load_file(src)), strict=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        _save_ckpt_atomic(model.state_dict(), out)
    return list(out_paths)


def render_trajectory_yaml(r, *, root, recipe, seed_base, n_train, units_per_epoch,
                           repo, revision) -> str:
    """bergson ``train`` config for attributor run ``r``, seeded ``seed_base + r``: the bank's
    recipe field by field (so trajectories and retrains cannot drift apart), shaped as bergson's
    ``wikitext_gpt2_train.yaml`` but without Adam state, which EK-FAC and TracIn never read."""
    steps = trajectory_steps(recipe, n_train, units_per_epoch)
    return f"""# attributor trajectory run {r} -- rendered by corrected_lds.pools.bergson
steps:
  - train:
      run_path: {root}/run_{int(r):02d}
      model: {recipe.model}
      overwrite: true
      precision: {recipe.precision}
      seed: {int(seed_base) + int(r)}

      data:
        dataset: {repo}
        split: train
        chunk_length: 0
        data_kwargs: revision={revision}

{_recipe_train_fields(recipe)}
      save_mode: interval
      save_interval: {steps[0]}
      save_optimizer_state: none
"""


def _recipe_train_fields(recipe) -> str:
    """The recipe-fixed YAML lines of a single-process bergson ``train`` step, shared by the
    trajectories and the TRAK members so neither drifts from the retrains the recipe describes."""
    if recipe.lr_schedule != "constant":
        raise ValueError(f"[bergson] only a constant schedule is rendered, got "
                         f"{recipe.lr_schedule!r}")
    return f"""      distributed:
        nproc_per_node: 1
        nnode: 1

      optimizer: {recipe.optimizer}
      adam_beta1: {recipe.adam_beta1}
      adam_beta2: {recipe.adam_beta2}
      adam_eps: {recipe.adam_eps}
      eps_root: {recipe.eps_root}
      weight_decay: {recipe.weight_decay}
      batch_size: {recipe.batch}
      num_epochs: {recipe.epochs}
      lr_schedule:
        lr_scheduler_type: constant
        lr: {recipe.peak_lr}
      train_mode: {str(bool(recipe.train_mode)).lower()}
"""


def render_trajectory_script(n_runs, *, root, venv_bergson, venv_ours, preset, tag,
                             pools_root, steps) -> str:
    """Per-worker driver: train, export and import attributor runs ``r % NW == W``.  Each run is
    skip-if-imported, so re-running the command resumes; bergson's raw and exported
    checkpoints are deleted once a run is imported (~1.5 GB per kept step otherwise)."""
    steps_py = "[" + ", ".join(str(int(s)) for s in steps) + "]"
    return f"""#!/usr/bin/env bash
# bergson attributor trajectories -- worker $1 of $2 (GPU $3, default $1).  Re-run to resume.
set -u
W=$1; NW=$2
export CUDA_VISIBLE_DEVICES=${{3:-$W}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((29700 + W))
ROOT={root}
FAILED=0
for (( r=0; r<{int(n_runs)}; r++ )); do
  (( r % NW == W )) || continue
  RUN=$(printf "run_%02d" $r)
  if [[ -f $ROOT/imported/$RUN.done ]]; then echo "[w$W] SKIP $RUN (imported)"; continue; fi
  echo "[w$W] TRAIN $RUN $(date -Is)"
  {venv_bergson}/bin/bergson $ROOT/yaml/$RUN.yaml >> $ROOT/logs/$RUN.log 2>&1 \\
    && {venv_bergson}/bin/python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('$ROOT/$RUN', steps={steps_py}, overwrite=True)" >> $ROOT/logs/$RUN.log 2>&1 \\
    && {venv_ours}/bin/corrected-lds import-bergson-trajectory --preset {preset} --tag {tag} \\
         --root {pools_root} --run $r --export-dir $ROOT/$RUN/exported >> $ROOT/logs/$RUN.log 2>&1
  rc=$?
  if (( rc == 0 )); then
    mkdir -p $ROOT/imported && touch $ROOT/imported/$RUN.done
    rm -rf $ROOT/$RUN/checkpoints $ROOT/$RUN/exported
    echo "[w$W] OK $RUN $(date -Is)"
  else
    echo "[w$W] FAIL $RUN rc=$rc $(date -Is)"; FAILED=1
  fi
done
echo "=== TRAJECTORY WORKER $W DONE failed=$FAILED $(date -Is) ==="
exit $FAILED
"""


def write_trajectory_scripts(out_dir, n_runs, *, recipe, seed_base, n_train,
                             units_per_epoch, repo, revision, venv_bergson, venv_ours,
                             preset, tag, pools_root):
    """Write ``yaml/``, ``run_trajectories.sh`` and ``logs/`` under ``out_dir``, the run root."""
    for sub in ("yaml", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for r in range(int(n_runs)):
        with open(os.path.join(out_dir, "yaml", f"run_{r:02d}.yaml"), "w") as f:
            f.write(render_trajectory_yaml(r, root=out_dir, recipe=recipe,
                                           seed_base=seed_base, n_train=n_train,
                                           units_per_epoch=units_per_epoch, repo=repo,
                                           revision=revision))
    sh = os.path.join(out_dir, "run_trajectories.sh")
    with open(sh, "w") as f:
        f.write(render_trajectory_script(
            n_runs, root=out_dir, venv_bergson=venv_bergson, venv_ours=venv_ours,
            preset=preset, tag=tag, pools_root=pools_root,
            steps=trajectory_steps(recipe, n_train, units_per_epoch)))
    os.chmod(sh, 0o755)
    return sh


# -- bergson's own EK-FAC on one full-data model: tauq_ekfac_bergson.npz ------ #
# The released file's provenance; bergson's EK-FAC scores are influence-signed, as ours are.
EKFAC_PROVENANCE = {"method": "ekfac", "harness": "bergson", "hessian": "kfac+ev_correction",
                    "damping": 1e-8, "filter_modules": "lm_head", "projection": "none",
                    "nproc_per_node": 1, "bergson_commit": BERGSON_COMMIT}


def write_ekfac_scripts(out_dir, *, recipe, seed, n_train, units_per_epoch, repo, revision,
                        venv_bergson, venv_ours, preset, out):
    """``train.yaml`` (the full-data model, seeded ``seed``), ``ekfac.yaml`` (bergson's Bae
    example at our pin: K-FAC with eigenvalue correction, damping 1e-8, ``lm_head`` excluded,
    unprojected, one process, every validation query) and ``run_ekfac.sh``, which trains,
    exports the final checkpoint, scores, and imports the score store as ``out``."""
    last = trajectory_steps(recipe, n_train, units_per_epoch)[-1]
    train = render_trajectory_yaml(0, root=out_dir, recipe=recipe, seed_base=seed,
                                   n_train=n_train, units_per_epoch=units_per_epoch,
                                   repo=repo, revision=revision).replace(
        "# attributor trajectory run 0", f"# bergson's EK-FAC model, seed {int(seed)}")
    ekfac = f"""# bergson's EK-FAC on the full-data model -- rendered by corrected_lds.pools.bergson
steps:
  - ekfac:
      index_cfg:
        run_path: {out_dir}/ekfac
        model: {out_dir}/run_00/exported/checkpoint-{last}
        overwrite: true
        precision: {recipe.precision}
        token_batch_size: 1024
        max_batch_size: 64
        filter_modules: "lm_head"
        distributed:
          nproc_per_node: 1
          nnode: 1
        data:
          dataset: {repo}
          split: train
          chunk_length: 0
          data_kwargs: revision={revision}
      hessian_cfg:
        method: kfac
        hessian_dtype: fp32
        ev_correction: true
      score_cfg:
        batch_size: 1024
        query_batch_size: 64
      preprocess_cfg:
        unit_normalize: false
      hessian_pipeline_cfg:
        query:
          dataset: {repo}
          split: validation
          chunk_length: 0
          data_kwargs: revision={revision}
        query_aggregation: none
        inversion_cfg:
          damping_factor: 1.0e-8
"""
    sh = f"""#!/usr/bin/env bash
# bergson's EK-FAC reference -- rendered by corrected_lds.pools.bergson.  One GPU; the
# unprojected query gradients need about 330 GB free under {out_dir}/ekfac.
set -euo pipefail
{venv_bergson}/bin/bergson {out_dir}/train.yaml
{venv_bergson}/bin/python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('{out_dir}/run_00', steps=[{last}], overwrite=True)"
{venv_bergson}/bin/bergson {out_dir}/ekfac.yaml
{venv_ours}/bin/corrected-lds bergson-ekfac import --preset {preset} --dir {out_dir} --out {out}
"""
    os.makedirs(out_dir, exist_ok=True)
    for name, text in (("train.yaml", train), ("ekfac.yaml", ekfac), ("run_ekfac.sh", sh)):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
    os.chmod(os.path.join(out_dir, "run_ekfac.sh"), 0o755)
    return os.path.join(out_dir, "run_ekfac.sh")


def run_seed(run_dir):
    """The training seed in ``<run_dir>/config.yaml``, or None if absent: bergson writes
    ``{steps: [{command: payload}], ...}`` there, and the seed is the first ``payload["seed"]``."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return None
    import yaml

    with open(path) as fh:
        loaded = yaml.safe_load(fh)
    steps = loaded.get("steps", loaded) if isinstance(loaded, dict) else loaded
    payload = next(iter(steps[0].values())) if isinstance(steps, list) else None
    if not isinstance(payload, dict) or "seed" not in payload:
        raise ValueError(f"[bergson] {path} is not a bergson run config with a seed")
    return int(payload["seed"])


# -- TRAK members, each trained by bergson on its own half, imported into ckpts/ --- #
def member_stamp(recipe, n_train, seed_base, dataset="wikitext2") -> str:
    """The ``ckpts/_provenance.json`` of imported TRAK members: ``trak.train_ensemble``'s stamp
    plus ``harness``, so our members and bergson's are never read as each other."""
    return json.dumps({"protocol": "subset-trained", "harness": "bergson",
                       "seed_base": int(seed_base), "recipe": recipe.to_dict(),
                       "dataset": dataset, "n_train": int(n_train)}, sort_keys=True)


def member_rows(m, n_train, alpha) -> list:
    """Member ``m``'s training rows: ``trak.member_subset``'s draw, sorted because bergson trains
    on a dataset in row order and shuffles it per epoch by the seed."""
    from corrected_lds.tda.trak import member_subset
    return sorted(int(i) for i in member_subset(m, n_train, alpha))


def member_final_step(recipe, n_train) -> int:
    """A member's last optimizer step, ``epochs * n_sub / batch``: a member trains on its half as
    a dataset of its own (as ``train_ensemble`` does), not as the pool's weight-zeroed retrains.
    Refuses a ragged last batch: the exported step would then depend on bergson's batching."""
    n_sub = max(1, round(recipe.alpha * int(n_train)))
    spe, rem = divmod(n_sub, int(recipe.batch))
    if rem:
        raise ValueError(f"[bergson] a member's {n_sub} rows do not split into batches "
                         f"of {recipe.batch}")
    return spe * int(recipe.epochs)


def check_member_data(meta, m, n_train, alpha, train_ids) -> None:
    """Check bergson trained member ``m`` on our rows: ``meta``'s member index, row list
    (:func:`member_rows`) and sha256/16 of the rows' int64 ``input_ids`` must match ours (from
    ``train_ids``, the verified ``(n_train, T)`` cache), so no other revision or order passes."""
    from corrected_lds.pools.datasets import _sha16
    rows = member_rows(m, n_train, alpha)
    bad = []
    if int(meta.get("member", -1)) != int(m):
        bad.append(f"member {meta.get('member')!r} != {m}")
    if list(meta.get("rows", [])) != rows:
        bad.append("row list differs from member_rows")
    else:
        ours = _sha16(np.asarray(train_ids, dtype=np.int64)[rows])
        if meta.get("sha16") != ours:
            bad.append(f"input_ids sha16 {meta.get('sha16')} != ours {ours}")
    if bad:
        raise ValueError(f"[bergson] member {m}'s training data is not our member: "
                         + "; ".join(bad))


def render_member_yaml(m, *, root, recipe, seed_base) -> str:
    """bergson ``train`` config for TRAK member ``m``: the bank's recipe on the member's half,
    saved at ``<root>/data/member_MMM``, seeded ``seed_base + m``; only the final state is kept."""
    return f"""# TRAK member {m} -- rendered by corrected_lds.pools.bergson
steps:
  - train:
      run_path: {root}/runs/member_{int(m):03d}
      model: {recipe.model}
      overwrite: true
      precision: {recipe.precision}
      seed: {int(seed_base) + int(m)}

      data:
        dataset: {root}/data/member_{int(m):03d}
        split: train
        chunk_length: 0

{_recipe_train_fields(recipe)}
      save_mode: final
      save_optimizer_state: none
"""


def render_member_data_script(repo, revision) -> str:
    """Python for the bergson venv, ``<spec.json> <out_dir>``: ``save_to_disk`` the member's rows
    of the pinned train split at ``out_dir`` (a layout bergson's loader reads) and write
    ``<out_dir>.json``, the spec plus the rows' sha256/16 for :func:`check_member_data`."""
    return f'''# TRAK member dataset writer -- rendered by corrected_lds.pools.bergson
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from datasets import load_dataset

spec_path, out = sys.argv[1], sys.argv[2]
with open(spec_path) as fh:
    spec = json.load(fh)
ds = load_dataset("{repo}", split="train", revision="{revision}")
sub = ds.select(spec["rows"])
ids = np.ascontiguousarray(np.asarray(sub["input_ids"], dtype=np.int64))
tmp = out + ".tmp"
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(out, ignore_errors=True)     # a crash between save and meta leaves one
sub.save_to_disk(tmp)
os.replace(tmp, out)
meta = dict(spec, sha16=hashlib.sha256(ids.tobytes()).hexdigest()[:16], n_rows=len(sub))
with open(out + ".json", "w") as fh:
    json.dump(meta, fh)
print("member", spec["member"], "rows", len(sub), "sha16", meta["sha16"])
'''


def render_member_script(*, root, venv_bergson, venv_ours, preset, tag, pools_root,
                         final_step) -> str:
    """Per-worker driver: members ``m`` in ``[LO, HI)`` with ``m % NW == W`` -> dataset, bergson
    train, export, ``corrected-lds import-bergson-member``.  Skip-if-imported, so a re-run
    resumes; the run, its export and the member dataset are deleted after import."""
    return f"""#!/usr/bin/env bash
# TRAK members -- worker $1 of $2 (GPU $3), members [$4, $5).  Re-run to resume.
set -u
W=$1; NW=$2; LO=$4; HI=$5
export CUDA_VISIBLE_DEVICES=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((29800 + W))
ROOT={root}
FAILED=0
for (( m=LO; m<HI; m++ )); do
  (( m % NW == W )) || continue
  MEM=$(printf "member_%03d" $m)
  if [[ -f $ROOT/imported/$MEM.done ]]; then echo "[w$W] SKIP $MEM (imported)"; continue; fi
  echo "[w$W] TRAIN $MEM $(date -Is)"
  {{ [[ -f $ROOT/data/$MEM.json ]] \\
       || {venv_bergson}/bin/python $ROOT/make_member_data.py $ROOT/subsets/$MEM.json $ROOT/data/$MEM; }} >> $ROOT/logs/$MEM.log 2>&1 \\
    && {venv_bergson}/bin/bergson $ROOT/yaml/$MEM.yaml >> $ROOT/logs/$MEM.log 2>&1 \\
    && {venv_bergson}/bin/python -c "from bergson.utils.trainer_export import export_checkpoints; export_checkpoints('$ROOT/runs/$MEM', steps=[{int(final_step)}], overwrite=True)" >> $ROOT/logs/$MEM.log 2>&1 \\
    && {venv_ours}/bin/corrected-lds import-bergson-member --preset {preset} --tag {tag} \\
         --root {pools_root} --member $m --export-dir $ROOT/runs/$MEM/exported \\
         --data-meta $ROOT/data/$MEM.json >> $ROOT/logs/$MEM.log 2>&1
  rc=$?
  if (( rc == 0 )); then
    mkdir -p $ROOT/imported && touch $ROOT/imported/$MEM.done
    rm -rf $ROOT/runs/$MEM $ROOT/data/$MEM
    echo "[w$W] OK $MEM $(date -Is)"
  else
    echo "[w$W] FAIL $MEM rc=$rc $(date -Is)"; FAILED=1
  fi
done
echo "=== MEMBER WORKER $W DONE failed=$FAILED $(date -Is) ==="
exit $FAILED
"""


def write_member_scripts(out_dir, n_members, *, recipe, seed_base, n_train, repo,
                         revision, venv_bergson, venv_ours, preset, tag, pools_root):
    """Write each member's ``subsets/`` JSON and ``yaml/`` config, ``make_member_data.py``,
    ``run_members.sh`` and ``logs/`` under ``out_dir`` (also the bergson run root).  Members are
    seed-determined, so writing more later only adds files."""
    for sub in ("subsets", "yaml", "logs", "data", "runs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for m in range(int(n_members)):
        with open(os.path.join(out_dir, "subsets", f"member_{m:03d}.json"), "w") as f:
            json.dump({"member": m, "seed": int(seed_base) + m,
                       "rows": member_rows(m, n_train, recipe.alpha)}, f)
        with open(os.path.join(out_dir, "yaml", f"member_{m:03d}.yaml"), "w") as f:
            f.write(render_member_yaml(m, root=out_dir, recipe=recipe, seed_base=seed_base))
    with open(os.path.join(out_dir, "make_member_data.py"), "w") as f:
        f.write(render_member_data_script(repo, revision))
    sh = os.path.join(out_dir, "run_members.sh")
    with open(sh, "w") as f:
        f.write(render_member_script(root=out_dir, venv_bergson=venv_bergson,
                                     venv_ours=venv_ours, preset=preset, tag=tag,
                                     pools_root=pools_root,
                                     final_step=member_final_step(recipe, n_train)))
    os.chmod(sh, 0o755)
    return sh


# -- eval-mode replication: the bank's cells retrained and read in eval mode --- #
def setup_replication_inputs(root, M, n, alpha, Q):
    """Write the masks :func:`build_pool` stamps, ``sample_subsets(M, n, alpha, 777)`` (a prefix
    of any larger ``M``'s), as ``masks.npy`` and as ``subsets.json``, checked to round-trip
    exactly, plus the zeros score store ``validate`` requires."""
    masks = sample_subsets(int(M), int(n), alpha, seed=777)
    os.makedirs(root, exist_ok=True)
    sp = os.path.join(root, "subsets.json")
    write_subsets_json(masks, sp)
    if not np.array_equal(masks, read_subsets_json(sp, masks.shape[1])):
        raise AssertionError("[bergson] subsets.json round-trip is NOT exact")
    write_score_store(os.path.join(root, "scores"), masks.shape[1], Q)
    np.save(os.path.join(root, "masks.npy"), masks)
    return masks


def read_retrained(retrained_dir, lo, hi, model, data, device="cuda"):
    """Eval-mode query losses ``(hi - lo, Q)`` of the models ``validate --save_models`` wrote to
    ``retrained_dir/subset_<i>``, ``lo <= i < hi``, each loaded strictly: bergson's published
    readout (``model.eval()``); the same run's in-loop CSV is the dropout readout of the bank."""
    from safetensors.torch import load_file

    from corrected_lds.pools.tasks.wikitext_gpt2 import hf_to_ours, query_losses

    out = []
    for i in range(int(lo), int(hi)):
        src = os.path.join(retrained_dir, f"subset_{i}", "model.safetensors")
        if not os.path.exists(src):
            raise FileNotFoundError(f"[bergson] {src} missing; was validate run with "
                                    f"--save_models true?")
        model.load_state_dict(hf_to_ours(load_file(src)), strict=True)
        out.append(query_losses(model, data, device, eval_batch=16))
    return np.stack(out)


def build_eval_pool(out_path, run_dirs, seeds, masks, *, alpha, recipe, Q,
                    dataset="wikitext2", code_version="bergson-pool-eval-1"):
    """Pool of each replica directory's ``eval_<lo>_<hi>.npz`` readouts: :func:`build_pool`'s
    cells read in eval mode, stamped ``readout="eval"``."""
    M = np.asarray(masks).shape[0]
    ys = []
    for rd in run_dirs:
        y = np.full((M, int(Q)), np.nan)
        owner = [None] * M
        for name in sorted(os.listdir(rd)):
            if name.startswith("eval_") and name.endswith(".npz"):
                with np.load(os.path.join(rd, name), allow_pickle=False) as z:
                    lo, hi = int(z["lo"]), int(z["hi"])
                    _refuse_overlap(rd, name, owner, range(lo, hi))
                    y[lo:hi] = z["losses"]
        missing = int(np.isnan(y).any(axis=1).sum())
        if missing:
            raise ValueError(f"[bergson] {rd} is INCOMPLETE: {missing} of {M} masks have "
                             f"no eval-mode readout")
        ys.append(y)
    return _save_pool(out_path, np.stack(ys), masks, seeds, alpha=alpha, recipe=recipe,
                      Q=Q, dataset=dataset, code_version=code_version, readout="eval")


def replication_units(ranges, chunk):
    """``[(seed, lo, hi), ...]``: each ``(seed, [(start, stop), ...])`` span cut into
    ``chunk``-subset units counted from the span's own start, in the order given.  The cut is
    part of the cell: bergson seeds once per ``validate`` call, so a subset's dropout draws
    depend on its position in the call.  The released bank cut subsets 0-99 as 0-13, ..., 78-91,
    91-100; an extension cuts new subsets from their own start (100-113, ..., 295-300)."""
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError(f"[bergson] chunk must be >= 1, got {chunk}")
    units, spans = [], {}
    for seed, seed_spans in ranges:
        for lo, hi in seed_spans:
            lo, hi = int(lo), int(hi)
            if not 0 <= lo < hi:
                raise ValueError(f"[bergson] seed {seed}: bad subset range ({lo}, {hi})")
            for a, b in spans.setdefault(int(seed), []):
                if lo < b and a < hi:
                    raise ValueError(f"[bergson] seed {seed}: ranges ({a}, {b}) and "
                                     f"({lo}, {hi}) overlap")
            spans[int(seed)].append((lo, hi))
            units += [(int(seed), a, min(a + chunk, hi)) for a in range(lo, hi, chunk)]
    return units


def render_replication_worker(*, root, venv_bergson, venv_ours, preset, recipe, M, Q,
                              repo, revision, data_dir, seeds=None, chunk=None, units=None,
                              work="/root/repl_work", gpu_check="nvidia-smi -L") -> str:
    """Per-GPU driver: ``validate`` the bank's masks per seed with ``--save_models true``, keep
    the in-loop CSVs, read each saved model in eval mode, then delete the models.

    Units are ``units`` (:func:`replication_units`) or ``seeds`` over masks ``0..M`` (the
    ``subsets.json`` count) cut every ``chunk``.  Workers claim, in list order, the next unit
    nobody finished or holds (``claims/`` under ``flock``; a dead worker's claim is free), each
    on its own run path; a failed unit is retried only by a re-run, keeping its models if only
    the eval read failed.  A worker stops when ``gpu_check`` fails, as a container can lose GPUs."""
    if recipe.lr_schedule != "constant":
        raise ValueError(f"[bergson] only a constant schedule is rendered, got "
                         f"{recipe.lr_schedule!r}")
    if units is None:
        if seeds is None or chunk is None:
            raise ValueError("[bergson] pass units, or seeds and chunk")
        units = replication_units([(s, [(0, M)]) for s in seeds], chunk)
    elif seeds is not None or chunk is not None:
        raise ValueError("[bergson] pass units, or seeds and chunk -- not both")
    bad = [u for u in units if not 0 <= u[1] < u[2] <= int(M)]
    if bad or len(set(units)) != len(units) or not units:
        raise ValueError(f"[bergson] units must be distinct, non-empty and inside "
                         f"0..{M}: {bad[:3] or 'duplicates or none'}")
    return f"""#!/usr/bin/env bash
# bergson eval-mode replication -- worker $1 (GPU $2, default $1).  Re-run to resume.
set -u
W=$1
export CUDA_VISIBLE_DEVICES=${{2:-$W}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((29900 + W))
export CORRECTED_LDS_DATA={data_dir}
ROOT={root}
WORK={work}
U_SEED=({" ".join(str(int(u[0])) for u in units)})
U_LO=({" ".join(str(int(u[1])) for u in units)})
U_HI=({" ".join(str(int(u[2])) for u in units)})
M={int(M)}; Q={int(Q)}
REV={revision}
DS={repo}
FAILED=0
mkdir -p $ROOT/claims $ROOT/logs $WORK
claim() {{   # index of the next unit nobody finished and no live worker holds, else nothing
  (
    flock 9
    for (( u=0; u<${{#U_SEED[@]}}; u++ )); do
      key=r${{U_SEED[$u]}}_${{U_LO[$u]}}_${{U_HI[$u]}}
      [[ -f $ROOT/r${{U_SEED[$u]}}/done_${{U_LO[$u]}}_${{U_HI[$u]}} ]] && continue
      if [[ -f $ROOT/claims/$key ]] && kill -0 "$(cat $ROOT/claims/$key)" 2>/dev/null; then
        continue
      fi
      echo $$ > $ROOT/claims/$key; echo $u; break
    done
  ) 9> $ROOT/claims/.lock
}}
while u=$(claim) && [[ -n $u ]]; do
    SEED=${{U_SEED[$u]}}; lo=${{U_LO[$u]}}; hi=${{U_HI[$u]}}
    DEST=$ROOT/r$SEED
    RP=$WORK/r${{SEED}}_c_${{lo}}_${{hi}}            # LOCAL disk: ~0.5 GB per saved model
    TRAINED=$RP/summary_${{lo}}_${{hi}}.csv           # bergson writes it after the last model
    if ! {gpu_check} > /dev/null 2>&1; then
      echo "[w$W] GPU check failed ({gpu_check}): stopping $(date -Is)"; FAILED=1; break
    fi
    mkdir -p "$DEST"
    if [[ -f $TRAINED ]]; then
      echo "[w$W] READ r$SEED masks $lo-$hi (trained by an earlier run) $(date -Is)"
    else
      rm -rf "$RP"
      echo "[w$W] RUN r$SEED masks $lo-$hi $(date -Is)"
      {venv_bergson}/bin/bergson validate "$RP" \\
        --overwrite false --model {recipe.model} --precision {recipe.precision} \\
        --scores $ROOT/scores --subsets $ROOT/subsets.json \\
        --num_subsets $M --subset_start $lo --subset_stop $hi \\
        --subset_fraction {1.0 - float(recipe.alpha)} --seed $SEED \\
        --optimizer {recipe.optimizer} --adam_beta1 {recipe.adam_beta1} \\
        --adam_beta2 {recipe.adam_beta2} --adam_eps {recipe.adam_eps} \\
        --eps_root {recipe.eps_root} --weight_decay {recipe.weight_decay} \\
        --batch_size {recipe.batch} --num_epochs {recipe.epochs} \\
        --lr_scheduler_type constant --lr {recipe.peak_lr} \\
        --train_mode {str(bool(recipe.train_mode)).lower()} \\
        --save_models true --save_mode final \\
        --data.dataset $DS --data.split train --data.chunk_length 0 \\
        --data.data_kwargs revision=$REV \\
        --query.dataset $DS --query.split "validation[0:$Q]" --query.chunk_length 0 \\
        --query.data_kwargs revision=$REV \\
        --nproc_per_node 1 >> $ROOT/logs/w$W.log 2>&1
    fi
    if [[ -f $TRAINED ]] \\
       && {venv_ours}/bin/corrected-lds read-bergson-retrained --preset {preset} \\
            --dir $RP/retrained --lo $lo --hi $hi --out $DEST/eval_${{lo}}_${{hi}}.npz \\
            >> $ROOT/logs/w$W.log 2>&1; then
      cp -f $RP/validation_${{lo}}_${{hi}}.csv $RP/summary_${{lo}}_${{hi}}.csv $DEST/
      rm -rf "$RP"; touch $DEST/done_${{lo}}_${{hi}}
      echo "[w$W] OK r$SEED $lo-$hi $(date -Is)"; STREAK=0
    else
      [[ -f $TRAINED ]] || rm -rf "$RP"       # finished trainings stay for a read-only retry
      echo "[w$W] FAIL r$SEED $lo-$hi $(date -Is)"; FAILED=1; STREAK=$((${{STREAK:-0}} + 1))
      if (( STREAK >= 3 )); then echo "[w$W] 3 failures in a row: stopping"; break; fi
    fi
done
echo "=== REPLICATION WORKER $W DONE failed=$FAILED $(date -Is) ==="
exit $FAILED
"""


def write_replication_scripts(out_dir, **kw):
    """``run_replication.sh`` (+ ``logs/``) under ``out_dir``; launch one per GPU as
    ``run_replication.sh <w> [<gpu>]`` with distinct ``w`` (it sets ``MASTER_PORT``)."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    sh = os.path.join(out_dir, "run_replication.sh")
    with open(sh, "w") as f:
        f.write(render_replication_worker(root=out_dir, **kw))
    os.chmod(sh, 0o755)
    return sh
