"""Scheduler behind ``corrected-lds bergson-source-run``: bergson's SOURCE (Bae et al. 2024,
Adam variant, L segments) over many attributor runs on one multi-GPU pod.

Every job is one subprocess on one lane (a GPU + a scratch dir), as bergson's distributed path
crashes on RTX 5090s.  SOURCE keeps every query gradient unprojected (~340 MB per query per
copy), so queries go in pieces of ``query_batch`` (64, bergson's own batch); a score does not
depend on its piece (``query_aggregation: none``), so the query-independent factors (steps 1-4)
are fitted once per run and copied to the other lanes.  Jobs: ``train`` (seed
``seed_base + r``, Adam state kept, export, baseline-loss gate), ``factor`` (a run's first
piece, fitted fresh; its LDS must reach ``gate_min``) and ``piece`` (the rest, on copied
factors).  Marker files under ``root`` let a restarted driver resume; the driver is torch-free.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "SOURCE_FIX_COMMIT",
    "SourceConfig",
    "SourceDriver",
    "combine_source_runs",
    "piece_peak_gb",
    "piece_spans",
    "render_source_bootstrap",
    "render_source_train_yaml",
    "render_source_yaml",
    "source_segments",
]

# Upstream fix #511: the Adam variant's F_segment ignored ``inversion_cfg`` at our pin.
# Without it this bank's SOURCE scored 0.13-0.17; with it 0.4622.
SOURCE_FIX_COMMIT = "08e50effa843f8350332371cd4eef57823778e94"
STEP_FACTORS_DONE = "Step 5/8"       # steps 1-4 (query-independent factors) are complete
STEP_SCORING = "Step 8/8"            # steps 5-7 consumed; only query_grad_segment is read
_TOP_QUERY_DEPENDENT = ("query", "query.part", "scores", "scores.part")
_SEG_QUERY_DEPENDENT = ("query_grad", "query_grad_backward", "query_grad_segment", "scores")


# -- configs bergson reads -------------------------------------------------- #
def source_segments(steps, segments):
    """SOURCE's ``(checkpoint steps per segment, step_size_list)`` for ``segments`` equal
    segments of ``steps``; refuses an inexact split (bergson slices ``n_ckpts // n_segments``)."""
    steps = [int(s) for s in steps]
    n, L = len(steps), int(segments)
    if L < 1 or n % L:
        raise ValueError(f"[source] {n} checkpoints do not split into {L} equal segments")
    per = n // L
    groups = [steps[i * per:(i + 1) * per] for i in range(L)]
    ends = [g[-1] for g in groups]
    return groups, [e - s for s, e in zip([0] + ends[:-1], ends)]


def piece_spans(n_queries, query_batch):
    """``[(lo, hi), ...]`` covering ``range(n_queries)`` in pieces of ``query_batch``."""
    q, b = int(n_queries), int(query_batch)
    return [(lo, min(lo + b, q)) for lo in range(0, q, b)]


def piece_peak_gb(query_batch, segments=3, gb_per_query_copy=0.34, factors_gb=19.0):
    """Scratch a fresh piece may reach: the step-5 query plus ``L-1`` backward and ``L`` segment
    copies of ~0.34 GB per query (fp32 GPT-2 minus the tied lm_head), the factors, plus 10%."""
    copies = 1 + (int(segments) - 1) + int(segments)
    return 1.1 * (factors_gb + copies * gb_per_query_copy * int(query_batch))


def _yaml_float(x) -> str:
    """A float PyYAML (YAML 1.1) reads as one: its mantissa needs a dot (``3e-05`` is a string)."""
    s = repr(float(x))
    if "e" in s and "." not in s.split("e")[0]:
        m, e = s.split("e")
        s = f"{m}.0e{e}"
    return s


def render_source_train_yaml(r, *, root, recipe, seed_base, n_train, units_per_epoch,
                             repo, revision) -> str:
    """bergson ``train`` config for SOURCE attributor run ``r``: the trajectory config with the
    Adam state kept at every checkpoint, for SOURCE's Adam preconditioner (Bae et al. App. C)."""
    from corrected_lds.pools.bergson import render_trajectory_yaml

    y = render_trajectory_yaml(r, root=root, recipe=recipe, seed_base=seed_base,
                               n_train=n_train, units_per_epoch=units_per_epoch, repo=repo,
                               revision=revision)
    old = "save_optimizer_state: none"
    if y.count(old) != 1:
        raise AssertionError("[source] trajectory yaml no longer carries one "
                             "save_optimizer_state line")
    return y.replace(old, "save_optimizer_state: all").replace(
        "# attributor trajectory run", "# SOURCE attributor run (Adam state kept)")


def render_source_yaml(*, run_dir, run_path, lo, hi, overwrite, recipe, n_train,
                       units_per_epoch, repo, revision, segments=3, damping=1e-8,
                       query_batch=64) -> str:
    """bergson ``approxunrolling`` (SOURCE) config for validation queries ``[lo, hi)`` of the run
    in ``run_dir``, with the settings of bergson's Bae example at our pin, the dataset revision
    pinned and one GPU; ``overwrite: false`` reuses the factors already in ``run_path``."""
    from corrected_lds.pools.bergson import trajectory_steps

    if recipe.lr_schedule != "constant":
        raise ValueError(f"[source] only a constant schedule is rendered, got "
                         f"{recipe.lr_schedule!r}")
    steps = trajectory_steps(recipe, n_train, units_per_epoch)
    _, sizes = source_segments(steps, segments)
    ckpts = "\n".join(f"          - {run_dir}/checkpoints/step_{s}.ckpt" for s in steps)
    lrs = ", ".join([_yaml_float(recipe.peak_lr)] * int(segments))
    sizes_s = ", ".join(str(s) for s in sizes)
    return f"""# bergson SOURCE (Bae et al. App. C, Adam variant) -- queries [{lo}, {hi}) of {run_dir}
# rendered by corrected_lds.pools.bergson_source
steps:
  - approxunrolling:
      index_cfg:
        run_path: {run_path}
        model: {run_dir}/exported/checkpoint-{steps[-1]}
        overwrite: {str(bool(overwrite)).lower()}
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
      approx_unrolling_cfg:
        checkpoints:
{ckpts}
        segments: {int(segments)}
        lr_list: [{lrs}]
        step_size_list: [{sizes_s}]
        query:
          dataset: {repo}
          split: validation[{int(lo)}:{int(hi)}]
          chunk_length: 0
          data_kwargs: revision={revision}
        query_aggregation: none
        query_batch_size: {int(query_batch)}
        use_adam_preconditioner: true
        inversion_cfg:
          damping_factor: {_yaml_float(damping)}
"""


def _render_fix_check(venv_bergson, site_fix) -> str:
    """Bash that passes iff the pin + #511 install differs from the pin only in
    ``approx_unrolling/approx_unrolling_math.py``; ``diff``'s expected exit 1 is swallowed."""
    return f"""DIFF=$( {{ diff -rq {venv_bergson}/lib/python*/site-packages/bergson {site_fix}/bergson -x __pycache__ || true; }} )
if [[ $(printf '%s\\n' "$DIFF" | grep -c .) -eq 1 && "$DIFF" == *"approx_unrolling/approx_unrolling_math.py"*differ* ]]; then
  echo "[boot] pin+#511 OK (only approx_unrolling_math.py differs)"
else
  echo "[boot] pin+#511 does not differ from the pin in exactly approx_unrolling_math.py:"
  printf '%s\\n' "$DIFF"; exit 1
fi"""


def render_source_bootstrap(*, src="/root/corrected-lds", venv_ours="/root/venv",
                            venv_bergson="/root/bv", bergson_src="/root/bergson_src",
                            site_fix="/root/bergson_fix_site", data_dir="/workspace/cldata",
                            pin=None) -> str:
    """Fresh-pod bootstrap for the SOURCE driver (our repo unpacked at ``src``): the pinned
    bergson in ``venv_bergson`` (training, as for the bank), and pin + fix #511 as a ``--target``
    dir via ``PYTHONPATH`` for SOURCE, checked to differ only in ``approx_unrolling_math.py``."""
    from corrected_lds.pools.bergson import BERGSON_COMMIT, BERGSON_TRANSFORMERS

    pin = pin or BERGSON_COMMIT
    return f"""#!/usr/bin/env bash
# SOURCE driver bootstrap -- rendered by corrected_lds.pools.bergson_source.  Idempotent.
set -euo pipefail
echo "[boot] $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
python -c "import torch; assert torch.cuda.is_available(); print('[boot] torch', torch.__version__, torch.version.cuda)"
df -h / /dev/shm /workspace 2>/dev/null | tail -n +2
# ours: inherit the docker torch, pin what the bergson banks were read with
[[ -x {venv_ours}/bin/python ]] || python -m venv --system-site-packages {venv_ours}
{venv_ours}/bin/pip install -q -e {src} --no-deps
{venv_ours}/bin/pip install -q "transformers==4.36.2" pyarrow pyyaml safetensors
# bergson at the pin with the bank's transformers, as the pool's bootstrap installs it: the
# git build reports PyPI's version, so it is force-reinstalled over the PyPI one
[[ -x {venv_bergson}/bin/python ]] || python -m venv --system-site-packages {venv_bergson}
{venv_bergson}/bin/pip install -q bergson "transformers=={BERGSON_TRANSFORMERS}"
{venv_bergson}/bin/pip install -q --force-reinstall --no-deps \\
  "git+https://github.com/EleutherAI/bergson.git@{pin}"
# bergson pin + #511, for SOURCE only (PYTHONPATH={site_fix})
[[ -d {bergson_src}/.git ]] || git clone -q https://github.com/EleutherAI/bergson.git {bergson_src}
git -C {bergson_src} fetch -q origin
rm -rf {bergson_src}-fix && git -C {bergson_src} worktree prune
git -C {bergson_src} worktree add -q --detach {bergson_src}-fix {pin}
git -C {bergson_src}-fix -c user.name=pod -c user.email=pod@localhost cherry-pick {SOURCE_FIX_COMMIT}
rm -rf {site_fix} && {venv_bergson}/bin/pip install -q --no-deps --target {site_fix} {bergson_src}-fix
{_render_fix_check(venv_bergson, site_fix)}
CORRECTED_LDS_DATA={data_dir} {venv_ours}/bin/corrected-lds build-data --dataset wikitext2 2>&1 | grep "^\\[data\\]"
echo "=== SOURCE BOOT DONE ==="
"""


# -- the driver ------------------------------------------------------------- #
@dataclass
class SourceConfig:
    """Everything one pod's driver needs; JSON round-trip via :meth:`load` / :meth:`dump`."""

    preset: str
    runs: list                       # attributor run indices r; seed = seed_base + r
    lanes: list                      # [[gpu, scratch_dir], ...] -- one job per lane
    root: str                        # work root: runs/ factors/ out/ logs/ (local disk)
    pool: str                        # eval-readout pool.npz: the gates + the tauq identity
    baselines_dir: str               # bank dir whose pool carries baselines (train gate)
    venv_bergson: str = "/root/bv"
    venv_ours: str = "/root/venv"
    site_fix: str = "/root/bergson_fix_site"
    data_dir: str = "/workspace/cldata"
    seed_base: int = 30_000          # == tracin.TRACIN_SEED_BASE (the attributor runs' seeds)
    query_batch: int = 64
    segments: int = 3
    damping: float = 1e-8
    piece_gb: float | None = None    # scratch a fresh piece may grow to; None -> piece_peak_gb
    train_gb: float = 30.0           # a trained run with Adam state (~25 GB)
    gate_min: float = 0.30           # LDS a run's first piece must reach
    check_run: int | None = None     # re-scored on its first piece only, against check_ref
    check_ref: str | None = None     # score store of that piece from the earlier run
    check_tol: float = 0.02
    max_attempts: int = 2
    poll_s: float = 30.0
    extra_env: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))

    def dump(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)


@dataclass
class _Job:
    kind: str                        # train | factor | piece
    run: int
    span: tuple | None
    lane: int
    proc: subprocess.Popen
    log: str
    attempt: int
    t0: float
    offset: int = 0
    markers: set = field(default_factory=set)


class SourceDriver:
    """See the module docstring; ``gate``, ``free_gb`` and the ``*_argv`` methods are test seams."""

    def __init__(self, cfg, *, gate=None, free_gb=None, out=None):
        from corrected_lds.banks.presets import PRESETS

        self.cfg = cfg
        self.p = PRESETS[cfg.preset]
        self.recipe = self.p["recipe"]
        self.k = int(self.p["tracin"].get("units_per_epoch", 1))
        self.n_train, self.n_q = int(self.p["n_train"]), int(self.p["n_targets"])
        self.spans = piece_spans(self.n_q, cfg.query_batch)
        self.piece_gb = (float(cfg.piece_gb) if cfg.piece_gb is not None
                         else piece_peak_gb(cfg.query_batch, cfg.segments))
        self.gate = gate or self._pool_gate
        self.free_gb = free_gb or (lambda d: shutil.disk_usage(d).free / 1e9)
        self.out = out or sys.stdout
        self.root = cfg.root
        for d in ("runs", "factors", "out/pieces", "out/partials", "out/gates", "logs",
                  "yaml"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        for _, scratch in cfg.lanes:
            os.makedirs(scratch, exist_ok=True)
        self.jobs = {}                               # lane -> _Job
        self.attempts = {}                           # (kind, run, span) -> attempts used
        self.lane_factors = {}                       # lane -> run whose factors it holds
        self._pool = None

    # ---------------- paths / markers ------------------------------------- #
    def _run_dir(self, r):
        return os.path.join(self.root, "runs", f"run_{r:02d}")

    def _master(self, r):
        return os.path.join(self.root, "factors", f"run_{r:02d}")

    def _piece_out(self, r, span):
        return os.path.join(self.root, "out", "pieces", f"run_{r:02d}",
                            f"scores_{span[0]}_{span[1]}")

    def _partial(self, r):
        return os.path.join(self.root, "out", "partials", f"tauq_source_r_{r:03d}.npz")

    def _failed(self, r):
        return os.path.join(self.root, "out", f"FAILED_run_{r:02d}")

    def _lane_rp(self, lane):
        return os.path.join(self.cfg.lanes[lane][1], f"lane{lane}")

    def trained(self, r):
        return os.path.exists(os.path.join(self._run_dir(r), "TRAINED"))

    def factors_ready(self, r):
        return os.path.exists(os.path.join(self._master(r), "READY"))

    def piece_done(self, r, span):
        return os.path.exists(os.path.join(self._piece_out(r, span), "info.json"))

    def failed(self, r):
        return os.path.exists(self._failed(r))

    def run_spans(self, r):
        return self.spans[:1] if r == self.cfg.check_run else self.spans

    def run_done(self, r):
        if r == self.cfg.check_run:
            return os.path.exists(os.path.join(self.root, "out", "gates", f"run_{r:02d}.json"))
        return os.path.exists(self._partial(r))

    def _all_runs(self):
        runs = list(self.cfg.runs)
        if self.cfg.check_run is not None and self.cfg.check_run not in runs:
            runs = [self.cfg.check_run] + runs
        return runs

    def stopped(self):
        return os.path.exists(os.path.join(self.root, "STOP"))

    # ---------------- commands -------------------------------------------- #
    def _env(self, lane):
        gpu = int(self.cfg.lanes[lane][0])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   MASTER_PORT=str(29900 + gpu), CORRECTED_LDS_DATA=self.cfg.data_dir)
        env.update({k: str(v) for k, v in self.cfg.extra_env.items()})
        return env

    def train_argv(self, r):
        from corrected_lds.pools.bergson import trajectory_steps
        from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION

        y = os.path.join(self.root, "yaml", f"train_{r:02d}.yaml")
        with open(y, "w") as fh:
            fh.write(render_source_train_yaml(
                r, root=os.path.join(self.root, "runs"), recipe=self.recipe,
                seed_base=self.cfg.seed_base, n_train=self.n_train, units_per_epoch=self.k,
                repo=WIKITEXT2_REPO, revision=WIKITEXT2_REVISION))
        rd, c = self._run_dir(r), self.cfg
        last = trajectory_steps(self.recipe, self.n_train, self.k)[-1]
        chain = (f"rm -rf {rd} && {c.venv_bergson}/bin/bergson {y} && "
                 f"{c.venv_bergson}/bin/python -c \"from bergson.utils.trainer_export import "
                 f"export_checkpoints; export_checkpoints('{rd}', steps=[{last}], "
                 f"overwrite=True)\" && "
                 f"{c.venv_ours}/bin/corrected-lds bergson-source-check --preset {c.preset} "
                 f"--export-dir {rd}/exported --baselines-dir {c.baselines_dir} "
                 f"--expect-seed {c.seed_base + r} && touch {rd}/TRAINED")
        return ["bash", "-c", chain]

    def bergson_argv(self, yaml_path):
        c = self.cfg
        return ["env", f"PYTHONPATH={c.site_fix}", f"{c.venv_bergson}/bin/python", "-m",
                "bergson", yaml_path]

    def piece_argv(self, r, span, lane, fresh):
        from corrected_lds.pools.datasets import WIKITEXT2_REPO, WIKITEXT2_REVISION

        rp = self._lane_rp(lane)
        y = os.path.join(self.root, "yaml", f"source_{r:02d}_{span[0]}_{span[1]}_l{lane}.yaml")
        with open(y, "w") as fh:
            fh.write(render_source_yaml(
                run_dir=self._run_dir(r), run_path=rp, lo=span[0], hi=span[1],
                overwrite=fresh, recipe=self.recipe, n_train=self.n_train,
                units_per_epoch=self.k, repo=WIKITEXT2_REPO, revision=WIKITEXT2_REVISION,
                segments=self.cfg.segments, damping=self.cfg.damping,
                query_batch=self.cfg.query_batch))
        if fresh:
            prep = f"rm -rf {rp}"
        elif self.lane_factors.get(lane) == r:
            prep = "true"                                    # cleaned in place
        else:
            prep = f"rm -rf {rp} && mkdir -p {rp} && cp -a {self._master(r)}/. {rp}/ " \
                   f"&& rm -f {rp}/READY"
        return ["bash", "-c", prep + " && exec " + " ".join(self.bergson_argv(y))]

    # ---------------- logging ---------------------------------------------- #
    def log(self, msg):
        print(f"[sd] {msg} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", file=self.out, flush=True)

    # ---------------- disk -------------------------------------------------- #
    def _same_fs(self, a, b):
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except FileNotFoundError:
            return False

    def _scratch_ok(self, lane):
        """Start a piece only if ``piece_gb`` fits for it and for every other job on its
        filesystem that has not reached step 8 (each may still grow that much)."""
        scratch = self.cfg.lanes[lane][1]
        growing = sum(1 for j in self.jobs.values()
                      if j.kind in ("factor", "piece") and STEP_SCORING not in j.markers
                      and self._same_fs(self.cfg.lanes[j.lane][1], scratch))
        return self.free_gb(scratch) >= self.piece_gb * (1 + growing)

    def _train_ok(self):
        training = sum(1 for j in self.jobs.values() if j.kind == "train")
        return self.free_gb(os.path.join(self.root, "runs")) >= self.cfg.train_gb * (1 + training)

    # ---------------- scheduling ------------------------------------------- #
    def _busy(self, kind, r, span=None):
        return any(j.kind == kind and j.run == r and (span is None or j.span == span)
                   for j in self.jobs.values())

    def _exhausted(self, key):
        return self.attempts.get(key, 0) >= self.cfg.max_attempts

    def next_job(self, lane):
        """``(kind, run, span, fresh)`` for an idle lane, or None.  Priority: train ->
        a run's factor piece -> other pieces (this lane's factors first)."""
        live = [r for r in self._all_runs() if not self.failed(r) and not self.run_done(r)]
        for r in live:
            if (not self.trained(r) and not self._busy("train", r)
                    and not self._exhausted(("train", r, None)) and self._train_ok()):
                return ("train", r, None, False)
        if not self._scratch_ok(lane):
            return None
        for r in live:
            span = self.spans[0]
            if (self.trained(r) and not self.factors_ready(r) and not self._busy("factor", r)
                    and not self._exhausted(("factor", r, span))):
                return ("factor", r, span, True)
        mine = self.lane_factors.get(lane)
        for r in sorted(live, key=lambda x: x != mine):
            if not self.factors_ready(r) or r == self.cfg.check_run:
                continue
            for span in self.run_spans(r)[1:]:
                if (not self.piece_done(r, span) and not self._busy("piece", r, span)
                        and not self._exhausted(("piece", r, span))):
                    return ("piece", r, span, False)
        return None

    def start(self, lane, kind, r, span, fresh):
        key = (kind, r, span)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        tag = f"{kind}_{r:02d}" + (f"_{span[0]}_{span[1]}" if span else "")
        log = os.path.join(self.root, "logs", f"{tag}_a{self.attempts[key]}.log")
        if kind == "train":
            argv = self.train_argv(r)
        else:
            if kind == "piece" and self.lane_factors.get(lane) == r:
                self._clean(self._lane_rp(lane))
            argv = self.piece_argv(r, span, lane, fresh)
        with open(log, "ab") as fh:              # the child keeps its own descriptor
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                    env=self._env(lane), start_new_session=True)
        self.jobs[lane] = _Job(kind, r, span, lane, proc, log, self.attempts[key], time.time())
        if kind in ("factor", "piece"):
            self.lane_factors[lane] = r
        self.log(f"{tag} START lane {lane} (GPU {self.cfg.lanes[lane][0]}, "
                 f"{self.cfg.lanes[lane][1]}) attempt {self.attempts[key]}")

    # ---------------- watching -------------------------------------------- #
    @staticmethod
    def _clean(rp):
        """Drop every query-dependent output of a run path; the factors stay."""
        for name in _TOP_QUERY_DEPENDENT:
            shutil.rmtree(os.path.join(rp, name), ignore_errors=True)
        if os.path.isdir(rp):
            for seg in os.listdir(rp):
                if seg.startswith("segment_"):
                    d = os.path.join(rp, seg)
                    for name in os.listdir(d):
                        if name in _SEG_QUERY_DEPENDENT or name.startswith("scores_ckpt_"):
                            shutil.rmtree(os.path.join(d, name), ignore_errors=True)

    def _snapshot_factors(self, r, rp):
        dst, tmp = self._master(r), self._master(r) + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)

        def ignore(d, names):
            top = os.path.abspath(d) == os.path.abspath(rp)
            return [n for n in names
                    if (top and n in _TOP_QUERY_DEPENDENT)
                    or n in _SEG_QUERY_DEPENDENT or n.startswith("scores_ckpt_")]
        shutil.copytree(rp, tmp, ignore=ignore)
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
        open(os.path.join(dst, "READY"), "w").close()

    def watch(self, job):
        """Read the job's new log lines: factors done -> snapshot them; step 8 -> prune
        what step 8 no longer reads (the step-5 query and the backward walk)."""
        if job.kind == "train":
            return
        with open(job.log, "rb") as fh:
            fh.seek(job.offset)
            new = fh.read().decode("utf-8", "replace")
            job.offset = fh.tell()
        rp = self._lane_rp(job.lane)
        if (STEP_FACTORS_DONE in new and STEP_FACTORS_DONE not in job.markers):
            job.markers.add(STEP_FACTORS_DONE)
            if job.kind == "factor" and not self.factors_ready(job.run):
                self._snapshot_factors(job.run, rp)
                self.log(f"run_{job.run:02d} factors READY (lane {job.lane})")
        if STEP_SCORING in new and STEP_SCORING not in job.markers:
            job.markers.add(STEP_SCORING)
            shutil.rmtree(os.path.join(rp, "query"), ignore_errors=True)
            if os.path.isdir(rp):
                for seg in os.listdir(rp):
                    shutil.rmtree(os.path.join(rp, seg, "query_grad_backward"),
                                  ignore_errors=True)
            self.log(f"run_{job.run:02d} {job.span[0]}-{job.span[1]} scoring (lane {job.lane}, "
                     f"{self.free_gb(self.cfg.lanes[job.lane][1]):.0f} GB free)")

    def reap(self, lane, job, rc):
        del self.jobs[lane]
        tag = f"{job.kind}_{job.run:02d}" + (f"_{job.span[0]}_{job.span[1]}" if job.span else "")
        mins = (time.time() - job.t0) / 60
        if job.kind == "train":
            if rc == 0 and self.trained(job.run):
                self.log(f"{tag} OK ({mins:.0f} min)")
            else:
                self.log(f"{tag} FAIL rc={rc}")
                if self._exhausted(("train", job.run, None)):
                    self._fail(job.run, f"training failed {self.cfg.max_attempts}x")
            return
        rp = self._lane_rp(lane)
        src = os.path.join(rp, "scores")
        if rc == 0 and os.path.exists(os.path.join(src, "info.json")):
            out = self._piece_out(job.run, job.span)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.rmtree(out + ".tmp", ignore_errors=True)
            shutil.copytree(src, out + ".tmp")
            os.replace(out + ".tmp", out)
            self._clean(rp)
            self.log(f"{tag} OK ({mins:.0f} min)")
            if job.kind == "factor":
                if not self.factors_ready(job.run):
                    self._snapshot_factors(job.run, rp)
                self._gate_run(job.run)
            self._maybe_finish(job.run)
            return
        self.lane_factors.pop(lane, None)
        shutil.rmtree(rp, ignore_errors=True)
        if self.failed(job.run) or self.stopped():
            self.log(f"{tag} stopped")
            return
        self.log(f"{tag} FAIL rc={rc}")
        if self._exhausted((job.kind, job.run, job.span)):
            self._fail(job.run, f"{tag} failed {self.cfg.max_attempts}x")

    def _gate_run(self, r):
        ok, info = self.gate(r, self._piece_out(r, self.spans[0]))
        info = dict(info, ok=bool(ok))
        with open(os.path.join(self.root, "out", "gates", f"run_{r:02d}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        self.log(f"run_{r:02d} GATE {'PASS' if ok else 'FAIL'} {json.dumps(info)}")
        if r == self.cfg.check_run:
            shutil.rmtree(self._run_dir(r), ignore_errors=True)
            shutil.rmtree(self._master(r), ignore_errors=True)
        if not ok:
            if r == self.cfg.check_run:
                self.log("environment CHECK failed -> STOP (no new jobs, running ones killed)")
                open(os.path.join(self.root, "STOP"), "w").close()
                self.kill_all()
            self._fail(r, f"gate: {info}")

    def _fail(self, r, why):
        with open(self._failed(r), "w") as fh:
            fh.write(why + "\n")
        for lane, j in list(self.jobs.items()):
            if j.run == r:
                self._kill(j)
        shutil.rmtree(self._master(r), ignore_errors=True)
        self.log(f"run_{r:02d} FAILED: {why}")

    def _kill(self, job):
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill_all(self):
        for j in list(self.jobs.values()):
            self._kill(j)

    def _maybe_finish(self, r):
        if r == self.cfg.check_run or self.failed(r) or os.path.exists(self._partial(r)):
            return
        if not all(self.piece_done(r, s) for s in self.spans):
            return
        from corrected_lds.banks.artifacts import write_tauq
        from corrected_lds.pools.bergson import read_score_store

        S = np.concatenate([read_score_store(self._piece_out(r, s)) for s in self.spans],
                           axis=1)
        prov = self.provenance(r)
        write_tauq(self._partial(r), -S, self.pool(), provenance=prov)
        info = self._lds_info(-S)
        self.log(f"run_{r:02d} DONE: {json.dumps(info)}")
        shutil.rmtree(self._run_dir(r), ignore_errors=True)
        shutil.rmtree(self._master(r), ignore_errors=True)

    def provenance(self, r):
        from corrected_lds.pools.bergson import trajectory_steps

        steps = trajectory_steps(self.recipe, self.n_train, self.k)
        return {"method": f"source_L{self.cfg.segments}", "harness": "bergson",
                "scorer": "bergson approxunrolling", "use_adam_preconditioner": True,
                "segments": int(self.cfg.segments), "damping": float(self.cfg.damping),
                "train_seed": int(self.cfg.seed_base + r),
                "checkpoints": f"steps {steps[0]}..{steps[-1]} ({len(steps)}, "
                               f"{_unit_name(self.k)})",
                "bergson_commit": f"{self.recipe.bergson_commit[:8]} + #511 "
                                  f"({SOURCE_FIX_COMMIT[:8]})",
                "sign": "negated: bergson SOURCE scores are loss-signed",
                "queries": f"{len(self.spans)} pieces of <={self.cfg.query_batch} "
                           f"validation queries, factors fitted once and shared"}

    # ---------------- gates -------------------------------------------------- #
    def pool(self):
        if self._pool is None:
            from corrected_lds.banks.artifacts import load_pool
            self._pool = load_pool(self.cfg.pool)
        return self._pool

    def _lds(self, tau, lo):
        from corrected_lds.evaluate import _rank_correlations
        pool = self.pool()
        g = pool.subsets.astype(np.float64) @ tau
        y = pool.margins.mean(axis=0)[:, lo:lo + tau.shape[1]]
        return float(np.nanmean(_rank_correlations(g, y)[0]))

    def _lds_info(self, tau):
        return {"lds_naive_all_queries": round(self._lds(tau, 0), 4)}

    def _pool_gate(self, r, store):
        from corrected_lds.pools.bergson import read_score_store
        lo = self.spans[0][0]
        tau = -read_score_store(store)
        v = self._lds(tau, lo)
        info = {"run": r, "queries": list(self.spans[0]), "lds": round(v, 4)}
        if r == self.cfg.check_run:
            ref = -read_score_store(self.cfg.check_ref)
            if ref.shape[1] < tau.shape[1]:
                raise ValueError(f"[sd] check_ref has {ref.shape[1]} queries, the check "
                                 f"piece {tau.shape[1]}; both must start at query {lo}")
            ref = ref[:, :tau.shape[1]]          # the same queries: both pieces start at lo
            vr = self._lds(ref, lo)
            info.update(ref_lds=round(vr, 4), score_corr=round(float(
                np.corrcoef(tau.ravel(), ref.ravel())[0, 1]), 4))
            return abs(v - vr) <= self.cfg.check_tol, info
        return v >= self.cfg.gate_min, info

    # ---------------- main loop -------------------------------------------- #
    def status(self):
        runs = {}
        for r in self._all_runs():
            done = sum(self.piece_done(r, s) for s in self.run_spans(r))
            runs[f"run_{r:02d}"] = {"trained": self.trained(r),
                                    "factors": self.factors_ready(r),
                                    "pieces": f"{done}/{len(self.run_spans(r))}",
                                    "failed": self.failed(r), "done": self.run_done(r)}
        lanes = {str(i): (f"{j.kind} run_{j.run:02d}"
                          + (f" {j.span[0]}-{j.span[1]}" if j.span else "")
                          + (" scoring" if STEP_SCORING in j.markers else "")
                          + f" {(time.time() - j.t0) / 60:.0f} min")
                 for i, j in sorted(self.jobs.items())}
        return {"runs": runs, "lanes": lanes, "stopped": self.stopped()}

    def finished(self):
        return not self.jobs and all(self.failed(r) or self.run_done(r)
                                     for r in self._all_runs())

    def housekeeping(self):
        """Finish what a previous driver left half-done: a scored first piece without
        its gate verdict, a run whose pieces are all scored but not combined."""
        gates = os.path.join(self.root, "out", "gates")
        for r in self._all_runs():
            if self.failed(r) or self._busy("factor", r):
                continue
            if (self.piece_done(r, self.spans[0])
                    and not os.path.exists(os.path.join(gates, f"run_{r:02d}.json"))):
                self._gate_run(r)
            if not self.failed(r):
                self._maybe_finish(r)

    def step(self):
        """One scheduling round: reap, watch, housekeeping, assign.  Returns
        ``finished()`` (or True once STOP is set and nothing is running)."""
        if os.path.exists(os.path.join(self.root, "STOP_NOW")):
            open(os.path.join(self.root, "STOP"), "w").close()
            self.kill_all()
        for lane, job in list(self.jobs.items()):
            self.watch(job)
            rc = job.proc.poll()
            if rc is not None:
                self.watch(job)
                self.reap(lane, job, rc)
        self.housekeeping()
        if not self.stopped():
            for lane in range(len(self.cfg.lanes)):
                if lane in self.jobs:
                    continue
                nxt = self.next_job(lane)
                if nxt:
                    self.start(lane, *nxt)
        with open(os.path.join(self.root, "status.json.tmp"), "w") as fh:
            json.dump(self.status(), fh, indent=1)
        os.replace(os.path.join(self.root, "status.json.tmp"),
                   os.path.join(self.root, "status.json"))
        if self.stopped() and not self.jobs:
            return True
        return self.finished()

    def run(self):
        lock = os.path.join(self.root, "driver.pid")
        if os.path.exists(lock):
            with open(lock) as fh:
                pid = int(fh.read().strip() or 0)
            if pid and pid != os.getpid() and _alive(pid):
                raise RuntimeError(f"[sd] another driver (pid {pid}) owns {self.root}")
        with open(lock, "w") as fh:
            fh.write(str(os.getpid()))
        self.log(f"start: runs {self._all_runs()} on {len(self.cfg.lanes)} lanes, "
                 f"{len(self.spans)} pieces per run")
        try:
            while not self.step():
                time.sleep(self.cfg.poll_s)
        except KeyboardInterrupt:
            self.kill_all()
            raise
        finally:
            os.remove(lock)
        failed = [r for r in self._all_runs() if self.failed(r)]
        self.log(("STOPPED" if self.stopped() else "DONE")
                 + f": {sum(self.run_done(r) for r in self._all_runs())} runs done, "
                   f"failed {failed}")
        return 1 if failed or self.stopped() else 0


def _unit_name(k):
    return {1: "epoch", 2: "half-epoch"}.get(int(k), f"1/{int(k)}-epoch")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# -- combine single-run SOURCE tauqs -> E-ensemble + replica-resolved pool_tda --- #
_COMPAT = ("method", "harness", "scorer", "use_adam_preconditioner", "segments", "damping",
           "checkpoints", "sign")


def combine_source_runs(paths, pool, out_dir, *, seed_base=30_000):
    """Plain mean of single-run SOURCE tauqs (Bae et al. §3.3's multiple-training-run estimator)
    -> ``tauq_<method>_e<E>.npz``, plus the per-run slices as a replica-resolved
    ``pool_tda_<method>.npz`` in train-seed order (a prefix is the first runs).  Refuses
    differing scorer settings, a repeated train seed, or a build other than the pin + #511."""
    from corrected_lds.banks.artifacts import load_tauq, write_pool_tda, write_tauq

    taus = [load_tauq(p) for p in paths]
    provs = [dict(t.provenance) for t in taus]
    base = {k: provs[0].get(k) for k in _COMPAT}
    for p, prov in zip(paths, provs):
        if {k: prov.get(k) for k in _COMPAT} != base:
            raise ValueError(f"[combine] {p} was scored with different settings: "
                             f"{ {k: prov.get(k) for k in _COMPAT} } vs {base}")
        if not str(prov.get("bergson_commit", "")).startswith("0ad2e5b0 + #511"):
            raise ValueError(f"[combine] {p}: bergson build {prov.get('bergson_commit')!r} "
                             f"is not the pin + #511")
    seeds = [int(prov["train_seed"]) for prov in provs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"[combine] repeated train seeds {sorted(seeds)}")
    order = np.argsort(seeds)
    taus = [taus[i] for i in order]
    seeds = [seeds[i] for i in order]
    E = len(taus)
    mean = np.mean([t.tauq.astype(np.float64) for t in taus], axis=0)
    prov = dict(base, n_models=E, train_seeds=seeds,
                bergson_commit=f"{provs[0]['bergson_commit'].split(' cherry')[0]}",
                aggregation="plain-mean-trajectories")
    method = base["method"]
    os.makedirs(out_dir, exist_ok=True)
    tq = write_tauq(os.path.join(out_dir, f"tauq_{method}_e{E}.npz"), mean, pool,
                    n_models=E, provenance=prov)
    S = pool.subsets.astype(np.float64)
    g = np.stack([S @ t.tauq.astype(np.float64) for t in taus])
    pt = write_pool_tda(os.path.join(out_dir, f"pool_tda_{method}.npz"), g, pool,
                        subsets=pool.subsets, method=method, inner="single-trajectory",
                        e_features=1, seed_base=seed_base, provenance=prov)
    return tq, pt


def _parse_lanes(gpus, shm_lanes, shm_dir, disk_dir):
    """First ``shm_lanes`` GPUs scratch on ``shm_dir``, the rest on ``disk_dir``."""
    lanes = []
    for i, g in enumerate(gpus):
        lanes.append([int(g), shm_dir if i < int(shm_lanes) else disk_dir])
    return lanes

