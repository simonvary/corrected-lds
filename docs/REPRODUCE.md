# Reproducing the results

Three levels, each needing more than the one before:

1. **Score the released banks**: CPU, minutes.
2. **Rebuild the paper's figures** from the released banks: CPU, about an hour.
3. **Rebuild a bank**: GPU, from a few GPU-hours (one attribution file) to thousands (a pool).

File formats and method definitions are in [`FORMATS.md`](FORMATS.md); how each released file
was made is in [`PROVENANCE.md`](../PROVENANCE.md).

## Setup

```bash
git clone https://github.com/simonvary/corrected-lds && cd corrected-lds
pip install -e .              # CPU: scoring (numpy, scipy)
pip install -e ".[figs]"      # the paper's figures (matplotlib)
pip install -e ".[gpu]"       # rebuilding: torch, traker, kronfluence, transformers
pip install -e ".[imagenet]"  # the ImageNet data build
```

`uv sync --locked` (with `--extra figs`, `--extra gpu` and so on) installs the exact versions
of CI's main job. The banks themselves were built on the GPU pods' stack, pinned in
`env/release-gpu-2026-10.txt` (Python 3.12, torch 2.8.0+cu128), which CI's torch job tests on
CPU. On GPU, TRAK and TracIn need traker's CUDA projector, `fast_jl`, which the `gpu` extra
leaves out because it has to be built against the installed torch (`pip install "traker[fast]"`
builds it); at run time it needs torch's `lib` directory on `LD_LIBRARY_PATH`. Dataset caches
go to `~/.cache/corrected_lds`; set `CORRECTED_LDS_DATA` to move them.

## 1. Score the released banks

```bash
corrected-lds fetch-banks                       # all banks into banks/, sha256-verified
corrected-lds fetch-banks --bank cifar10        # or just one
corrected-lds evaluate banks/cifar10 --tauq trak10_e10 --json result.json
corrected-lds compare banks/cifar10 --consistency
```

`fetch-banks` checks every file against the sha256 pinned in `banks-manifest.json`. `evaluate`
attaches the matching `pool_tda` file automatically, so the result also holds the LDS at every
smaller ensemble size. `compare --consistency` checks that each plain-mean tauq is the mean of
its pool_tda replicas.

## 2. Rebuild the figures

`paper/figN.py` builds paper Figure N. The committed plotdata in `paper/plotdata/` holds every
number the figures draw, and `paper/figures/` holds the rendered PDFs and PNGs.

```bash
python paper/fig1.py          # render one figure from committed plotdata: seconds, no banks
```

With the same matplotlib, a re-render reproduces the committed files byte for byte.

To recompute the plotdata from the banks, fetch the four banks the figures read: about
4.2 GB, and no `imagenet`. Then run every script with `--recompute`, stage by stage, since
later scripts read what earlier ones wrote:

```bash
corrected-lds fetch-banks --bank cifar10 --bank mnist --bank qnli --bank wikitext2_gpt2_bae
R="--recompute --banks-root banks"
# stage 1: reads banks only
for s in _rmse_corrected _rmse_heldout _spearman_debias _fig2_panels _bridge_estimators \
         _proxy_sweep fig3 fig6 fig8 fig9; do python paper/$s.py $R; done
# stage 2
for s in _fig1_panels fig2 _bridge_bias fig4 fig7; do python paper/$s.py $R; done
# stage 3, then 4
for s in fig1 _fig5_panels; do python paper/$s.py $R; done
python paper/fig5.py $R
# the talk's figures (slides/README.md): the cifar10 bank, then figure 2's plotdata
for i in 1 2 3; do python slides/fig_slides$i.py $R; done
```

All of it runs on CPU. A recompute on the same machine is byte-identical to the committed
plotdata; another CPU or BLAS differs in the last digits.

| figure | script and the stages it reads | bank files read |
|---|---|---|
| 1 | `fig1` ← `_fig1_panels` (← `_rmse_heldout`, `_spearman_debias`), `_rmse_corrected` | cifar10: pool, `pool_tda_{ekfac,trak10,trak20,trak1,source_L3}`, `tauq_trak1_e100` |
| 2 | `fig2` ← `_fig2_panels` | pools of cifar10, qnli and wikitext2; cifar10 `pool_tda_{source_L3,ekfac,trak10,trak20,trak1}`; qnli and wikitext2 `pool_tda_{ekfac,trak5,tracin,trak1}` |
| 3 | `fig3` | the three pools; `pool_tda` of cifar10 `{ekfac,source_L3,trak10}`, qnli and wikitext2 `{ekfac,tracin,trak5}`; wikitext2 `tauq_source_L3_e1` |
| 4 | `fig4` ← `_spearman_debias` | cifar10: pool, `tauq_trak1_e100` |
| 5 | `fig5` ← `_fig5_panels` (← `_bridge_bias` ← `_bridge_estimators`), `_proxy_sweep` | pools of cifar10 and qnli; cifar10 `pool_tda_{ekfac,trak1,source_L3}`; qnli `pool_tda_{ekfac,trak1,trak5,tracin}` |
| 6 | `fig6` | cifar10, qnli and wikitext2: pool plus three `pool_tda` files each |
| 7 | `fig7` ← `fig6` | none |
| 8 | `fig8` | cifar10: pool, `tauq_trak1_e100` |
| 9 | `fig9` | cifar10 and mnist: pool, `tauq_{trak1_e100,ekfac_e30,source_L3_e30}` |

## 3. Rebuild a bank

Every build is driven by a frozen preset (`corrected_lds.banks.presets`): the recipe, the data,
the masks and each method's settings. The expensive commands print their cost and run only with
`--yes`. Everything resumes after an interruption, and every resume, merge and load re-checks
the identity: content hashes, the mask draw, the stamped indices.

### The workflow

```bash
corrected-lds build-data --dataset cifar10
corrected-lds build-pool --preset cifar10 --tag cifar10 --M 300 --R 40 --yes
corrected-lds compute-tau --preset cifar10 --tag cifar10 --method source --n-models 30 --yes
corrected-lds compare pools/cifar10/tauq_source_L3_e30.npz banks/cifar10/tauq_source_L3_e30.npz
corrected-lds evaluate pools/cifar10 --tauq source_L3_e30
```

Builds go to `pools/<tag>/`. `compare NEW REF` checks a rebuilt file against the released one:
identity first, then how closely the numbers agree. `compute-tau` reuses a matching file that
already exists, so give each rebuild a fresh tag.

**Several GPUs.** Run `build-pool ... --shard i N` once per GPU (pin each process with
`CUDA_VISIBLE_DEVICES`), then `build-pool ... --merge N` on one machine. A cell's seed and its
shard depend only on its position, so any N gives the same pool. In the same way, TRAK
splits by members (`compute-tau --method trak --members 0-24`, one process per GPU, then a
run without `--members` that assembles the cached pieces bit for bit), and TracIn, EK-FAC and
SOURCE split by runs (`--runs`, then a run without it).

**Determinism.** A retrained cell is bit-identical on the same GPU model and torch version. On
another GPU model, training drifts: mnist cells retrained on an L40S sit 0.29 of the
seed-to-seed spread away from the released ones, and a cifar10 TRAK member retrained there is
as unrelated to its released counterpart as two members are to each other. A file built from
retrained models therefore reproduces the released LDS rather than the released arrays.
fast_jl, the CUDA projector of TRAK and TracIn, draws its projection from the seed and the
GPU's streaming-multiprocessor count, so the presets pin the count of the GPU that made the
release (`projector_num_sms`: 170, an RTX 5090's) and another GPU model draws the same
projection: released WikiText-2 TRAK members re-featurized on an L40S (142 SMs) match the
release to 5e-4 relative.

### What rebuilds the released files

Status: **ok**, rebuilt and matched; **ok (stat)**, rebuilt on another GPU model, where
retraining drifts: a rebuilt score file gives the released corrected LDS within 2 SE (each row
gives the rebuilt value against the released one, both scored on the released pool), and
rebuilt pool cells sit well inside the seed-to-seed spread; **unverified**, the command is
right but the arrays have not been re-made; **gap**, the command does not yet reproduce the
file (see the end of this page). `hw`: bitwise only on the GPU model that made it, at float
level on others.

**cifar10** (`build-pool --preset cifar10 --M 300 --R 40`, about 300 GPU-hours)

| files | command | status |
|---|---|---|
| `pool.npz` | `build-pool --preset cifar10 --M 300 --R 40` | ok: a retrained cell matches to the last bit |
| `tauq_trak10_e{2,10}`, `tauq_trak1_e{20,100}`, `pool_tda_trak1` | `compute-tau --method trak --n-models 100 --aggregation both` | ok (stat): `trak10_e10` 0.5826 vs 0.5826, `trak1_e100` 0.4965 vs 0.4961 |
| `pool_tda_trak10` (members 0-499), `pool_tda_trak20` (members 100-499) | members 100-499 with `--members`, assembled with `--block 100-499` at `--e-features 10` and `20`, then `merge-pool-tda` onto the members-0-99 file | gap: built in three rounds; the recipe is untested |
| `tauq_source_L3_e30`, `pool_tda_source_L3` | `compute-tau --method source --n-models 30` | unverified |
| `tauq_ekfac_e30`, `pool_tda_ekfac` | `compute-tau --method ekfac --n-models 30`, on the same 30 trajectories | unverified |

**qnli** (`build-pool --preset qnli --M 400 --R 20`, about 2,300-2,650 GPU-hours)

| files | command | status |
|---|---|---|
| `pool.npz` (bf16 finetunes) | `build-pool --preset qnli --M 400 --R 20` | ok: 14 retrained cells bit-identical |
| `tauq_trak1_e{10,20,50,100}`, `tauq_trak5_e{2,4,10,20}`, `pool_tda_trak{1,5}` | `compute-tau --method trak --n-models 100 --aggregation both` | ok, hw |
| `tauq_tracin_e20`, `pool_tda_tracin` | `compute-tau --method tracin --n-models 20` | ok, hw |
| `tauq_ekfac_e20`, `pool_tda_ekfac` | `compute-tau --method ekfac --n-models 20` on the same 20 trajectories | ok |

**mnist** (`build-pool --preset mnist --M 300 --R 40`, about 30 GPU-hours)

| files | command | status |
|---|---|---|
| `pool.npz` | `build-pool --preset mnist --M 300 --R 40` | ok (stat): 16 cells, 0.29 of the seed-to-seed spread from the release |
| `tauq_trak1_e100` | `compute-tau --method trak --n-models 100 --aggregation both` | ok (stat): 0.6448 vs 0.6447 |
| `tauq_source_L3_e30` | `compute-tau --method source --n-models 30` | ok (stat): 0.6450 vs 0.6457 |
| `tauq_ekfac_e30` | `compute-tau --method ekfac --n-models 30`, on the SOURCE trajectories | ok (stat): 0.5846 vs 0.5821 |

The GPU that made the mnist bank was not recorded; these were rebuilt on an L40S, SOURCE first
(it trains the 30 full-set runs), then EK-FAC (it trains nothing new), in a tag holding only a
copy of the released `pool.npz`.

**imagenet** (pool only)

```bash
corrected-lds build-data --dataset imagenet --smoke   # checks access and schema, ~1 GB
corrected-lds build-data --dataset imagenet --yes     # ~153 GB download, ~75 GB cache
corrected-lds build-pool --preset imagenet --tag imagenet --M 100 --R 5 --yes
```

ImageNet is license-gated: accept the terms on the
[`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) page and set
`HF_TOKEN` (or `hf auth login`). The cache's content hashes depend on the Pillow version that
decoded the images. Status: unverified.

### wikitext2_gpt2_bae: models trained by bergson

The GPT-2 bank follows Bae et al. 2024 as packaged in EleutherAI's
[bergson](https://github.com/EleutherAI/bergson), pinned at commit `0ad2e5b`; bergson trains
every model, in its own Python environment. Our side renders bergson's configs, imports what
it trains, and scores it. Every step in our environment first needs
`corrected-lds build-data --dataset wikitext2`.

- **Pool** (`pool.npz` eval-mode readout, `pool_dropout_readout.npz`; 20 × 400 × 481, about
  605 GPU-hours), on a GPU pod:

  ```bash
  corrected-lds bergson-pool setup --dir /root/wt      # inputs, bootstrap.sh, run_replication.sh
  bash /root/wt/bootstrap.sh                           # bergson at the pin, in /root/bv
  setsid nohup bash /root/wt/run_replication.sh 0 0 > /root/wt/logs/w0.out 2>&1 &
  corrected-lds bergson-pool check --dir /root/wt      # about an hour later: the gate
  corrected-lds bergson-pool assemble --dir /root/wt   # when every worker is done
  ```

  Once the check passes, start a worker on each other GPU the same way
  (`run_replication.sh 1 1`, and so on); re-running a worker resumes it. bergson seeds once per
  `validate` call, so a cell depends on its place in the call, and `setup` renders the
  release's 640 calls: every seed's subsets 0-99, 100-299 and 300-399, cut every 13 from their
  start. Worker 0 starts with seed 1004's subsets 0-12, which `check` compares with the released
  pool (`--ref`, by default where `fetch-banks` puts it). `assemble` writes
  `pools/wikitext2_gpt2_bae/`. Status: ok, hw: retrained cells were bit-identical on RTX 5090s,
  and `assemble` rebuilds both released pools byte for byte from the release's own chunks.
- **TracIn and EK-FAC** (`tauq_tracin_e10`, `tauq_ekfac_e{5,10}` and their `pool_tda`): render
  the training configs with `bergson.write_trajectory_scripts` (seeds 30000-30009), train them
  with bergson, `corrected-lds import-bergson-trajectory` each run, then
  `compute-tau --preset wikitext2_gpt2_bae --method tracin`, and `--method ekfac --n-models 5`
  followed by `--method ekfac` (10 runs; its `pool_tda` replaces the 5-run one). Status:
  unverified; TracIn is hw.
- **TRAK** (`tauq_trak1_e{10,20,50}`, `tauq_trak5_e{2,4,10}`, `pool_tda_trak{1,5}`): render
  the members with `bergson.write_member_scripts` (seeds 10000-10049), train each on its own
  random half with bergson, `corrected-lds import-bergson-member`, then `compute-tau --method
  trak --n-models 50 --aggregation both`. Status: ok, hw: re-featurized on an L40S, members 0-4
  match their released slices to 5e-4 relative. The released files mix members featurized on
  two GPU models, so no single machine reproduces them bitwise.
- **SOURCE** (`tauq_source_L3_e1`): bergson's own SOURCE scorer on run 0 (train seed 30000),
  on a pod with this repository cloned to `/root/corrected-lds`, which the bootstrap installs:

  ```bash
  corrected-lds bergson-source-config --runs 0 --gpus 0 --out /root/src \
      --pool banks/wikitext2_gpt2_bae/pool.npz --baselines-dir banks/wikitext2_gpt2_bae
  bash /root/src/bootstrap_source.sh
  /root/venv/bin/corrected-lds bergson-source-run --config /root/src/source_config.json
  ```

  A query piece needs about 165 GB of scratch, on `/dev/shm` unless `--shm-lanes 0`. The run's
  `/root/source_e10/out/partials/tauq_source_r_000.npz` is `tauq_source_L3_e1`, and
  `bergson-source-combine` averages several runs. Status: unverified: a rerun made before the
  bootstrap pinned bergson's transformers agreed only statistically (score correlation 0.968).
- **bergson's EK-FAC** (`tauq_ekfac_bergson.npz`, a check of ours): bergson's own EK-FAC on one
  full-data model, seeded with the pool's first seed (1004), on one GPU with about 330 GB free
  in `--dir`:

  ```bash
  corrected-lds bergson-ekfac setup --dir /root/ek     # bootstrap.sh, configs, run_ekfac.sh
  bash /root/ek/bootstrap.sh && bash /root/ek/run_ekfac.sh
  ```

  `run_ekfac.sh` trains and exports the model, scores all 481 queries, and ends with
  `bergson-ekfac import`, which writes `pools/wikitext2_gpt2_bae/tauq_ekfac_bergson.npz`. Status:
  ok (stat): 0.4938 vs 0.4938, rebuilt on an L40S with the scoring split into four query pieces
  to fit its 200 GB disk; `import` also turns the release's own score store into the released
  file byte for byte.

### Known gaps

cifar10's `pool_tda_trak10` and `pool_tda_trak20` were built in three rounds, a recipe that
has not been re-run. mnist's TRAK file has no projector pin, as the GPU that made it was not
recorded.

Rebuilding every bank from scratch costs about 3,800-4,100 RTX 5090 GPU-hours, of which the
qnli pool is about 2,300-2,650.

## Sharing a bank

```bash
pip install -e ".[hub]"
corrected-lds push-bank pools/<tag> --repo <user>/<dataset-repo>
```

`push-bank` refuses anything the loaders would refuse, uploads the bank in one commit (the
repo is created private unless `--public` is given) and prints the update for
`banks-manifest.json`, the push's commit as its new `url` and the bank's sha256s, so that
`fetch-banks` can download and verify it. `fetch-banks` sends `HF_TOKEN`, when set, only to the Hub over https.
