# Provenance

How each released bank file was made, and why it can be trusted: the code and hardware behind
it, how it grew, what it was checked against, and the corrections made along the way. How to
rerun anything is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md); what the files hold, how they are
named and what each method computes is in [`docs/FORMATS.md`](docs/FORMATS.md). Dates are in
2026. Each LDS value names its `evaluate` field: `corrected_rho_inf` (the Pearson form) or the
corrected Spearman LDS (`debiased_noise_corrected_spearman`).

## The release

The banks are in the Hub dataset repository `corrected-lds/data`. `banks-manifest.json`,
committed here, pins each of the 49 released files by its full sha256 at the dataset's single
revision, and `corrected-lds fetch-banks` checks every download against it. Hub files are never changed in place: an
update is a new commit. No model checkpoints are distributed, nor the per-run score matrices
behind the pool_tda files.

| bank | files | pool: R × M × Q, n | pool `code_version` |
|---|---|---|---|
| `cifar10` | 12 | 40 × 300 × 2,000, 50,000 | `corrected-lds-0.1.0` |
| `qnli` | 15 | 20 × 400 × 2,000, 50,000 | `corrected-lds-7a1a04f` |
| `mnist` | 4 | 40 × 300 × 2,000, 6,144 | `corrected-lds-0.1.0` |
| `imagenet` | 1 | 5 × 100 × 2,000, 1,281,167 | `corrected-lds-source-798d4cf-dirty` |
| `wikitext2_gpt2_bae` | 17 | 20 × 400 × 481, 4,656 | `bergson-pool-eval-1` |

A pool's `code_version` names the code that last wrote it: a commit when built from a git
checkout (`-dirty` for uncommitted changes), else the package version `corrected-lds-0.1.0`,
which names no commit. The commits named in the files and here are in the private development
history; this repository starts at v1.0.0; the GPT-2 pools carry fixed labels from `pools/bergson.py`. Tauq and
pool_tda files record none; the bank sections name the commit behind one where the build record
gives it. Every pool's data hashes
equal the `presets.DATA_HASHES` pins the loaders check on every load, and every file in
`paper/plotdata/` stamps the identity of the pools it read, all matching this table.

## cifar10

**Pool.** 12,000 ResNet-9 retrains on α = 0.5 subsets of the CIFAR-10 training set (40 seeds ×
300 masks, 2,000 test queries; the TRAK App. A.1 recipe: 24 epochs, peak LR 0.5). Its stamp
names no commit, but its cell (0, 0) has the battery mean of the bit-for-bit trainer check,
6.285445046067238 (see Findings). `lambda_median` 3.430146 and `sigma2_mean` 2.903498,
functions of the pool alone, are bit-equal across every evaluation.

**TRAK** (`tauq_trak10_e{2,10}`, `tauq_trak1_e{20,100}`, `pool_tda_trak{1,10,20}`): members on
α = 0.5 subsets with seed 10000 + member id, featurized on augmented inputs with traker 0.3.2's
fast_jl CUDA projector at k = 20,480.
- One pass over members 0-99, at `38ded77` on an RTX 5090, made the four tauqs (the 20-member
  ones as prefix snapshots) and `pool_tda_trak1`, whose 100 slices average to `tauq_trak1_e100`
  to 2.6e-08. `corrected_rho_inf`: `trak10_e10` **0.582628** (SE 0.00099; naive single-seed
  0.285896), `trak10_e2` 0.357315, `trak1_e100` 0.496087, `trak1_e20` 0.278701.
- `pool_tda_trak10`: 50 Eq. 15 units of 10 (members 0-499), the first 10 equal to that pass's
  units. `pool_tda_trak20`: 20 units of 20 (members 100-499). Members 340-499 were trained and
  featurized on RTX 5090s on 2026-10-12 and appended with `merge-pool-tda`, the released units
  first and bit-identical; a new TRAK-20 unit scores 0.3436 against 0.3438 for a released one.
  At 20 units TRAK-20's corrected Spearman LDS is 0.689; the 1/E law forecast 0.687 from two.
- Caveats: the mean of `pool_tda_trak10`'s first 10 units, 0.572892, is 0.0097 (about 10 SE)
  below the pooled `tauq_trak10_e10`, which is the headline. `pool_tda_trak20` stamps
  `members: [100, 500]`; an earlier revision said [100, 160] for members 100-339, because
  `merge-pool-tda` then copied its first part's stamp.

**EK-FAC** (`tauq_ekfac_e30`, `pool_tda_ekfac`): kronfluence 1.0.1 at the final checkpoint of
each of 30 full-set runs (seed base 30000, the runs SOURCE reads), damping 1e-8, plain mean;
protocol stamp `full-set-trajectory-finals`; corrected Spearman LDS 0.2352. It predates the
CLI's EK-FAC flow: no in-repo builder reproduces its provenance (`tests/test_provenance.py`).

**SOURCE** (`tauq_source_L3_e30`, `pool_tda_source_L3`): L = 3 over the checkpoints at epochs
4, 8, ..., 24 of the same runs, plain mean, kronfluence 1.0.1; `lr_times_steps` =
[2576.8421052631584, 2476.842105263159, 826.3157894736844], pinned by
`test_released_cifar10_lr_k_is_unchanged`; corrected Spearman LDS 0.2359.

## qnli

**Pool.** 8,000 bert-base-cased fine-tunes on α = 0.5 subsets of 50,000 QNLI training examples
(20 seeds × 400 masks, 2,000 validation queries; TRAK App. A.1's fine-tune: 20 epochs, SGD at
LR 1e-3, batch 8), in bf16 autocast where the paper uses fp32. The tokenized data is pinned
under transformers 4.36.2.
- Built in August as 10 seeds × 200 masks at `e95b405` on RTX 5090s (torch 2.8.0+cu128),
  sharded; an independent re-assembly of its 288 shard pieces reproduced it bit for bit.
- Grown between 2026-10-17 and 10-25 in four merged, checked stages (M = 300 and 400 at R = 10,
  then R = 15 and 20) on two 8× RTX 5090 machines, new cells trained at `7a1a04f`. The
  prefix-stable `sample_subsets(400, n, 0.5, 777)` keeps the old masks as its first 200 rows;
  each merge checked every earlier cell bit-identical. Caveat: the stamp,
  `corrected-lds-7a1a04f`, names only the last writer; 2,000 cells were trained at `e95b405`.

**TRAK** (`tauq_trak5_e{2,4,10,20}`, `tauq_trak1_e{10,20,50,100}`, `pool_tda_trak5` with 20
units of 5, `pool_tda_trak1` with 100 slices): 100 members on α = 0.5 subsets (seeds
10000-10099), traker 0.3.2's fast_jl CUDA projector at k = 4,096. The roster grew from 20
members (2026-08-30) to 50 (2026-10-09; the released 20 reproduced with max |diff| 0) and 100
in the pool_tdas (2026-10-12, appended bit-identically). For the 400 masks, members 50-99 were
retrained and all 100 re-featurized on RTX 5090s: the six tauqs released before came out
bit-identical, and the slices matched to 1.4e-14 on the old masks. `tauq_trak1_e100` and
`tauq_trak5_e20` are new; `tauq_trak1_e{10,20}` and `tauq_trak5_e{2,4}` are the August files,
kept byte for byte. Their provenance spells `method: trak1` / `trak5`, later files
`method: trak` plus `e_features`; `evaluate` resolves both alike.

**EK-FAC** (`tauq_ekfac_e20`, `pool_tda_ekfac`): kronfluence 1.0.1 at the final checkpoints of
20 full-set runs (seeds 30000-30019, the TracIn runs), damping 1e-8, plain mean. Recomputed for
the 400 masks with `compute-tau --method ekfac` (the flow merged at `693841c`): runs 0-15, scored
on RTX 5090s, match the August file to 1.6e-15; runs 16-19, scored on an RTX 4090, to ≤ 1.7e-3
relative (correlation 1.000000, naive LDS 0.4910 for both). The protocol stamp changed from
`full-set-trajectory-finals` to `full-set-final-checkpoint`: the same protocol.

**TracIn** (`tauq_tracin_e20`, `pool_tda_tracin`): TracInCP over the same 20 runs, checkpoints
at epochs 4, 8, 12, 16 and 20, schedule-derived weights, k = 1,024. For the 400 masks the runs
were retrained and rescored on RTX 5090s: the tauq came out bit-identical, and the slices
matched to 7e-15. Its LDS is low, but its per-query Spearman against EK-FAC is 0.78.

**Measured** on the 200 masks × 10 seeds before the growth (`corrected_rho_inf`): `trak5_e4`
0.6113 (SE 0.0018), `trak5_e10` 0.6630 (SE 0.0019), `trak1_e20` 0.6105, `trak1_e50` 0.6626;
EK-FAC (the August file) 0.5155 and TracIn 0.1669 at 20 models. At 20 units TRAK-5's corrected
Spearman LDS was 0.668; the 1/E law forecast 0.670 from two.

## mnist

**Pool.** 12,000 retrains of a 512/256/128 MLP on α = 0.5 subsets of n = 6,144 MNIST images
(about 10%; 40 seeds × 300 masks, 2,000 queries; SOURCE App. B.1: 20 epochs of SGD with
momentum at a constant LR 3e-2, batch 64, weight decay 1e-3). Stamped `corrected-lds-0.1.0`.

**Release scope.** Trimmed on 2026-10-25 to the four files the paper's Figure 9 reads. The
bank's other 14 attribution files (every `pool_tda_*` and the smaller-E tauqs, TracIn included)
left the manifest then and the Hub on 2026-10-27.
- `tauq_trak1_e100`: TRAK, 100 members, `e_features` 1, k = 2,048, traker 0.3.2;
  `corrected_rho_inf` **0.6447** (SE 0.0011; naive 0.5454), no floored queries, no tied pairs.
- `tauq_source_L3_e30`: SOURCE, L = 3, plain mean over 30 full-set runs (seed base 30000), at
  App. B.1's pinned checkpoints (3, 6, 10, 13, 16, 20), whose segments span 6, 7 and 7 epochs
  (see Findings): `lr_times_steps` = [172.8, 201.6, 201.6], 28.8 per epoch at the constant
  3e-2, 96 steps per epoch, momentum 0.9. The preset (once 5 segments, a checkpoint every 4
  epochs) was corrected to this run and reproduces its full provenance exactly
  (`test_mnist_preset_matches_the_released_bank`).
- `tauq_ekfac_e30`: EK-FAC at the final checkpoint (epoch 20) of the same 30 runs, damping
  1e-8, kronfluence 1.0.1. Its provenance uses older protocol and aggregation names, from code
  never committed here (`tests/test_provenance.py`).
- Not yet verified: these checks cover identity and metadata. Nobody has re-scored mnist from
  its checkpoints and compared the tau matrices with the released arrays.

## imagenet

Pool only: 500 retrains of a BlurPool ResNet-18 on α = 0.5 subsets of the ImageNet-1k training
set (n = 1,281,167; 5 seeds × 100 masks, 2,000 validation queries), with TRAK App. A.1's
ImageNet knobs completed by the ffcv-imagenet rn18 config, a reconstruction since the paper
cites no training code (15 epochs, progressive resolution 160 → 192, peak LR 5.2). Built from
the license-gated `ILSVRC/imagenet-1k` parquet at revision `49e2ee26`; the bank holds derived
margins, masks and indices, no images. The pinned data hashes come from a full build of that
revision decoded with Pillow 12.3. Stamped `corrected-lds-source-798d4cf-dirty`: commit
`798d4cf` plus uncommitted changes. There is no build record for the pool beyond its stamps.

## wikitext2_gpt2_bae

A companion bank, not part of the retrain series above: its counterfactual differs, and mixing
it with the other banks produces wrong numbers. EleutherAI's bergson harness, at `0ad2e5b`,
fine-tuned GPT-2 (124M) on WikiText-2 in the Bae et al. 2024 setup as packaged in bergson's
`examples/replicate_bae_approx_unrolling_source/` (App. B.1: AdamW at a constant LR 3e-5, batch
8, 3 epochs, dropout on in training, α = 0.5). It is not the setup of bergson's own paper
(Table 2, EK-FAC 0.318), hence `bae`. Data: `EleutherAI/bergson-wikitext-2-4656-chunks` at
revision `971da0e6`, 4,656 training and 481 validation blocks of 512 tokens.

**Pool.** A mask zeroes the excluded documents' loss weights instead of retraining on the
subset, so batching and data order are identical across masks.
- bergson seeds once per replica (seeds 1004-1023), so the pool is `shared-per-replica`: the
  per-replica offset cancels across masks, and `evaluate` uses the centred seed-noise estimator.
- Two readouts of the same models. bergson's in-loop `validate` never calls `.eval()` under
  `train_mode`, so it reads losses with dropout active, while bergson's published number
  re-reads saved models in eval mode. Every cell was retrained with bergson, saved and read in
  eval mode: `pool.npz` (`readout="eval"`). The in-loop readout of the same runs is
  `pool_dropout_readout.npz` (`readout="dropout"`, stamped `bergson-pool-1`). On the 100 × 10
  bank the median lambda was 0.37 (eval) against 3.86 (dropout); the naive LDS of the two
  differed by 0.07-0.11, and the corrected ones agreed to about 0.01.
- It reproduces the published number: bergson's EK-FAC with Bae's protocol (mean per-query
  Spearman, replicas 0-4, eval readout, 100 × 10 bank) gave 0.4562 against the 0.468 ± 0.015 of
  bergson's example; the dropout readout alone (R = 5) had given 0.3506, a 24 SE gap.
- Grown from 10 × 100 to 20 × 400 between 2026-10-17 and 10-25, both readouts, bergson at
  `0ad2e5b`; masks 0-99 × seeds 1004-1013 equal the 2026-10-07 release bit for bit.
- Each `validate` call trained the full-data baseline and then 13 subsets, so a subset's
  dropout draws depend on its place in the call. Every seed's subsets 0-99, 100-299 and 300-399
  were cut every 13 from their start, the layout `corrected-lds bergson-pool setup` renders.

**Attribution files.** Values at Bae's five seeds (mean per-query Spearman) are on the 100 × 10
bank; every score file's 400 × 20 numbers are in `wikitext2_gpt2_bae/README.md` on the Hub.
- `tauq_ekfac_bergson`: bergson's own EK-FAC on one full-data model (seed 1004, final
  checkpoint), unprojected, damping 1e-8, `lm_head` excluded, all 481 queries in one process:
  a check that bergson's EK-FAC agrees with ours, which it does to ≤ 0.001.
- `tauq_ekfac_e{5,10}`, `pool_tda_ekfac`: our EK-FAC (kronfluence 1.0.1) at the final checkpoint
  of 10 full-set runs trained by bergson (seeds 30000-30009; E = 5 is runs 0-4), damping 1e-8,
  over the 48 attention and MLP projections (not `lm_head`, which is tied to the token
  embedding that Bae et al. exclude). `corrected_rho_inf` at E = 10: **0.4931** (SE 0.0022);
  0.4842 on 100 × 10.
- `tauq_tracin_e10`, `pool_tda_tracin`: our TracInCP on the same runs, six checkpoints one per
  half epoch (steps 291-1746), the 48 projections at k = 16,384 with a fresh projection per
  checkpoint: 0.117 at Bae's five seeds, against ≈ 0.10 in their Fig. 6 (an all-parameter
  version scored 0.055, dominated by GPT-2's tied embedding).
- `tauq_source_L3_e1`: SOURCE from bergson's own scorer (L = 3, the Adam-preconditioned variant
  of Bae et al. App. C) at `0ad2e5b` with bergson #511 (`08e50eff`, cherry-picked as
  `ea4f6941`); one model (train seed 30000), six half-epoch checkpoints, scores negated
  (bergson's are loss-signed). 0.455 at Bae's five seeds (bergson 0.475, Bae ≈ 0.44), +0.0135
  above our EK-FAC on the same model (paired SE 0.0018).
- TRAK (`tauq_trak5_e{2,4,10}`, `tauq_trak1_e{10,20,50}`, `pool_tda_trak5` with 10 units of 5,
  `pool_tda_trak1` with 50 slices): ours, as Bae et al. report no TRAK on WikiText. 50 members
  (seeds 10000-10049) trained by bergson on their own random halves, featurized over every
  parameter (the 48 projections scored lower) at k = 2,048. `trak5_e10`: `corrected_rho_inf`
  **0.2380** (SE 0.0025); 0.2397 on 100 × 10. It rises with members: 0.083 for one, 0.222 for 50.
- For the 400 masks, the EK-FAC and TracIn pool_tdas were re-projected on CPU from the kept
  per-run scores (masks 0-99 exact), and the 50 TRAK members were re-featurized: 0-24 and 26 on
  RTX 5090s (3e-15 from the release), 25 and 27-49 on RTX 4090s with the projector told 170 SMs
  (≤ 3e-3 relative; naive LDS 0.2267 for both). The TRAK files thus mix two GPU models.

## Findings and corrections

### Why every bank was rebuilt (2026-08-15)

The legacy banks, built by an earlier private implementation, cannot be reproduced by any code.
Legacy CIFAR-10 cells rebuilt from today's canonical data correlated (Pearson) about 0.037 with
the stored margins, with the old code and the new alike, while stored cells agreed with each
other at 0.86-0.90. The stored query indices equal `default_rng(0).choice(10000, 2000)` exactly
and the Hub parquet's test split is byte-identical to the original tarball's, yet no query
mapping (drawn order, sorted, first 2,000, value matching) aligned the stored columns with
today's margins, and neither the tarball's training order nor the parquet's reproduces the bank.
The legacy order check, an md5 of the query indices, was order-blind: its value follows from the
RNG draw alone. So every bank here was rebuilt with this pipeline, which hash-verifies the
canonical row order (`presets.DATA_HASHES`) at every build and every load.

The trainer was ported first and certified bit for bit: on 4× RTX 5090 (torch 2.8.0+cu128),
cifar10 cell (0, 0) (seed 50,000, mask row 0 of `sample_subsets(300, 50000, 0.5, 777)`, the
default ResNet-9 recipe), trained once with each implementation on the same hash-verified data,
gave identical margins on all 2,000 queries (battery mean 6.285445046067238 from both).

### EK-FAC: full-set models, damping 1e-8 (2026-08-30)

The first cifar10 EK-FAC files ensembled EK-FAC over TRAK's α = 0.5 subset-trained members at
kronfluence's `0.1 × mean eigenvalue` damping. Influence functions are defined at a model
trained on the data, so about half of each member's scores were for points it never saw. Under
the SOURCE paper's estimand (per-query Spearman against the mean of 20 seeds):

| cifar10 EK-FAC | E = 1 | E = 10 |
|---|---|---|
| subset-trained members (retired) | 0.1611 | 0.2553 |
| full-set models, damping 1e-8 | 0.1266 | 0.2036 |
| Bae et al. 2024, Figure 6 (IF) | 0.1387 | 0.2027 |

The gap, +0.053 at E = 10 (26% relative), grew with E: the signature of the ensembling protocol,
not of the influence estimator; the protocol-matched run lands 0.0009 from the published value.
The subset-trained files were withdrawn: every released EK-FAC file scores full-set models at
their final checkpoint, damping 1e-8, plain mean (TRAK's subset-trained members stay, by design).

### TRAK is one family indexed by `e_features` (2026-08-30)

`trak<e>` names the number of members in one Eq. 15 unit, and `trak1` is the plain mean: Eq. 15
over one member is that member's own estimate. The files were renamed to match (cifar10's
`trak-pm` / `trak` / `trak_f20` to `trak1` / `trak10` / `trak20`, qnli's `trak` / `trak-pm` to
`trak5` / `trak1`) by rewriting each through the writers a fresh build uses, which re-run the
pairing checks; the arrays were verified unchanged. Every tauq now records `e_features`: without
it, `evaluate` had silently dropped qnli's replica-resolved block, looking for a
`pool_tda_trak10.npz` that never existed. The aggregation matters at scale: on cifar10 (100
members, k = 20,480) pooled Eq. 15 beats the plain mean by 0.0865, about 80 SE, as the cov(s, q)
term grows with k/E; on qnli (k = 4,096, soft threshold off, 20 members) the gap was 0.0008.

### TRAK's projection depends on the GPU

traker's `CudaProjector` passes the GPU's streaming-multiprocessor count to fast_jl with the
seed, so a 128-SM RTX 4090 draws a different projection than a 170-SM RTX 5090: a member
featurized on a 4090 correlated 0.13 with the released one, though its naive LDS moved only
+0.002. TRAK scores therefore reproduce only with the SM count that made them. The WikiText-2
members featurized on RTX 4090s for the 400-mask release ran with the projector told 170 SMs
and match the 5090-made files to ≤ 3e-3. The presets now pin that count
(`projector_num_sms: 170` for cifar10, qnli and wikitext2_gpt2_bae; mnist's GPU was not
recorded), and `compute-tau` applies it to TRAK and TracIn on any GPU model.

### EK-FAC's members reproduce less tightly than its LDS (2026-08-24)

40 EK-FAC members retrained from their seeds and rescored gave a tauq that correlates 0.9756
with the original (per-query mean 0.9737, minimum 0.9360), below the ≥ 0.999 at which TRAK
regenerates, while the corrected LDS moved by about 0.4 SE and the pool-side statistics stayed
bit-equal (measured on the since-retired subset-trained cifar10 EK-FAC). EK-FAC compounds
retrained-member atomics, the sampled-label Fisher and eigendecomposition sensitivity: a
property of the method, not a defect of the pipeline. Where runs retrain bit-identically,
scoring is tight: qnli's EK-FAC reproduced to 1.6e-15 on the same GPU model and to ≤ 1.7e-3
relative on an RTX 4090.

### SOURCE segments are read off the checkpoints (2026-09-05)

SOURCE's segment spans and their LR × steps factors (`lr_times_steps`) used to assume evenly
spaced checkpoints; both are now counted from the checkpoints a run wrote. On a uniform grid
this is the old derivation: the released cifar10 `tauq_source_L3_e30`'s `lr_times_steps`
reproduce to 1e-9, and its full provenance string compares equal. It was needed for App. B.1's
six checkpoints over 20 epochs, `floor(20k/6)` = (3, 6, 10, 13, 16, 20), which no stride
expresses; at L = 3 they span 6, 7 and 7 epochs, where the old closed form would have covered 18
of the 20 and dropped the rest. `checkpoint_epochs` enters provenance only when pinned, so
earlier strings stay byte-identical.

### Ensemble forecasts count clipped queries (2026-10-16)

`tda_noise.battery_c` reduces the per-query member-noise ratios c to the battery-level c-hat
behind every ensemble forecast (figure 1(b), figure 2(c, f, i)). A query whose signal-variance
estimate is V_s ≤ 0 is clipped and carries c = +inf. The median used to drop those queries, the
noisiest ones, biasing c-hat and every forecast with clipping low; it now counts them as +inf,
so c-hat = 1 / median_q(V_s / sigma2), not estimable when more than half are clipped. For
CIFAR-10 TRAK-1 anchored on two members (44% of queries clipped) the mean forecast error went
from -0.192 to +0.002; series without clipping (EK-FAC, SOURCE, TracIn, TRAK-5) are unchanged.

### Held-out references are the figures' default (2026-10-16)

No plotted estimate shares data with its reference. `paper/_heldout.py` splits each pool's
seeds into a reference set A (the truth, a fitted law or a plateau) and an estimate set B,
sharing masks, so every target is the LDS on the bank's own subsets; split b is the same in all
figures. Ensemble forecasts anchor on the last K replicas read out with set A and are checked
against the earlier replicas read out with set B. With the law fitted on all seeds, the naive
point at the full seed count equals it by construction (all 9 series agreed to 6 decimals). A
seed count the split cannot afford gets no estimate and no predicted plateau or floor line.

### `evaluate` result schema 3 (2026-10-26)

Results carry `"schema": 3`. The Pearson-space debiasing engines, the pivotal bootstrap and the
replay samplers were removed, as no figure and no released file used them, and with them the
`debiased_analytic*`, `debiased_parametric` and `pivot` fields, the `seed`, `nsim`,
`nsim_refit`, `qsub` and `K` parameters, the `--seed`, `--K`, `--ci` and `--rank-debias` flags,
and `replica_curve`. Every other field is unchanged: on cifar10 `trak1_e100`, wikitext2_gpt2_bae
`ekfac_e10`, qnli `ekfac_e20` and mnist `trak1_e100`, with their pool_tda blocks, the results
matched the schema-2 code leaf for leaf apart from these.

## Figures

`paper/figN.py` builds paper Figure N into `plotdata/figN.json` and `figures/figN.pdf`; a stage
`paper/_name.py` writes `plotdata/name.json`; the other `_*.py` files are libraries. Bank files
per figure and the recompute order are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

| Figure | script | plotdata it builds on (stages) | banks |
|---|---|---|---|
| 1 | `fig1` | `_fig1_panels`, `_rmse_heldout`, `_spearman_debias`, `_rmse_corrected` | cifar10 |
| 2 | `fig2` | `_fig2_panels`; (a, d, g) from the banks | cifar10, qnli, wikitext2_gpt2_bae |
| 3 | `fig3` | none | cifar10, qnli, wikitext2_gpt2_bae |
| 4 | `fig4` | `_spearman_debias` | cifar10 |
| 5 | `fig5` | `_fig5_panels`, `_bridge_bias`, `_bridge_estimators`, `_proxy_sweep` | cifar10, qnli |
| 6 | `fig6` | none | cifar10, qnli, wikitext2_gpt2_bae |
| 7 | `fig7` | `fig6` (its plotdata) | those of Figure 6 |
| 8 | `fig8` | none | cifar10 |
| 9 | `fig9`, also `fig9_noise_corrected.pdf` | none | cifar10, mnist |

The scripts were renamed after the paper's figures on 2026-10-26; every figure rendered
pixel-identically, and the plotdata changed only in name strings. The paper's LaTeX still
includes the PDFs under their old names. Brackets give a figure's old plotdata name where it
differed from its script.

| old script | new script |
|---|---|
| `fig1_lds` | `fig1` (Figure 1) |
| `fig2_rmse` | `fig2` (Figure 2) |
| `fig5_simple` | `fig3` (Figure 3) |
| `fig_spearman_bias_variance` (`spearman_bias_variance`) | `fig4` (Figure 4) |
| `fig5_app` | `fig5` (Figure 5) |
| `fig6_rmse` | `fig6` (Figure 6) |
| `fig6_single_draw` | `fig7` (Figure 7) |
| `fig_spearman_bias_variance_queries` (`spearman_bias_variance_queries`) | `fig8` (Figure 8) |
| `fig_spearman_grid` (`spearman_grid`) | `fig9` (Figure 9) |
| `fig1`, `fig2`, `fig5` | `_fig1_panels`, `_fig2_panels`, `_fig5_panels` |
| `fig3`, `fig2_base` | `_forecast`, `_shared` |
| `fig1_rmse_corrected`, `fig1_rmse_heldout` | `_rmse_corrected`, `_rmse_heldout` |
| `fig1_spearman_debias`, `fig1_lds_ensemble` | `_spearman_debias`, `_lds_ensemble` |
| `fig_assumptions`, `fig_bridge_estimators` | `_assumptions`, `_bridge_estimators` |
| `fig_bridge_bias`, `fig_proxy_sweep` | `_bridge_bias`, `_proxy_sweep` |
