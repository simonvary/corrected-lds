# Formats and methods

What is inside each bank file, how the files are named, what `evaluate` reports, and what each
attribution method computes. How the released files were made is in
[`PROVENANCE.md`](../PROVENANCE.md); how to rebuild them is in [`REPRODUCE.md`](REPRODUCE.md).

A **bank** is one directory: a `pool.npz` of retrain readouts plus the attribution files that
pair with it. All files are plain `.npz` archives with every identity field stored inside; the
loaders in `corrected_lds.banks.artifacts` validate them and refuse anything inconsistent.

## The three file kinds

### `pool.npz`: retrain readouts

| key | shape / type | meaning |
|---|---|---|
| `margins` | `(R, M, Q)` float64 | readout of the model trained on mask `j` under seed `r`, for query `q`: the correct-class margin (for the GPT-2 bank, minus the query loss) |
| `subsets` | `(M, n)` int8 | the masks: row `j` has a 1 for every training example kept |
| `M`, `R`, `alpha` | int, int, float | masks, seeds per mask, kept fraction |
| `subsets_source` | str | the sampler that re-derives `subsets` (`sample_subsets:777`) |
| `recipe` | JSON | the frozen training recipe (`to_dict()` of a `presets` recipe) |
| `seed_mode` | str | `nested` (every cell its own seed) or `shared-per-replica` (bergson's protocol) |
| `dataset`, `train_idx`, `query_idx`, `train_hash`, `query_hash` | | which examples were used, and content hashes pinned in `presets.DATA_HASHES` |
| `schema`, `code_version`, `completed_cells`, `done_rows` | | format version (3), the code that built it, build bookkeeping |

The GPT-2 bank also stores `losses` and per-replica `baselines`.

### `tauq_<method>_e<E>.npz`: one attributor's scores

| key | shape / type | meaning |
|---|---|---|
| `tauq` | `(n_train, n_targets)` float32 | score of training example `i` for query `q`; positive means the example helps the query |
| `dataset`, `n_train`, `n_targets`, `train_idx`, `query_idx`, `train_hash`, `query_hash` | | pairing identity, copied from the pool |
| `n_models` | int | models trained for this file (its cost) |
| `proj_dim` | int | random-projection dimension (0 when unprojected) |
| `provenance` | JSON | method, `e_features`, aggregation, recipe, library versions and every knob that changes the scores |

### `pool_tda_<method>.npz`: replica-resolved predictions

The attributor-side mirror of `pool.npz`: one slice per self-contained estimator (a
*replica*), already projected onto the pool's masks.

| key | shape / type | meaning |
|---|---|---|
| `g` | `(R_tda, M, Q)` float64 | replica `k`'s prediction `sum_{i in S_j} tau_k[i, q]` |
| `models_per_replica` | int | models inside one replica (the TRAK `e_features`; 1 for the others) |
| `inner` | str | `separated-avg-eq15` (TRAK unit), `single-member` (one model) or `single-trajectory` (one TracIn run) |
| `method`, `seed_base`, `provenance` | | the estimator and the seeds its replicas start from |
| pairing identity, `M`, `subsets_source`, `subsets_sha16` | | as for a tauq, plus a digest that binds it to the pool's exact masks |
| `schema` | int | format version (1) |

Replicas combine by plain mean, so the average over any subset of them is an exact
smaller-ensemble estimate: the whole LDS-versus-ensemble-size curve, CPU-only, with no
rescoring.

### What the loaders refuse

A pool with missing cells; masks that do not re-derive from the pinned sampler; data hashes
that differ from the pins; a tauq or pool_tda whose pairing identity (examples, order or
masks) differs from the pool's. `write_tauq(path, scores, pool, provenance=...)` copies the
identity from the pool, so a file written for a pool pairs with it by construction.

## Names

```
tauq_<method>_e<E>.npz       one ensemble size
pool_tda_<method>.npz        every replica, so no size in the name
```

`<method>` carries the estimator's own knob when it has one (`trak<e_features>`,
`source_L<segments>`, plain `ekfac` and `tracin`). The provenance records the knob too: TRAK
files stamp `method: trak` with `e_features` beside it (some older ones spell `trak5`), and
SOURCE files stamp `source_L<segments>` with the count also in `source_args`. `evaluate` finds
a tauq's `pool_tda` file from the provenance, never from the file name, and reports a key it
cannot determine rather than guessing. One file breaks the pattern on purpose: WikiText-2's
`tauq_ekfac_bergson.npz`, bergson's own EK-FAC on one model, kept as a check of ours.

**`E` counts replicas, not models.** It is the axis the attributor's 1/E noise law lives on.
The model cost is `E × e_features`, and `n_models` inside the file records it:

| file | E | e_features | models |
|---|---|---|---|
| `tauq_trak10_e10.npz` | 10 | 10 | 100 |
| `tauq_trak10_e2.npz` | 2 | 10 | 20 |
| `tauq_trak5_e4.npz` | 4 | 5 | 20 |
| `tauq_trak1_e100.npz` | 100 | 1 | 100 |
| `tauq_ekfac_e30.npz` | 30 | 1 | 30 |

A `pool_tda` holds `R_tda = n_models / e_features` replicas. Two consequences are easy to get
wrong:

* `e_features`, `E` and `R_tda` are three different numbers: `pool_tda_trak5` on qnli has 20
  replicas covering 100 models.
* The TRAK plain mean is `e_features = 1` (`trak1`): Eq. 15 over a unit of one member is that
  member's own estimate.

## What `evaluate` reports

`corrected-lds evaluate <bank> --tauq <name>` and `evaluate_pool(pool, tau)` return one JSON
object (`"schema": 3`):

| block | contents |
|---|---|
| `pool`, `tauq` | the identities of the two inputs (hashes, sizes, recipe, provenance) |
| `params` | `clip`, `level` (interval coverage) and `soft_threshold_k` |
| `results` | the estimates, below |
| `diagnostics` | checks of the estimator's assumptions, below |
| `pool_tda` | added by the CLI when the matching `pool_tda_<method>.npz` sits next to the pool: the same results per replica subset |

| `results` field | meaning |
|---|---|
| `debiased_noise_corrected_spearman` | **the paper's Spearman LDS**: finite-mask debiased, then corrected for retraining noise |
| `corrected_rho_inf` | its Pearson form: the noise-free LDS `rho_inf`, with `corrected_se_jackknife` and `corrected_ci_jackknife` (delete-one-mask jackknife) |
| `spearman`, `debiased_spearman` | the plain Spearman LDS of the R-seed mean, and its finite-mask debiased version |
| `naive_rho1` | the plain single-seed Pearson LDS |
| `bridged_rho_inf`, `bridged_rho1` | the Pearson values mapped to Spearman by `(6/pi) arcsin(r/2)` |
| `reconstructed_rho_R` | the LDS a study with R seeds per mask would measure, for R in 2, 5, 10 |
| `sigma2_mean`, `lambda_median`, `floored_frac` | seed-noise variance, noise-to-signal ratio, and the fraction of queries whose correction hit the clip floor |

The diagnostics test the assumptions the correction makes: `corr_g_epsdiff_mean` (seed noise
independent of the prediction), the bridge residuals and the latent-shape terms (joint
normality), `kurtosis_y1_mean`, and `rank_tied_pair_frac` (ties in the Spearman pairs). A
shared-seed pool also reports `seed_offset_share_mean`.

## The estimators

**Corrected LDS.** The LDS correlates a prediction `g_q(S) = sum_{i in S} tau_q[i]` with a
retrain readout `y_q(S) = F_q(S) + eps` across masks. Seed noise `eps` leaves the covariance
unbiased but inflates `Var(y)`, so plain correlations converge to
`rho_R = rho_inf / sqrt(1 + lambda/R)`, below the noise-free `rho_inf` however many masks are
used. Paired retrains estimate the noise variance, `sigma2 = mean (y1 - y2)^2 / 2`, which the
corrected estimator removes from the denominator (`corrected_lds.estimator`). The only
assumption is that the noise is uncorrelated with the prediction.

**Spearman LDS.** The U-statistic removes the finite-mask bias of the ordinary Spearman
correlation exactly (`corrected_lds.rank_debias`); the retraining-noise factor measured in
Pearson space then scales it, through the arcsin bridge.

**Seed modes.** A `nested` pool seeds every cell independently. bergson's
`shared-per-replica` pools share one seed across a replica's masks; the resulting offset
cancels in correlations across masks, so `evaluate` measures the noise with the centred
estimator instead.

**The attributor's own noise.** An ensemble of E members has prediction noise too: its LDS
follows `rho(E) = rho(∞) / sqrt(1 + c/E)` (`corrected_lds.tda_noise`), so a `pool_tda` file's
replicas give both the curve and its limit.

## Attribution methods

| method | what it computes | protocol in the released banks |
|---|---|---|
| **TRAK** (Park et al. 2023), `tda/trak.py` on `traker==0.3.2` | projected-gradient kernel scores, Eq. 15: the separated average of a unit of `e_features` members | members trained on random α = 0.5 halves; projection `proj_dim` per preset |
| **EK-FAC influence**, `tda/ekfac.py` on `kronfluence==1.0.1` | inverse-Hessian-vector influence with EK-FAC curvature, damping 1e-8 | one estimate per model trained on the full set, at its final checkpoint; plain mean over models |
| **TracInCP**, `tda/tracin.py` | Eq. 1: learning-rate-weighted gradient dot products summed over checkpoints, projected | the same full-set trajectories, one replica per run |
| **SOURCE** (Bae et al. 2024), `tda/source.py` | approximate unrolled differentiation over L segments of the trajectory, on kronfluence's EK-FAC factors | L = 3 segments over 6 checkpoints per run; plain mean over runs |

TRAK's members see half the data each; EK-FAC, TracIn and SOURCE score models trained on all
of it, so a TRAK model and an EK-FAC model differ in training cost as well as in method. The
GPT-2 bank's SOURCE file comes from bergson's own scorer (its Adam variant); see
`PROVENANCE.md`.
