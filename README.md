# corrected-lds

Seed-noise-corrected **Linear Datamodeling Score (LDS)** evaluation for training-data
attribution (TDA).

The LDS scores a TDA method by correlating its predictions with retrained-model outcomes
across random training subsets. Retraining is seed-noisy, so the plain LDS is attenuated:
biased low by a floor that more subsets never remove, and substantially so in practice (a
single-retrain ground truth understates the noise-free LDS by ~37% in our CIFAR-10 pilot).
This repository estimates the seed-noise variance from a few paired retrains and removes it,
debiases the Spearman LDS for the finite number of masks, and ships the retrain pools that
make all of it measurable.

Three nouns name the artifacts: a **pool** is an (R seeds × M masks × Q queries) array of
retrain readouts, a **tauq** is one TDA method's (n_train × Q) score matrix, and a **bank** is
a released directory bundling one pool with its tauqs.

## Score a bank

```bash
git clone https://github.com/simonvary/corrected-lds && cd corrected-lds
pip install -e .                    # CPU only: numpy and scipy
corrected-lds fetch-banks           # download the released banks, sha256-verified
corrected-lds evaluate banks/cifar10 --tauq trak10_e10 --json result.json
```

In the result, `debiased_noise_corrected_spearman` is the paper's Spearman LDS and
`corrected_rho_inf` its Pearson form. The result also carries the naive values, a jackknife
interval, the LDS at every smaller ensemble size, and checks of the correction's assumptions
([`docs/FORMATS.md`](docs/FORMATS.md)).

**Evaluate your own TDA method** against a released bank, without a GPU:

```python
from corrected_lds import load_pool, load_tauq, write_tauq, evaluate_pool

pool = load_pool("banks/cifar10/pool.npz")
scores = ...                        # (pool.n, pool.Q), computed with any code
write_tauq("banks/cifar10/tauq_mymethod.npz", scores, pool,
           provenance={"method": "mymethod"})
result = evaluate_pool(pool, load_tauq("banks/cifar10/tauq_mymethod.npz"))
```

`write_tauq` copies the pool's pairing identity (data hashes, row indices), so the file pairs
with the pool by construction; a mismatched pairing is refused.

## The banks

| bank | pool: R × M, Q queries | attributors | setup |
|---|---|---|---|
| `cifar10` | 40 × 300, 2,000 | TRAK, EK-FAC, SOURCE | CIFAR-10 / ResNet-9 |
| `qnli` | 20 × 400, 2,000 | TRAK, EK-FAC, TracIn | GLUE QNLI / BERT |
| `mnist` | 40 × 300, 2,000 | TRAK, EK-FAC, SOURCE | MNIST subset / MLP |
| `wikitext2_gpt2_bae` | 20 × 400, 481 | TRAK, EK-FAC, TracIn, SOURCE | WikiText-2 / GPT-2, trained by bergson |
| `imagenet` | 5 × 100, 2,000 | none yet | ImageNet-1k / ResNet-18 |

The banks are on the Hugging Face Hub (`corrected-lds/data`), and `banks-manifest.json` pins
every file by sha256 and commit. The [dataset card](docs/hf-dataset-card.md) describes each
bank's files.

## Documentation

- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): scoring the banks, rebuilding the paper's figures,
  rebuilding banks on GPU.
- [`docs/FORMATS.md`](docs/FORMATS.md): what each file holds, how files are named, what
  `evaluate` reports, and the attribution methods.
- [`PROVENANCE.md`](PROVENANCE.md): how each released file was made.

## Code layout

```
corrected_lds/
├── banks/                 what a bank is, and loading and checking one (no torch)
│   ├── presets.py           frozen recipes, experiment presets, dataset content-hash pins
│   ├── masks.py             the pinned subset-mask sampler
│   ├── artifacts.py         load and validate pool, tauq and pool_tda files; pairing checks
│   ├── compare.py           a rebuilt file against the released one
│   └── hub.py               verified download, and upload to a Hub dataset repo
├── pools/                 GPU: builds a bank's pool
│   ├── runner.py            cell seeding, resume and extend, shards and merge
│   ├── train.py             the reproducible SGD loop and margin readout
│   ├── datasets.py          canonical dataset builds, hash-checked on every load
│   ├── bergson.py           the bridge to bergson, which trains the GPT-2 bank
│   ├── bergson_source.py    bergson's SOURCE scorer over many runs
│   └── tasks/               one module per setup
├── tda/                   GPU: builds a bank's attribution files
│   ├── trak.py              ensembled TRAK via traker
│   ├── ekfac.py             EK-FAC influence via kronfluence
│   ├── tracin.py            TracInCP over checkpoints
│   └── source.py            SOURCE on kronfluence's EK-FAC factors
├── estimator.py           the corrected estimator: seed-noise variance out of the denominator
├── rank_debias.py         the finite-mask Spearman U-statistic
├── tda_noise.py           the attributor's own ensemble noise (the 1/E law)
├── evaluate.py            one bank → the scored JSON result
└── cli.py                 the corrected-lds command
paper/                     the nine figure scripts (figN.py), their stages, plotdata and figures
slides/                    the talk's figure scripts, plotdata and figures, built as paper/'s
```

`pools/` and `tda/` spend GPU time once to fill a bank; `banks/` defines a bank's identity and
refuses malformed or mismatched files; the statistics read a loaded bank on CPU with numpy and
scipy only (a torch-free CI job enforces that).

## Citation

If you use this code or the banks, please cite the paper, *Efficient and Accurate Causal
Validation of Training Data Attribution* (ATTRIB workshop, NeurIPS 2026);
[`CITATION.cff`](CITATION.cff) has the entry.

The code is MIT-licensed, with adapted third-party parts listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the banks are CC BY-SA 4.0 (see the
[dataset card](docs/hf-dataset-card.md)).
