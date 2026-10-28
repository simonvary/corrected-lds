---
license: cc-by-sa-4.0
pretty_name: corrected-lds banks
viewer: false
tags:
- training-data-attribution
- linear-datamodeling-score
- datamodels
- trak
---

# corrected-lds banks

Retrain banks for [`corrected-lds`](https://github.com/simonvary/corrected-lds), which
evaluates training-data attribution (TDA) with a seed-noise-corrected Linear Datamodeling
Score (LDS).

- A **pool** holds retrain readouts, an R seeds × M masks × Q queries array: cell (r, j) is
  the per-query margins of a model retrained on random subset j with seed r.
- A **tauq** is one attribution method's n_train × Q score matrix, and a **pool_tda** holds
  the same method one replica at a time.
- A **bank** is one folder: a pool and the score files that pair with it.

## Layout

```
cifar10/              CIFAR-10, ResNet-9: R=40 × M=300, Q=2,000, n=50,000
├── pool.npz
├── tauq_trak1_e{20,100}.npz, tauq_trak10_e{2,10}.npz        TRAK
├── pool_tda_trak1.npz, pool_tda_trak10.npz, pool_tda_trak20.npz
├── tauq_ekfac_e30.npz, pool_tda_ekfac.npz                   EK-FAC, 30 full-set models
└── tauq_source_L3_e30.npz, pool_tda_source_L3.npz           SOURCE, the same 30 runs

qnli/                 GLUE QNLI, BERT-base: R=20 × M=400, Q=2,000, n=50,000
├── pool.npz
├── tauq_trak1_e{10,20,50,100}.npz, tauq_trak5_e{2,4,10,20}.npz   TRAK, 100 members
├── pool_tda_trak1.npz, pool_tda_trak5.npz
├── tauq_ekfac_e20.npz, pool_tda_ekfac.npz                   EK-FAC, 20 full-set models
└── tauq_tracin_e20.npz, pool_tda_tracin.npz                 TracIn, the same 20 runs

mnist/                MNIST subset, MLP: R=40 × M=300, Q=2,000, n=6,144
├── pool.npz
└── tauq_trak1_e100.npz, tauq_ekfac_e30.npz, tauq_source_L3_e30.npz

imagenet/             ImageNet-1k, ResNet-18: R=5 × M=100, Q=2,000, n=1,281,167
└── pool.npz

wikitext2_gpt2_bae/   WikiText-2, GPT-2: R=20 × M=400, Q=481, n=4,656 (a companion bank)
├── pool.npz                                            eval-mode readout, the default
├── pool_dropout_readout.npz                            the same models, dropout on
├── tauq_ekfac_bergson.npz                              bergson's EK-FAC, one model: a check
├── tauq_ekfac_e{5,10}.npz, pool_tda_ekfac.npz          our EK-FAC, 10 full-set runs: the main one
├── tauq_tracin_e10.npz, pool_tda_tracin.npz            our TracIn, the same runs
├── tauq_source_L3_e1.npz                               SOURCE (L=3, Adam), one model
├── tauq_trak1_e{10,20,50}.npz, tauq_trak5_e{2,4,10}.npz     TRAK, 50 members
└── pool_tda_trak1.npz, pool_tda_trak5.npz
```

## Names and protocols

- `tauq_<method>_e<E>.npz` averages **E replicas**, independent estimates of the scores.
  TRAK forms one replica from `e_features` members (one Eq. 15 unit), so `trak10_e2` is
  2 × 10 = 20 trained models and `trak1_e100` is 100. SOURCE names its segment count
  (`source_L3`).
- `pool_tda_<method>.npz` keeps one slice per replica, so any smaller ensemble is an exact
  average of slices: the whole ensemble-size curve without rescoring.
- TRAK members train on random halves of the data, as TRAK intends. EK-FAC (damping 1e-8,
  final checkpoint), TracIn and SOURCE (six checkpoints in L = 3 segments, Bae et al. 2024)
  score models trained on the full set.
- Recipes: cifar10 and qnli follow TRAK's App. A.1, mnist SOURCE's App. B.1, and imagenet
  a 15-epoch ResNet-18. Every file stamps its recipe, data hashes, index draws and mask draw
  inside, and the loaders refuse a pool and a tauq that do not pair.

## The WikiText-2 companion bank

Built with EleutherAI's [bergson](https://github.com/EleutherAI/bergson) (commit
`0ad2e5b`) in the setup of Bae et al. 2024 (GPT-2 124M, AdamW at a constant 3e-5, batch 8,
3 epochs, dropout on in training). Its numbers do not mix with the other banks':

- A mask zeroes the excluded documents' loss weights rather than retraining on the subset,
  and a replica's masks share one seed. The pool is stamped `shared-per-replica`, and
  `evaluate` uses the matching seed-noise estimator.
- `pool.npz` reads every model in eval mode, as bergson's published number does;
  `pool_dropout_readout.npz` reads the same models with dropout on, as bergson's `validate`
  loop logs them.
- Two EK-FACs: ours, over 10 full-set runs, is the bank's EK-FAC; bergson's own, on one
  model, checks it (they agree to within 0.001) and the replication: on Bae et al.'s
  protocol (5 seeds, the first 100 masks) it scores 0.456 in mean per-query Spearman, against
  0.468 ± 0.015 in bergson's example. Every file's numbers are in
  `wikitext2_gpt2_bae/README.md`.

## Use

These are raw npz files, not a `load_dataset` dataset; read them with the corrected-lds
tooling:

```bash
git clone https://github.com/simonvary/corrected-lds && cd corrected-lds
pip install -e .
corrected-lds fetch-banks              # every file checked against banks-manifest.json
corrected-lds evaluate banks/cifar10 --tauq source_L3_e30
```

`banks-manifest.json` in the code repo pins every file's sha256 at one commit of this
dataset. To score your own method, save its scores with `corrected_lds.write_tauq` and pass
the file to `evaluate`.

## Licence

The banks are released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
They hold margins, masks, indices and scores only, no dataset content, but they derive from
datasets with their own terms: QNLI (from SQuAD, CC BY-SA 4.0), WikiText-2 (CC BY-SA 3.0) and
MNIST (CC BY-SA 3.0) are share-alike, and the ImageNet pool was built from the license-gated
[`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k), so its use is also
subject to the ImageNet terms of access. The code is MIT-licensed.
