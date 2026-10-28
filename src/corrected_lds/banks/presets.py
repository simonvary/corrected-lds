"""Frozen training recipes, presets, and dataset content-hash pins.

Torch-free, so the CPU-only install can import a bank's identity: a recipe's ``to_dict()`` is
the pool identity stamped into every ``pool.npz``.  The content hashes (sha256/16 of the raw
array bytes) pin the canonical row order, which an order-blind digest cannot.
"""

import hashlib
from dataclasses import asdict, dataclass

__all__ = [
    "DATA_HASHES",
    "GPT2_TRACKED",
    "MNIST_CKPT_EPOCHS",
    "PRESETS",
    "BertRecipe",
    "Gpt2WikitextBaeRecipe",
    "MnistRecipe",
    "Resnet9Recipe",
    "Resnet18Recipe",
    "bank_hashes",
]


# -- dataset content-hash pins — sha256/16 of array bytes ------------------- #
# cifar10: raw uint8 images and int64 labels in the canonical HF-parquet row order.  qnli:
# every array of the tokenized cache (bert-base-cased @ seq 128 under transformers==4.36.2),
# so a cache rebuilt with a drifted tokenizer cannot pass on labels alone.
DATA_HASHES = {
    "cifar10": {"Xtr": "28543e0281cbc3c1", "ytr": "d6d2fe5c7528f766",
                "Xte": "177777c33ba79825", "yte": "cbb7365de8ed11f0"},
    # ylecun/mnist parquet (no revision pin: these hashes pin it), canonical row order;
    # images are hashed as uint8 and normalized only after the train/query draws.
    "mnist": {"Xtr": "741c988805d008ac", "ytr": "e474323ac02fc156",
              "Xte": "6d87418db22cc802", "yte": "2fd8de6bdb8d1178"},
    "qnli": {"train_ids": "3047c9ae92673801", "train_tok": "8197fbc2d8e6fd74",
             "train_att": "db364f11a78e011c", "train_y": "baa48c276cd7fcf2",
             "val_ids": "cc1b9d85eee9bb77", "val_tok": "d0541f1eff5ef244",
             "val_att": "0de446c4a87f3b87", "val_y": "e4a834d06b407239"},
    # From a full build of the pinned parquet revision with Pillow 12.3.  build-data checks
    # the full blob hashes; the cheaper gates are checked on every load.
    "imagenet": {
        "train_blob": "a9eb6e040cac10f5",
        "train_blob_gate": "d6ac5667d4009499",
        "train_manifest": "7ec3417419952523",
        "train_y": "6c3754c2a0376a8c",
        "val_blob": "e5614025f06cd001",
        "val_blob_gate": "4200c31c1d3e6d4f",
        "val_manifest": "512cea7700ec683a",
        "val_y": "8771f37f189dd18f",
    },
    # int64 input_ids of EleutherAI/bergson-wikitext-2-4656-chunks (wikitext-2-raw-v1, GPT-2
    # tokenizer, HF run_clm 512-token blocks: 4656 train / 481 validation) at revision
    # 971da0e6da181b9e147081087f1d3fd51669024c, which bergson also reads (data.data_kwargs).
    "wikitext2": {"train_ids": "5a8977cd5594b775", "val_ids": "54123a820f561677"},
}

# which pins constitute each side of a bank's data identity
_SIDE_KEYS = {
    "cifar10": (("Xtr", "ytr"), ("Xte", "yte")),
    "mnist": (("Xtr", "ytr"), ("Xte", "yte")),
    "qnli": (("train_ids", "train_tok", "train_att", "train_y"),
             ("val_ids", "val_tok", "val_att", "val_y")),
    "imagenet": (("train_y", "train_manifest", "train_blob"),
                 ("val_y", "val_manifest", "val_blob")),
    "wikitext2": (("train_ids",), ("val_ids",)),
}


# -- MNIST recipe (SOURCE App. B.1) ----------------------------------------- #
# App. B.1's six checkpoints over 20 epochs, ``floor(20k/6)``, written out because no uniform
# stride gives six in 20 epochs.  ``tracin.resolve_ckpt_epochs`` scores this subset of the
# saved epochs; ``source.segment_epochs`` reads the ragged segments (6, 7, 7 epochs) off the
# checkpoints.  These built the released bank (`test_mnist_preset_matches_the_released_bank`).
MNIST_CKPT_EPOCHS = (3, 6, 10, 13, 16, 20)


@dataclass(frozen=True)
class MnistRecipe:
    """The SOURCE paper's MNIST setup: a three-hidden-layer MLP on ~10% of MNIST, SGDm, batch
    64, a constant 3e-2 learning rate and weight decay 1e-3 for 20 epochs.  ``hidden``
    resolves the paper's ~560k parameter count to 512/256/128."""

    hidden: tuple[int, int, int] = (512, 256, 128)
    epochs: int = 20
    batch: int = 64
    peak_lr: float = 3e-2
    momentum: float = 0.9
    weight_decay: float = 1e-3
    label_smoothing: float = 0.0
    alpha: float = 0.5

    def __post_init__(self):
        if len(self.hidden) != 3 or any(int(w) < 1 for w in self.hidden):
            raise ValueError(f"hidden must contain three positive widths, got {self.hidden!r}")

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)       # stable across a JSON round trip
        return d


def bank_hashes(dataset):
    """The ``(train_hash, query_hash)`` pair stamped into every pool and tauq: one sha256/16
    digest per side over all of that side's pins (images/tokens and labels).  With the
    stamped ``train_idx``/``query_idx`` it identifies the exact ordered tensors used."""
    h = DATA_HASHES[dataset]
    tr, qy = _SIDE_KEYS[dataset]

    def digest(keys):
        return hashlib.sha256("|".join(h[k] for k in keys).encode()).hexdigest()[:16]

    return digest(tr), digest(qy)


# -- image recipe (TRAK App. A.1 + released FFCV code) ---------------------- #
@dataclass(frozen=True)
class Resnet9Recipe:
    """The TRAK CIFAR-10 recipe, pinned for reproducibility (App. A.1 + released code)."""

    epochs: int = 24
    batch: int = 512
    peak_lr: float = 0.5
    peak_epoch: int = 5
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0     # released TRAK CIFAR recipe / quickstart
    alpha: float = 0.5               # subset fraction for the LDS ground-truth retrains
    augment: bool = True
    per_sample_aug: bool = True      # per-image crop offset (FFCV-style) vs one per batch
    pad: int = 2
    cutout: int = 4
    width: float = 1.0

    # FFCV constants, not knobs (``pools/tasks/cifar_resnet9.py``): translate fill 0
    # (= pixel-space mean fill post-normalization), cutout origin uniform in [0, H-cutout]
    # (fully inside), and BatchNorm.  An ablation adds a field together with its implementation.

    def to_dict(self):
        return asdict(self)


# -- text recipe (QNLI / BERT) — bf16 default, fp32 = paper-exact ----------- #
@dataclass(frozen=True)
class BertRecipe:
    """QNLI/BERT finetune recipe: TRAK App. A.1 pins SGD, LR 1e-3, 20 epochs, remove-tanh and
    the 50k subset; the rest are HuggingFace ``run_glue.py`` defaults (linear decay, no warmup),
    measured to matter as a unit: vary them deliberately, not by halves.  One deviation:
    ``precision`` defaults to ``"bf16"`` autocast (~4x cheaper per cell; the disattenuation
    result is recipe-agnostic); ``"fp32"`` is paper-exact.  TRAK knobs live in the preset."""

    model: str = "bert-base-cased"
    optimizer: str = "sgd"
    peak_lr: float = 1e-3
    epochs: int = 20
    momentum: float = 0.0
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    seq_len: int = 128
    batch: int = 8
    eval_batch: int = 128            # batched query readout (avoid OOM on ~2000 x seq_len 128)
    alpha: float = 0.5               # subset fraction for the LDS ground-truth retrains
    remove_pooler_tanh: bool = True  # TRAK linearization (paper App. A.1)
    precision: str = "bf16"          # finetune autocast dtype ("bf16" | "fp32" = autocast off)
    max_grad_norm: float = 1.0       # HF run_glue default grad clip; 0 disables

    def __post_init__(self):
        if self.optimizer != "sgd":
            raise ValueError(f"only optimizer='sgd' is implemented (paper pin; got "
                             f"{self.optimizer!r}).")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be 'fp32' or 'bf16' (got {self.precision!r}); "
                             f"anything else would silently select fp16 autocast.")

    def to_dict(self):
        return asdict(self)


# -- imagenet recipe (TRAK App. A.1 + ffcv-imagenet rn18 configs) ----------- #
@dataclass(frozen=True)
class Resnet18Recipe:
    """The TRAK ImageNet recipe: App. A.1's six stated knobs plus the ffcv-imagenet rn18
    config they parameterize (a reconstruction: the paper cites no ImageNet training code).
    The one interpreted knob, ``peak_lr``, reads the paper's "initial learning rate 5.2" as
    ffcv's ``lr.lr``, the cycle peak, as A.1 calls the CIFAR value "(max) learning rate".
    Task-module constants, not knobs: RandomResizedCrop scale (0.08, 1.0) and ratio (3/4, 4/3),
    the cyclic LR knots, BlurPool on every stride>1 conv with in_channels >= 16, resolution
    rounded to /32 (epochs 0-11 @160, 12-14 @192), and weight decay only for params whose name
    lacks "bn" (the released split, replicated: ``downsample.1`` BNs do decay)."""

    arch: str = "resnet18"
    use_blurpool: bool = True        # the paper's "modified" ResNet-18
    epochs: int = 15
    batch: int = 1024
    peak_lr: float = 5.2
    peak_epoch: int = 2
    lr_start_frac: float = 1e-4      # the cyclic schedule starts at lr_start_frac * peak_lr
    momentum: float = 0.8
    weight_decay: float = 4e-5       # bn-name-exempt (see class docstring)
    label_smoothing: float = 0.05
    alpha: float = 0.5               # subset fraction for the LDS ground-truth retrains
    augment: bool = True             # RandomResizedCrop + horizontal flip
    min_res: int = 160               # progressive train resolution ...
    max_res: int = 192
    start_ramp: int = 11             # ... interpolated between these epochs, rounded /32
    end_ramp: int = 13
    val_res: int = 256               # query decode: crop_ratio * short side -> val_res^2
    crop_ratio: float = 224 / 256

    def __post_init__(self):
        if self.arch != "resnet18":
            raise ValueError(f"only arch='resnet18' is implemented (the paper's model "
                             f"family pin; got {self.arch!r}).")
        if not 0 < self.peak_epoch < self.epochs:
            raise ValueError(f"need 0 < peak_epoch < epochs (got peak_epoch="
                             f"{self.peak_epoch}, epochs={self.epochs}): the cyclic LR "
                             f"knots [0, peak, epochs] must be strictly increasing, or "
                             f"np.interp silently returns nonsense.")
        if self.min_res > self.max_res:
            raise ValueError(f"min_res={self.min_res} > max_res={self.max_res}.")
        if self.start_ramp > self.end_ramp:
            raise ValueError(f"resolution ramp start_ramp={self.start_ramp} > "
                             f"end_ramp={self.end_ramp}.")

    def to_dict(self):
        return asdict(self)


# -- LM recipe (GPT-2 / WikiText-2) — bergson's harness, not our pools/runner --- #
@dataclass(frozen=True)
class Gpt2WikitextBaeRecipe:
    """GPT-2 / WikiText-2 finetune, the Bae et al. 2024 (App. B.1) recipe.

    Transcribed from bergson ``examples/replicate_bae_approx_unrolling_source/
    wikitext_gpt2_retrain.yaml`` @ commit 0ad2e5b, whose README reports EK-FAC IF =
    0.468 +- 0.015 over 481 queries with query losses averaged over five retrain seeds.
    Unlike our other recipes, a mask zeroes the excluded documents' loss weights
    (``subset_weight = 0.0``) instead of retraining on the subset, so batching and data order
    match across masks; and bergson's runner seeds once per replica, so the pool is
    ``seed_mode="shared-per-replica"`` and needs ``estimator.seed_noise_variance_centred``.
    Named ``bae`` because bergson's own paper Table 2 is a second, incomparable setup
    (alpha=0.99, dropout off, 24x larger batch; 0.318 vs 0.468): never name either one
    ``wikitext2_gpt2``."""

    model: str = "gpt2"
    optimizer: str = "adamw"
    peak_lr: float = 3e-5
    lr_schedule: str = "constant"
    epochs: int = 3
    batch: int = 8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    eps_root: float = 0.0
    weight_decay: float = 0.01
    train_mode: bool = True          # dropout on
    precision: str = "fp32"
    seq_len: int = 512
    alpha: float = 0.5               # fraction kept (bergson's subset_fraction = removed)
    subset_weight: float = 0.0       # loss-weight zeroing, not subset retraining
    harness: str = "bergson"
    bergson_commit: str = "0ad2e5b0ce09207d2af0e80383fe2d952dd8e747"
    dataset_revision: str = "971da0e6da181b9e147081087f1d3fd51669024c"

    def to_dict(self):
        return asdict(self)


# The 48 modules EK-FAC tracks on GPT-2, spelled out so provenance records them: the four
# projections of each of the 12 blocks as nn.Linear after the Conv1D swap, under the ``lm.``
# wrapper.  Not lm_head: it is tied to the token embedding, which Bae et al. exclude (App. C).
GPT2_TRACKED = tuple(f"lm.transformer.h.{i}.{m}" for i in range(12)
                     for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"))


# -- frozen presets — the identities of the banks --------------------------- #
# Pool builders read only dataset/n_train/n_targets/recipe; the method blocks configure the TDA
# stage.  proj_dim must be a multiple of 512 (fast_jl).  ``ensemble_sizes`` are the released
# n_models (the paper's M).  trak's ``e_features`` members form one Eq. 15 separated average
# (10 on cifar10, mnist and imagenet, 5 on the others), and E = n_models / e_features
# (divisibility enforced) counts plain-mean replicas, as for ekfac/tracin (e_features = 1).
# tracin: proj_dim 1024 = the TRAK paper's 500-1000 baseline range rounded up to fast_jl's
# granularity; ckpt_every keeps every c-th epoch checkpoint (paper Eq. 1 wants c=1; thinned
# where T would explode, per TRAK App. A.3); batch bounds grad chunks, not identity.
# projector_num_sms: fast_jl's projection depends on the GPU's streaming-multiprocessor count,
# so TRAK and TracIn reproduce a release only at the count of the GPU that made it (170, the
# RTX 5090's).  mnist's GPU was not recorded, so it has no pin.
PRESETS = {
    # cheap end-to-end smoke of the image pipeline (small M/R/ensemble); tracin
    # ckpt_every=4 exercises the thinned interval-summed weights, T=6
    "pilot": {"dataset": "cifar10", "n_train": 10_000, "n_targets": 200, "recipe": Resnet9Recipe(),
                  "trak": {"proj_dim": 2_048, "ensemble_sizes": [10],
                            "featurize_augment": True, "e_features": 5},
                  "tracin": {"proj_dim": 1_024, "ensemble_sizes": [10], "ckpt_every": 4,
                              "batch": 256},
                  # SOURCE shares tracin_ckpts/: ckpt_every 4 of 24 epochs = C=6 (the
                  # paper's own count, 3.4), L=3 -> 2 checkpoints per segment.
                  "source": {"segments": 3, "ckpt_every": 4, "ensemble_sizes": [10],
                              "factor_batch": 64, "grad_batch": 64, "query_chunk": 200}},
    # SOURCE App. B.1: ~10% of MNIST, a ~560k-parameter MLP, ten attribution models, six
    # checkpoints over 20 epochs.  Projection sizes are compromises: SOURCE used full TracIn
    # gradients and a 4096-d TRAK projection, while our TRAK noise guard requires d < n/2.
    "mnist": {"dataset": "mnist", "n_train": 6_144, "n_targets": 2_000,
                  "recipe": MnistRecipe(),
                  "trak": {"proj_dim": 2_048, "ensemble_sizes": [10, 20, 100],
                            "featurize_augment": False,
                            "trak_batch": 256, "e_features": 10},
                  "tracin": {"proj_dim": 1_024, "ensemble_sizes": [10, 30],
                              "ckpt_every": 1,
                              "checkpoint_epochs": MNIST_CKPT_EPOCHS,
                              "batch": 256},
                  "ekfac": {"damping": 1e-8, "ensemble_sizes": [30],
                            "factor_batch": 64, "query_batch": 512,
                            "train_batch": 64},
                  # L=3 over the six checkpoints -> 2 per segment.  ckpt_every=1 saves every
                  # epoch; checkpoint_epochs picks the six scored ones (MNIST_CKPT_EPOCHS).
                  "source": {"segments": 3, "ckpt_every": 1,
                              "checkpoint_epochs": MNIST_CKPT_EPOCHS,
                              "ensemble_sizes": [10, 30], "factor_batch": 64,
                              "grad_batch": 64, "query_chunk": 512,
                              "grad_dtype": "float32"}},
    # canonical CIFAR-10 (App. A.1): 50k train, 24 epochs, LR 0.5
    "cifar10": {"dataset": "cifar10", "n_train": 50_000, "n_targets": 2_000, "recipe": Resnet9Recipe(),
                    "projector_num_sms": 170,
                    "trak": {"proj_dim": 20_480, "ensemble_sizes": [20, 100],
                              "require_cuda_projector": True,
                              "featurize_augment": True, "e_features": 10},
                    # the 30 full-set runs EK-FAC and SOURCE score; no cifar10 TracIn file is
                    # released
                    "tracin": {"proj_dim": 1_024, "ensemble_sizes": [30], "ckpt_every": 1,
                                "batch": 256},
                    # Bae et al.'s IF protocol at each run's final checkpoint; ResNet-9 fits
                    # EkfacArgs' default chunking
                    "ekfac": {"damping": 1e-8, "ensemble_sizes": [30]},
                    # SOURCE thins tracin's every-epoch checkpoints to every 4th (C=6, L=3),
                    # needing no new training.  grad_dtype float32 matches float64 to 1.5e-6.
                    # query_chunk 700 peaks near 16 GB; 1000 has the footprint that OOM'd a
                    # 24.6 GB card in eigen_transform (a full-size temporary per module).
                    "source": {"segments": 3, "ckpt_every": 4, "ensemble_sizes": [30],
                                "factor_batch": 32, "grad_batch": 32, "query_chunk": 700,
                                "grad_dtype": "float32"}},
    # QNLI / BERT, bf16 autocast (precision="fp32" is paper-exact).  The CUDA projector is
    # required: BasicProjector is slow, and its block matrix alone is ~40 GB at BERT grad_dim.
    # trak_batch=8 bounds featurization memory (~3.5 GB vs ~14 GB at 32; tauq is invariant).
    # tracin ckpt_every=4 (T=5 of 20): checkpoints are ~0.4 GB and featurize scales with E*T.
    "qnli": {"dataset": "qnli", "n_train": 50_000, "n_targets": 2_000, "recipe": BertRecipe(),
                 "projector_num_sms": 170,
                 # 100 members take trak5 to E=20; the 10/20/50 snapshots write the released
                 # smaller tauqs
                 "trak": {"proj_dim": 4_096, "ensemble_sizes": [10, 20, 50, 100],
                           "require_cuda_projector": True, "trak_batch": 8,
                           "use_half_precision": False, "e_features": 5},
                 "tracin": {"proj_dim": 1_024, "ensemble_sizes": [20], "ckpt_every": 4,
                             "batch": 8, "require_cuda_projector": True},
                 # EK-FAC chunks measured on a 24 GB RTX 4090: bert-base's tracked grad_dim is
                 # 85,609,730, so scoring peaks at ~1.2 + 0.319*query_batch GB (64 fits, 80 OOMs).
                 # damping 1e-8: Bae et al.'s IF value, not kronfluence's 0.1*mean-eigenvalue rule.
                 "ekfac": {"damping": 1e-8, "ensemble_sizes": [20], "factor_batch": 8,
                            "query_batch": 64, "train_batch": 16}},
    # bergson's Bae-replication setup, built by bergson's runner (pools/bergson.py, hence
    # "harness"); n_targets is the full 481-doc validation split behind their published 0.468.
    # tracin/ekfac are our attributors on ten full-set runs trained by bergson and imported into
    # tracin_ckpts/ (seeds 30000-30009; E=5 is runs 0-4, whose EK-FAC partials carry over to
    # E=10); the bank's tauq_ekfac_bergson.npz is bergson's own EK-FAC, a check of ours.
    # Bae et al.'s protocol: six checkpoints, one per half epoch (steps 291..1746,
    # so units_per_epoch=2), IF at the last at damping 1e-8.  No source block: ours is SGD-only,
    # their GPT-2 runs App. C's Adam-preconditioned variant.  EK-FAC chunks are halved from
    # query 32 / train 16, which OOM'd a 32 GB RTX 5090 (chunking only).
    "wikitext2_gpt2_bae": {"dataset": "wikitext2", "n_train": 4_656, "n_targets": 481,
                           "recipe": Gpt2WikitextBaeRecipe(), "harness": "bergson",
                           "seed_mode": "shared-per-replica", "projector_num_sms": 170,
                           "seeds": tuple(range(1004, 1024)),
                           # bergson seeds once per validate call, so a cell depends on its
                           # place in the call: every seed's subsets 0-99, 100-299 and 300-399
                           # ran as calls of 13 subsets, cut from each span's start.
                           "pool_spans": ((0, 100), (100, 300), (300, 400)), "pool_chunk": 13,
                           # TracIn: the 48 Linears at k=16384, a fresh projection per
                           # checkpoint.  All 124M params (the tied wte dominates) halve LDS,
                           # and one k=1024 matrix per run halves it again.
                           "tracin": {"proj_dim": 16_384, "ensemble_sizes": [5, 10],
                                      "ckpt_every": 1, "units_per_epoch": 2, "batch": 8,
                                      "tracked": GPT2_TRACKED,
                                      "projection_per_checkpoint": True,
                                      "require_cuda_projector": True},
                           "ekfac": {"damping": 1e-8, "ensemble_sizes": [5, 10],
                                     "tracked": GPT2_TRACKED,
                                     "factor_batch": 8, "query_batch": 16,
                                     "train_batch": 8},
                           # Ours (Bae et al. report no TRAK on WikiText): members trained by
                           # bergson on their own random halves, imported into ckpts/; causal-LM
                           # output (TRAK App. F.3) on every parameter, as for qnli: its kernel
                           # inverse whitens the tied wte (5 members: 0.169 vs 0.120 on the 48
                           # Linears).  50 members as for qnli.  proj_dim: pilot LDS peaked at
                           # 2048 (0.120) and fell past the n/2 guard (2,328 on 4,656 docs).
                           "trak": {"proj_dim": 2_048, "ensemble_sizes": [10, 20, 50],
                                    "e_features": 5,
                                    "trak_batch": 8, "require_cuda_projector": True,
                                    "use_half_precision": False}},
    # canonical ImageNet-1k / ResNet-18 (App. A.1 + ffcv-imagenet).  proj_dim: the paper reports
    # 15,000, but fast_jl requires multiples of 512 (so must the paper's own tooling); 30*512 =
    # 15,360 is the nearest.  soft_threshold_k: Eq. 16 with a fixed k=1000 nonzeros per query
    # (no CV, App. A.2), applied at evaluation time.
    "imagenet": {"dataset": "imagenet", "n_train": 1_281_167, "n_targets": 2_000,
                     "recipe": Resnet18Recipe(),
                     "trak": {"proj_dim": 15_360, "ensemble_sizes": [100],
                               "soft_threshold_k": 1000, "require_cuda_projector": True,
                               "featurize_augment": False, "e_features": 10}},
}
