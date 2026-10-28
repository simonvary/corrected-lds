"""Torch-free recipe identity: ``to_dict()`` is the pool identity of the rerun banks, every
field serialized and json-stable.
"""

import json

from corrected_lds.banks.presets import (
    PRESETS,
    BertRecipe,
    Resnet9Recipe,
    Resnet18Recipe,
    bank_hashes,
)


def test_resnet9_recipe_identity_dict():
    d = Resnet9Recipe().to_dict()
    # dropped fields must not resurface (order_mode and the frozen legacy-compat knobs)
    for gone in ("order_mode", "translate_fill", "cutout_center", "norm"):
        assert gone not in d
    assert d["epochs"] == 24 and d["peak_lr"] == 0.5 and d["alpha"] == 0.5
    assert len(d) == 13                                   # every field serialized
    s = json.dumps(d, sort_keys=True)
    assert json.loads(s) == d                             # identity is json-stable


def test_bert_recipe_defaults():
    """The HF run_glue knobs the paper defers to, as one unit, plus one deliberate deviation:
    bf16 autocast is the rerun default (fp32 = paper-exact)."""
    r = BertRecipe()
    assert (r.momentum, r.weight_decay, r.batch, r.precision, r.max_grad_norm) == \
        (0.0, 0.0, 8, "bf16", 1.0)
    assert (r.optimizer, r.peak_lr, r.epochs, r.seq_len) == ("sgd", 1e-3, 20, 128)
    assert r.remove_pooler_tanh is True
    d = r.to_dict()
    assert len(d) == 14 and d["model"] == "bert-base-cased"   # every field serialized
    assert "trak_use_half_precision" not in d       # TRAK knobs are tauq identity, not pool
    assert json.loads(json.dumps(d, sort_keys=True)) == d


def test_resnet18_recipe_identity_dict():
    """App. A.1's six knobs + the ffcv-imagenet rn18 config values, as one frozen unit;
    "initial learning rate 5.2" is pinned as the peak (the one interpreted knob)."""
    d = Resnet18Recipe().to_dict()
    assert (d["epochs"], d["peak_lr"], d["peak_epoch"]) == (15, 5.2, 2)
    assert (d["momentum"], d["weight_decay"], d["label_smoothing"]) == (0.8, 4e-5, 0.05)
    assert (d["batch"], d["alpha"], d["lr_start_frac"]) == (1024, 0.5, 1e-4)
    assert (d["min_res"], d["max_res"], d["start_ramp"], d["end_ramp"]) == (160, 192, 11, 13)
    assert (d["val_res"], d["crop_ratio"]) == (256, 224 / 256)
    assert d["arch"] == "resnet18" and d["use_blurpool"] is True and d["augment"] is True
    assert len(d) == 18                                   # every field serialized
    assert json.loads(json.dumps(d, sort_keys=True)) == d # identity is json-stable


def test_presets_registry():
    assert set(PRESETS) == {"pilot", "cifar10", "mnist", "qnli", "imagenet",
                            "wikitext2_gpt2_bae"}
    assert PRESETS["qnli"]["recipe"] == BertRecipe()          # bf16 rerun default
    assert PRESETS["cifar10"]["trak"]["proj_dim"] % 512 == 0  # fast_jl constraint
    assert PRESETS["imagenet"]["recipe"] == Resnet18Recipe()
    assert PRESETS["imagenet"]["n_train"] == 1_281_167        # full ILSVRC2012 train
    assert PRESETS["imagenet"]["trak"]["proj_dim"] % 512 == 0 # paper's 15,000 -> 15,360
    assert PRESETS["imagenet"]["trak"]["soft_threshold_k"] == 1000   # Eq. 16 fixed-k
    assert PRESETS["imagenet"]["trak"]["require_cuda_projector"] is True
    # combined side digests over all of a side's pins (images/tokens and labels)
    assert bank_hashes("cifar10") == ("0c79d6744e51f344", "72eeeba5111be6e3")
    assert bank_hashes("qnli") == ("1615f1b2274e6f81", "c6a11dfc0feaeb7f")
    assert bank_hashes("wikitext2") == ("03091dbab9d466e3", "d374d5b0d1ccc2a4")
    # bergson's Bae replication is a companion bank, not a retrain-series member: masks zero loss
    # weights and one seed covers a whole replica, so it must never be stamped "nested"
    wt = PRESETS["wikitext2_gpt2_bae"]
    assert wt["seed_mode"] == "shared-per-replica"
    assert wt["harness"] == "bergson"
    assert wt["recipe"].alpha == 0.5 and wt["recipe"].subset_weight == 0.0
    assert wt["recipe"].train_mode is True                    # dropout on in this recipe
    assert (wt["n_train"], wt["n_targets"]) == (4_656, 481)   # full validation split
    # our attributors on bergson-trained full-set runs: Bae et al.'s six half-epoch checkpoints,
    # IF at damping 1e-8 over the 48 transformer Linears; no SOURCE (ours is SGD-only, their
    # GPT-2 runs App. C's Adam-preconditioned variant); TRAK, which Bae omit on WikiText, is
    # qnli's trak5/trak1 pair on every parameter, under the n/2 projection guard
    assert "source" not in wt
    assert wt["trak"]["e_features"] == 5 and max(wt["trak"]["ensemble_sizes"]) % 5 == 0
    assert 2 * wt["trak"]["proj_dim"] < wt["n_train"]
    assert wt["trak"]["proj_dim"] % 512 == 0                  # fast_jl constraint
    assert wt["tracin"]["units_per_epoch"] == 2 and wt["tracin"]["ckpt_every"] == 1
    assert wt["ekfac"]["damping"] == 1e-8
    from corrected_lds.banks.presets import GPT2_TRACKED
    assert wt["ekfac"]["tracked"] == GPT2_TRACKED and len(GPT2_TRACKED) == 48
    # TracIn traces the same Linears (all-parameter TracIn is wte-dominated) at k=16384
    assert wt["tracin"]["tracked"] == GPT2_TRACKED and wt["tracin"]["proj_dim"] == 16_384
    assert wt["tracin"]["projection_per_checkpoint"] is True
    assert "tracked" not in wt["trak"]                         # all parameters, as qnli
    assert not any("lm_head" in m or "wte" in m for m in GPT2_TRACKED)
