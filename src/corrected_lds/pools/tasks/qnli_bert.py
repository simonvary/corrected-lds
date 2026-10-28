"""The QNLI / BERT pool task: BERT with the pooler tanh removed, faithful HF-default
finetuning, batched margin readout.

``transformers`` is imported lazily, so the module imports with torch alone.  The LR is the
HF run_glue default the recipe pins (linear 1 -> 0, no warmup).  ``precision="bf16"`` (the
rerun default) autocasts without a ``GradScaler``; ``"fp32"`` (paper-exact) disables
autocast, as does the CPU path.
"""

import numpy as np
import torch

from corrected_lds.banks.presets import BertRecipe
from corrected_lds.pools.train import linear_decay_lr, per_query_margin, run_subset_sgd

__all__ = ["BertRecipe", "make_text_model", "margins", "train_text_model"]


class _BertLogits(torch.nn.Module):
    """Thin wrapper making a HF ``BertForSequenceClassification`` TRAK-compatible.

    traker calls the model by keyword and indexes the return as a logits tensor, so this
    returns ``.logits``.  The pooler ``tanh``, the one saturating nonlinearity on the
    [CLS] -> classifier path, becomes identity, as the paper linearizes for TRAK (App. A.1).
    Verified with traker 0.3.2; needs ``transformers<5`` (5.x's masking_utils breaks under
    vmap)."""

    def __init__(self, backbone, remove_pooler_tanh: bool = True):
        super().__init__()
        self.backbone = backbone
        if remove_pooler_tanh and getattr(backbone, "bert", None) is not None \
                and backbone.bert.pooler is not None:
            backbone.bert.pooler.activation = torch.nn.Identity()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        return self.backbone(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask).logits


def make_text_model(recipe: BertRecipe, num_labels: int, device="cuda", seed=None,
                    pretrained: bool = True) -> torch.nn.Module:
    """Build the ``_BertLogits`` model for ``recipe.model`` (``bert-base-cased``).

    ``pretrained=True`` loads BERT with a fresh head seeded by ``seed`` (the LDS retrains);
    ``False`` builds random weights from config without a download (TRAK's ``model_ctor``,
    whose ``load_checkpoint`` overwrites them)."""
    from transformers import AutoConfig, BertForSequenceClassification
    if seed is not None:
        torch.manual_seed(int(seed))
    conf = AutoConfig.from_pretrained(recipe.model, num_labels=int(num_labels))
    if pretrained:
        backbone = BertForSequenceClassification.from_pretrained(recipe.model, config=conf)
    else:
        backbone = BertForSequenceClassification(conf)
    model = _BertLogits(backbone, remove_pooler_tanh=recipe.remove_pooler_tanh)
    return model.to(device)


def train_text_model(data, subset_idx, seed, recipe: BertRecipe, device="cuda",
                     verbose=False, on_epoch_end=None) -> torch.nn.Module:
    """Finetune one BERT on the physical subset ``subset_idx`` (indices into the train tensors)
    and return it.  ``recipe`` has no default because it is the pool identity; ``seed`` sets
    the head init and data order; ``on_epoch_end`` is the TracIn checkpoint hook passed to
    ``run_subset_sgd`` (RNG-neutral by contract)."""
    model = make_text_model(recipe, data["meta"]["n_classes"], device, seed=int(seed),
                            pretrained=True)
    ids, tt, am, y = (data["input_ids"], data["token_type_ids"],
                      data["attention_mask"], data["labels"])
    idx = torch.as_tensor(np.asarray(subset_idx, dtype=np.int64), device=device)
    n_sub = int(idx.numel())
    steps_per_epoch = max(1, int(np.ceil(n_sub / recipe.batch)))

    opt = torch.optim.SGD(model.parameters(), lr=recipe.peak_lr, momentum=recipe.momentum,
                          weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(               # linear 1 -> 0, no warmup
        opt, linear_decay_lr(recipe.epochs, steps_per_epoch))
    dt = torch.device(device).type
    amp_dt = torch.bfloat16 if recipe.precision == "bf16" else torch.float16
    amp_on = (dt == "cuda" and recipe.precision != "fp32")   # fp32 -> autocast off
    clip = float(recipe.max_grad_norm)
    lossfn = torch.nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)

    def batch_loss(b, gen, _epoch):        # no augmentation: gen drives only the permutation
        with torch.amp.autocast(dt, dtype=amp_dt, enabled=amp_on):
            logits = model(input_ids=ids[b], token_type_ids=tt[b], attention_mask=am[b])
            return lossfn(logits, y[b])

    def backward_step(loss):
        loss.backward()
        if clip > 0:                                     # HF run_glue default max_grad_norm=1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

    run_subset_sgd(model, idx, recipe.epochs, recipe.batch, seed, opt, sched, device,
                   batch_loss, backward_step, on_epoch_end=on_epoch_end)
    if verbose:
        print(f"    [train-text] seed {seed} n_sub={n_sub} done", flush=True)
    return model


@torch.no_grad()
def margins(model, data, device="cuda", eval_batch: int = 128,
                      precision: str = "fp32") -> np.ndarray:
    """Per-query correct-class log-odds margin on the QNLI query set -> ``(Q,)`` float64: the
    LDS ground-truth readout and TRAK's text target, in ``eval_batch`` batches.  Pass the
    recipe's ``precision`` so the readout dtype matches training ("fp32" disables autocast)."""
    model.eval()
    dt = torch.device(device).type
    amp_dt = torch.bfloat16 if precision == "bf16" else torch.float16
    amp_on = (dt == "cuda" and precision != "fp32")
    ids, tt, am, y = (data["q_input_ids"], data["q_token_type_ids"],
                      data["q_attention_mask"], data["q_labels"])
    Q = int(ids.shape[0])
    outs = []
    for i in range(0, Q, eval_batch):
        with torch.amp.autocast(dt, dtype=amp_dt, enabled=amp_on):
            logits = model(input_ids=ids[i:i + eval_batch], token_type_ids=tt[i:i + eval_batch],
                           attention_mask=am[i:i + eval_batch])
        outs.append(per_query_margin(logits.float(), y[i:i + eval_batch]).double().cpu())
    return torch.cat(outs).numpy()
