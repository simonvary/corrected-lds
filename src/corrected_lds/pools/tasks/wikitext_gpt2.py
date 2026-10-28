"""The WikiText-2 / GPT-2 task: the model our attributors score, its loss, and the import of
bergson-trained checkpoints (``transformers`` is imported lazily).

No trainer lives here: bergson's harness (``pools/bergson.py``) trains both the retrains and
the scored full-set runs, so the two differ only in the seed.  Conventions:

1. Every HF ``Conv1D`` (``x @ W + b``, ``W`` shaped ``(in, out)``) is swapped exactly for the
   ``nn.Linear`` with ``weight = W.T``, the layout kronfluence and our gradient hooks track.
2. ``lm_head`` is tied to the ``wte`` embedding, which Bae et al. exclude (App. C) and
   bergson filters, so :data:`GPT2_TRACKED` is the 48 transformer Linears only.
3. The readout is the per-document mean token cross-entropy over the ``T - 1`` shifted
   targets, as bergson's ``validate`` writes it per query; every chunk is a full 512-token
   block, so mean and sum differ by a constant factor.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F

from corrected_lds.banks.presets import GPT2_TRACKED

__all__ = [
    "GPT2_TRACKED",
    "Gpt2Logits",
    "conv1d_to_linear",
    "hf_to_ours",
    "make_model",
    "per_doc_losses",
    "query_losses",
]

# the four Conv1D projections of every GPT-2 block, by their HF parameter names
_CONV1D_WEIGHT = re.compile(r"^transformer\.h\.\d+\.(attn\.c_attn|attn\.c_proj|"
                            r"mlp\.c_fc|mlp\.c_proj)\.weight$")


class Gpt2Logits(torch.nn.Module):
    """``input_ids (B, T) -> logits (B, T, V)``, the bare tensor kronfluence's ``Task`` and
    ``torch.func.functional_call`` want.  The HF model sits under ``lm``, so module names read
    ``lm.transformer.h.0.attn.c_attn`` (as :data:`GPT2_TRACKED` lists them)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids):
        return self.lm(input_ids=input_ids).logits


def conv1d_to_linear(module) -> int:
    """Replace every HF ``Conv1D`` under ``module``, in place, by the equivalent ``nn.Linear``
    (``weight = W.T``); returns how many were swapped."""
    from transformers.pytorch_utils import Conv1D

    swapped = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                w = child.weight                      # (in, out)
                lin = torch.nn.Linear(w.shape[0], w.shape[1], bias=child.bias is not None,
                                      device=w.device, dtype=w.dtype)
                with torch.no_grad():
                    lin.weight.copy_(w.t())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, lin)
                swapped += 1
    return swapped


def make_model(recipe=None, device="cuda", config=None) -> Gpt2Logits:
    """GPT-2 small (124M) with every Conv1D swapped for Linear, in eval mode.  The architecture
    is ``GPT2Config()``'s defaults (12 x 768, 50,257 tokens, 1,024 positions), so nothing is
    downloaded: weights come from an imported checkpoint.  ``use_cache=False``: scoring never
    decodes.  ``config`` serves the tests' tiny model; ``recipe``, if given, must name gpt2."""
    from transformers import GPT2Config, GPT2LMHeadModel

    if recipe is not None and getattr(recipe, "model", "gpt2") != "gpt2":
        raise ValueError(f"[wikitext] make_model builds gpt2-small; recipe asks for "
                         f"{recipe.model!r}")
    cfg = config if config is not None else GPT2Config()
    cfg.use_cache = False
    lm = GPT2LMHeadModel(cfg)              # ties lm_head to wte (tie_word_embeddings)
    conv1d_to_linear(lm)                   # touches neither, so the tie survives
    return Gpt2Logits(lm).to(device).eval()


def hf_to_ours(sd) -> dict:
    """A HF ``GPT2LMHeadModel`` state dict -> one :func:`make_model` loads strictly: Conv1D
    weights transposed, keys prefixed ``lm.``, the tied ``lm_head.weight`` (saved once)
    restored, and the ``attn.bias``/``attn.masked_bias`` buffers older transformers saved
    dropped; anything else is left for ``strict=True`` to name.  The layout is read, not
    assumed, as bergson's transformers (5.x) wrote the checkpoints: ``c_attn`` is
    ``(768, 2304)`` as Conv1D and ``(2304, 768)`` as Linear, so its shape decides for all four."""
    c_attn = sd["transformer.h.0.attn.c_attn.weight"]
    conv1d = c_attn.shape[0] < c_attn.shape[1]
    out = {}
    for k, v in sd.items():
        if re.search(r"\.attn\.(bias|masked_bias)$", k):
            continue
        out[f"lm.{k}"] = v.t().contiguous() if conv1d and _CONV1D_WEIGHT.match(k) else v
    if "lm.lm_head.weight" not in out and "lm.transformer.wte.weight" in out:
        out["lm.lm_head.weight"] = out["lm.transformer.wte.weight"]
    return out


def per_doc_losses(logits, ids) -> torch.Tensor:
    """``(B,)`` mean next-token cross-entropy per document (convention 3)."""
    V = logits.shape[-1]
    tok = F.cross_entropy(logits[:, :-1, :].reshape(-1, V).float(),
                          ids[:, 1:].reshape(-1), reduction="none")
    return tok.view(ids.shape[0], -1).mean(dim=1)


@torch.no_grad()
def query_losses(model, data, device="cuda", eval_batch: int = 8, dropout=False,
                 seed=0) -> np.ndarray:
    """Per-query mean token cross-entropy, ``(Q,)`` float64; ``dropout=False`` is the
    attributors' eval-mode view.  ``dropout=True`` (``seed`` fixes the draw) matches the Bae
    bank's pool, for the import gate: bergson's in-loop readout (``validate.py`` at the pinned
    commit) never calls ``.eval()``, so under ``train_mode: true`` dropout was active."""
    if dropout:
        model.train()
        torch.manual_seed(int(seed))
    else:
        model.eval()
    ids = data["q_lm_ids"]
    outs = []
    try:
        for i in range(0, int(ids.shape[0]), int(eval_batch)):
            b = ids[i:i + eval_batch].to(device)
            outs.append(per_doc_losses(model(b), b).double().cpu())
    finally:
        model.eval()
    return torch.cat(outs).numpy()
