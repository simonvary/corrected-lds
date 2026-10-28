"""The CIFAR / ResNet-9 pool task: model, FFCV-faithful augmentation, trainer, readout.

One cell follows TRAK App. A.1 and the released FFCV code: SGD (momentum 0.9, wd 5e-4),
single-triangle LR peaking at ``peak_epoch``, 24 epochs, batch 512, AMP, channels_last,
flip + pad-2 translate + Cutout(4).  Each model's seed sets its own init and data order, so
retrains carry real seed noise (the eps the LDS disattenuation removes).
"""
# Adapted in part, with changes, from TRAK (MIT) and FFCV (Apache-2.0): THIRD_PARTY_NOTICES.md

import numpy as np
import torch
from torch import nn

from corrected_lds.banks.presets import Resnet9Recipe
from corrected_lds.pools.train import per_query_margin, run_subset_sgd, triangular_lr

__all__ = [
    "Resnet9Recipe",
    "augment",
    "make_model",
    "margins",
    "resnet9_bn",
    "train_paper_model",
]


# -- ResNet-9-BN (the exact FFCV / TRAK construct_rn9) ---------------------- #
class Mul(nn.Module):
    """Constant output scale (the DAWNBench recipe's 0.2 logit multiplier)."""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = float(weight)

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def _conv_bn(ci, co, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


def resnet9_bn(num_classes: int = 10, in_ch: int = 3, width: float = 1.0) -> nn.Module:
    """ResNet-9-BN, the exact FFCV / TRAK ``construct_rn9`` (bias-free head, ``Mul(0.2)``);
    ``width`` scales the channel counts, keeping the block structure (1.0 = paper)."""
    def c(x):
        return max(1, round(x * width))

    c64, c128, c256 = c(64), c(128), c(256)
    return nn.Sequential(
        _conv_bn(in_ch, c64, kernel=3, stride=1, padding=1),
        _conv_bn(c64, c128, kernel=5, stride=2, padding=2),
        Residual(nn.Sequential(_conv_bn(c128, c128), _conv_bn(c128, c128))),
        _conv_bn(c128, c256, kernel=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(_conv_bn(c256, c256), _conv_bn(c256, c256))),
        _conv_bn(c256, c128, kernel=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(c128, num_classes, bias=False),
        Mul(0.2),
    )


def make_model(num_classes=10, in_ch=3, width=1.0, seed=None, device="cuda") -> nn.Module:
    """Build a ResNet-9-BN; a given ``seed`` seeds the global RNG first, since the init is
    part of cell identity."""
    if seed is not None:
        torch.manual_seed(int(seed))
    return resnet9_bn(num_classes, in_ch, width).to(device=device, dtype=torch.float32)


# -- FFCV-faithful augmentation (on-device, per batch) ---------------------- #
def augment(x, pad: int = 2, cutout: int = 4, gen: torch.Generator = None,
            per_sample: bool = True):
    """TRAK App. A.1 augmentation of a normalized batch ``x`` (B, C, H, W), same shape out:
    flip, pad-``pad`` translate with a 0 border (FFCV's mean fill, after normalization) and
    ``Cutout(cutout)`` fully inside (as ffcv cutout.py); ``per_sample=True`` (the FFCV recipe)
    draws per-image offsets.  All draws come from ``gen``, the data-order generator of
    ``run_subset_sgd``, in an order that is part of the reproducibility contract."""
    B, C, H, W = x.shape
    dev = x.device
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    # random translate: constant-0 pad then random HxW crop
    if pad > 0:
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        if per_sample:                                  # independent offset per image
            oy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
            ox = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
            b_ = torch.arange(B, device=dev)[:, None, None, None]
            c_ = torch.arange(C, device=dev)[None, :, None, None]
            y_ = oy[:, None, None, None] + torch.arange(H, device=dev)[None, None, :, None]
            x_ = ox[:, None, None, None] + torch.arange(W, device=dev)[None, None, None, :]
            x = xp[b_, c_, y_, x_]
        else:                                           # one shared offset per batch
            oy = int(torch.randint(0, 2 * pad + 1, (1,), device=dev, generator=gen).item())
            ox = int(torch.randint(0, 2 * pad + 1, (1,), device=dev, generator=gen).item())
            x = xp[:, :, oy:oy + H, ox:ox + W]
    # per-sample Cutout: zero a cutout x cutout square, origin fully inside
    if cutout > 0:
        top = torch.randint(0, max(1, H - cutout + 1), (B,), device=dev, generator=gen)
        left = torch.randint(0, max(1, W - cutout + 1), (B,), device=dev, generator=gen)
        yy = torch.arange(H, device=dev)[None, :, None]
        xx = torch.arange(W, device=dev)[None, None, :]
        ym = (yy >= top[:, None, None]) & (yy < (top[:, None, None] + cutout))
        xm = (xx >= left[:, None, None]) & (xx < (left[:, None, None] + cutout))
        x = x.masked_fill((ym & xm)[:, None, :, :], 0.0)
    return x.contiguous()


# -- the trainer + readout -------------------------------------------------- #
def train_paper_model(data, subset_idx, seed, recipe, device="cuda",
                      verbose=False, on_epoch_end=None) -> torch.nn.Module:
    """Train one ResNet-9-BN on the physical subset ``subset_idx`` (indices into the train
    tensors) and return it.  ``recipe`` has no default because it is the pool identity;
    ``seed`` sets the init, data order and augmentation; ``on_epoch_end`` is the TracIn
    checkpoint hook passed to ``run_subset_sgd`` (RNG-neutral by contract)."""
    model = make_model(num_classes=data["meta"]["n_classes"], in_ch=data["meta"]["in_ch"],
                       width=recipe.width, seed=int(seed), device=device)
    model = model.to(memory_format=torch.channels_last)
    X, y = data["X_train"], data["y_train"]
    idx = torch.as_tensor(np.asarray(subset_idx, dtype=np.int64), device=device)
    steps_per_epoch = max(1, int(np.ceil(int(idx.numel()) / recipe.batch)))

    opt = torch.optim.SGD(model.parameters(), lr=recipe.peak_lr, momentum=recipe.momentum,
                          weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, triangular_lr(recipe.epochs, recipe.peak_epoch, steps_per_epoch))
    dt = torch.device(device).type                       # "cuda" | "cpu" (CPU: tests)
    scaler = torch.amp.GradScaler(dt, enabled=(dt == "cuda"))
    lossfn = torch.nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)

    def batch_loss(b, gen, _epoch):                      # aug draws from the randperm's gen
        xb = X[b].to(memory_format=torch.channels_last)
        if recipe.augment:
            xb = augment(xb, recipe.pad, recipe.cutout, gen, per_sample=recipe.per_sample_aug)
        with torch.amp.autocast(dt, enabled=(dt == "cuda")):
            return lossfn(model(xb), y[b])

    def backward_step(loss):
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    run_subset_sgd(model, idx, recipe.epochs, recipe.batch, seed, opt, sched, device,
                   batch_loss, backward_step, on_epoch_end=on_epoch_end)
    if verbose:
        acc = _train_acc(model, X[idx], y[idx], device)
        print(f"    [train] seed {seed} n_sub={int(idx.numel())} -> train_acc {acc:.3f}",
              flush=True)
    return model


@torch.no_grad()
def _train_acc(model, X, y, device, batch=1024):
    model.eval()
    dt = torch.device(device).type
    correct = 0
    for i in range(0, len(y), batch):
        with torch.amp.autocast(dt, enabled=(dt == "cuda")):
            pred = model(X[i:i + batch].to(memory_format=torch.channels_last)).argmax(1)
        correct += int((pred == y[i:i + batch]).sum().item())
    return correct / max(1, len(y))


@torch.no_grad()
def margins(model, data, device="cuda") -> np.ndarray:
    """Per-query correct-class log-odds margin on the query set -> ``(Q,)`` float64: the LDS
    ground-truth readout and TRAK's target function, under the training autocast policy."""
    model.eval()
    dt = torch.device(device).type
    Xq = data["X_query"].to(memory_format=torch.channels_last)
    with torch.amp.autocast(dt, enabled=(dt == "cuda")):
        logits = model(Xq)
    return per_query_margin(logits.float(), data["y_query"]).double().cpu().numpy()
