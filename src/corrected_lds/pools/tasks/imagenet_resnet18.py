"""The ImageNet / ResNet-18 pool task: torchvision model + BlurPool, the ffcv-imagenet
recipe reimplemented in torch, streaming-JPEG trainer, batched margin readout.

One cell follows TRAK App. A.1 and the ffcv-imagenet rn18 configs: SGD with the bn-name
weight-decay split, two-level cyclic LR (knots ``[0, peak, epochs] -> [1e-4*lr, lr, 0]``,
interpolated per step), batch 1024, resolution 160 -> 192 (epochs 0-11 @160, 12-14 @192),
RandomResizedCrop + flip, fp16 autocast + GradScaler, channels_last.  ffcv is not imported;
all of this is reproduced exactly except the torchvision decode/resize kernels (vs
opencv/numba), a true randperm (vs QUASI_RANDOM), the kept partial batch (vs drop_last) and
fp32 + autocast inputs (vs fp16 tensors).  One ``Generator(seed+1)`` drives the randperm and
every augmentation draw, so a cell is bitwise-reproducible given its seed, as for CIFAR.
"""
# Adapted in part, with changes, from ffcv-imagenet (Apache-2.0) and torchvision
# (BSD-3-Clause): THIRD_PARTY_NOTICES.md

import math

import numpy as np
import torch

from corrected_lds.banks.presets import Resnet18Recipe
from corrected_lds.pools.datasets import IMAGENET_MEAN255, IMAGENET_STD255
from corrected_lds.pools.train import per_query_margin, run_subset_sgd

__all__ = [
    "BlurPoolConv2d",
    "Resnet18Recipe",
    "apply_blurpool",
    "featurize_batches",
    "imagenet_cyclic_lr",
    "make_model",
    "margins",
    "train_imagenet_model",
]


# -- the "modified ResNet-18": torchvision + BlurPool (ffcv-imagenet verbatim) --- #
class BlurPoolConv2d(torch.nn.Module):
    """ffcv-imagenet's BlurPoolConv2d, verbatim: a fixed 3x3 [1,2,1]x[1,2,1]/16
    depthwise blur applied before the wrapped strided conv."""

    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]]]) / 16.0
        self.conv = conv
        self.register_buffer("blur_filter", default_filter.repeat(conv.in_channels, 1, 1, 1))

    def forward(self, x):
        blurred = torch.nn.functional.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                                             groups=self.conv.in_channels, bias=None)
        return self.conv(blurred)


def apply_blurpool(model):
    """ffcv-imagenet's rule, verbatim: wrap every ``Conv2d`` with stride > 1 and in_ch >= 16
    (on resnet18, the six strided convs of layer2-4 incl. downsample 1x1s, not the stem)."""
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d) and (max(child.stride) > 1
                                                   and child.in_channels >= 16):
            setattr(model, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def make_model(recipe: Resnet18Recipe, num_classes=1_000, seed=None,
               device="cuda") -> torch.nn.Module:
    """Build the (blurpooled) torchvision ResNet-18 from scratch, as the paper trains; a given
    ``seed`` seeds the global RNG first, since the init is part of cell identity."""
    import torchvision
    if seed is not None:
        torch.manual_seed(int(seed))
    model = torchvision.models.resnet18(weights=None, num_classes=int(num_classes))
    if recipe.use_blurpool:
        apply_blurpool(model)
    return model.to(memory_format=torch.channels_last).to(device)


def _bn_split_param_groups(model, weight_decay):
    """ffcv-imagenet's ``create_optimizer`` split, quirk included: weight decay only for params
    whose name lacks "bn", so the downsample BNs (named ``downsample.1``) do decay."""
    named = list(model.named_parameters())
    return [{"params": [p for k, p in named if "bn" in k], "weight_decay": 0.0},
            {"params": [p for k, p in named if "bn" not in k], "weight_decay": weight_decay}]


# -- the reference schedules, as pure functions ----------------------------- #
def imagenet_cyclic_lr(epochs, peak_epoch, steps_per_epoch, peak_lr, start_frac):
    """The reference two-level schedule as a function of the global step: ``get_cyclic_lr``'s
    epoch knots ``[0, peak, epochs] -> [start_frac*lr, lr, 0]``, linearly interpolated within
    each epoch as in ``train_loop`` (unit-tested).  Returns the absolute LR, so pair it with
    ``SGD(lr=1)`` + ``LambdaLR``."""
    def epoch_lr(e):
        return float(np.interp([e], [0, peak_epoch, epochs],
                               [start_frac * peak_lr, peak_lr, 0.0])[0])

    def f(step):
        e, ix = divmod(int(step), int(steps_per_epoch))
        if e >= epochs:
            return 0.0
        lr0, lr1 = epoch_lr(e), epoch_lr(e + 1)
        return lr0 + (lr1 - lr0) * ix / steps_per_epoch

    return f


def _train_resolution(epoch, recipe: Resnet18Recipe) -> int:
    """ffcv-imagenet ``get_resolution``: min_res through start_ramp, max_res from end_ramp,
    else linear rounded half-to-even to a multiple of 32 (never 176 with the paper defaults)."""
    if epoch <= recipe.start_ramp:
        return recipe.min_res
    if epoch >= recipe.end_ramp:
        return recipe.max_res
    interp = np.interp([epoch], [recipe.start_ramp, recipe.end_ramp],
                       [recipe.min_res, recipe.max_res])
    return int(np.round(interp[0] / 32)) * 32


# -- augmentation: decode + RRC + flip, all randomness from the cell generator --- #
def _rrc_params(height, width, gen, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
    """torchvision v1 ``RandomResizedCrop.get_params`` drawing from ``gen`` in the same order,
    so a shared seed gives identical boxes (tested); ffcv's ``get_random_crop`` (fast_crop.py:17)
    is the same 10-try + center-fallback algorithm.  Returns ``(top, left, h, w)``."""
    dev = gen.device
    area = height * width
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(10):
        target_area = area * torch.empty(1, device=dev).uniform_(
            scale[0], scale[1], generator=gen).item()
        aspect = math.exp(torch.empty(1, device=dev).uniform_(
            log_lo, log_hi, generator=gen).item())
        w = round(math.sqrt(target_area * aspect))
        h = round(math.sqrt(target_area / aspect))
        if 0 < w <= width and 0 < h <= height:
            top = int(torch.randint(0, height - h + 1, (1,), device=dev,
                                    generator=gen).item())
            left = int(torch.randint(0, width - w + 1, (1,), device=dev,
                                     generator=gen).item())
            return top, left, h, w
    in_ratio = float(width) / float(height)               # central-crop fallback
    if in_ratio < min(ratio):
        w = width
        h = round(w / min(ratio))
    elif in_ratio > max(ratio):
        h = height
        w = round(h * max(ratio))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def _resize(crop_u8, res):
    """Crop (C, h, w) u8 -> (C, res, res) float32, bilinear + antialias (torchvision's kernel)."""
    x = crop_u8.to(torch.float32).unsqueeze(0)
    return torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear",
                                           antialias=True)[0]


def _decode_jpeg_batch(raws, device):
    """Decode 1-D u8 CPU ``raws`` to (3, h, w) u8 tensors on ``device``: batched nvjpeg on
    CUDA (safe, the cache re-encoded every image to baseline RGB), per-image libjpeg on CPU."""
    from torchvision.io import ImageReadMode, decode_jpeg
    if torch.device(device).type == "cuda":
        return decode_jpeg(raws, mode=ImageReadMode.RGB, device=device)
    return [decode_jpeg(r, mode=ImageReadMode.RGB) for r in raws]


def _augment_batch(imgs, res, gen):
    """Per-image RandomResizedCrop, then one batched flip, all drawn from ``gen`` in this
    order -> (B, 3, res, res) float32 on the images' device, raw-255 scale."""
    out = torch.empty(len(imgs), 3, res, res, device=imgs[0].device, dtype=torch.float32)
    for i, img in enumerate(imgs):
        _, h, w = img.shape
        top, left, ch, cw = _rrc_params(h, w, gen)
        out[i] = _resize(img[:, top:top + ch, left:left + cw], res)
    flip = torch.rand(len(imgs), device=gen.device, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def _center_crop_resize(img_u8, res, crop_ratio):
    """Deterministic val-geometry crop for the ``augment=False`` ablation: the centered
    ``crop_ratio * short side`` square, resized to res^2 (as in the query decode)."""
    _, h, w = img_u8.shape
    s = max(1, round(crop_ratio * min(h, w)))
    top, left = (h - s) // 2, (w - s) // 2
    return _resize(img_u8[:, top:top + s, left:left + s], res)


def featurize_batches(data, batch, device, val_res, crop_ratio):
    """Deterministic TRAK featurization batches over the train universe in canonical order
    (tauq row i <-> mask column i): the val-geometry center crop, no RNG, so every ensemble
    member sees identical tensors (``featurize_augment: False``).  Decodes with nvjpeg on
    CUDA; queries use the loader's PIL decode (same geometry, different kernel).
    Yields ``(xb (B, 3, val_res, val_res) f32 normalized, yb)``."""
    jpegs, offsets, rows, y = (data["jpegs"], data["offsets"], data["train_rows"],
                               data["y_train"])
    mean = torch.as_tensor(IMAGENET_MEAN255, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD255, device=device).view(1, 3, 1, 1)
    for s in range(0, len(rows), batch):
        rs = rows[s:s + batch]
        raws = [torch.from_numpy(np.array(jpegs[offsets[r]:offsets[r + 1]]))
                for r in rs]
        imgs = _decode_jpeg_batch(raws, device)
        xb = torch.stack([_center_crop_resize(im, val_res, crop_ratio) for im in imgs])
        yield ((xb - mean) / std).contiguous(), y[s:s + batch]


# -- the trainer + readout -------------------------------------------------- #
def train_imagenet_model(data, subset_idx, seed, recipe: Resnet18Recipe, device="cuda",
                         verbose=False, on_epoch_end=None) -> torch.nn.Module:
    """Train one (blurpooled) ResNet-18 on ``subset_idx`` (indices into the loader's train
    universe; blob rows via ``data["train_rows"]``) and return it.  ``recipe`` has no default
    because it is the pool identity; ``seed`` sets the init, data order and augmentation;
    ``on_epoch_end`` is the TracIn checkpoint hook passed to :func:`run_subset_sgd`
    (RNG-neutral by contract).  The partial batch is kept, so a size-1 final batch, which
    BatchNorm rejects, is refused up front (the canonical 640,583 % 1024 = 583 is safe)."""
    idx = torch.as_tensor(np.asarray(subset_idx, dtype=np.int64), device=device)
    n_sub = int(idx.numel())
    if n_sub > 1 and n_sub % recipe.batch == 1:
        raise ValueError(
            f"[imagenet] n_sub={n_sub} with batch={recipe.batch} leaves a size-1 final "
            f"batch, which BatchNorm rejects in train mode; pick a batch that does not "
            f"leave a lone sample (the canonical config is safe: 640,583 % 1024 = 583).")
    model = make_model(recipe, num_classes=data["meta"]["n_classes"], seed=int(seed),
                       device=device)
    jpegs, offsets = data["jpegs"], data["offsets"]
    train_rows, y = data["train_rows"], data["y_train"]
    steps_per_epoch = max(1, int(np.ceil(n_sub / recipe.batch)))

    opt = torch.optim.SGD(_bn_split_param_groups(model, recipe.weight_decay), lr=1.0,
                          momentum=recipe.momentum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, imagenet_cyclic_lr(recipe.epochs, recipe.peak_epoch, steps_per_epoch,
                                recipe.peak_lr, recipe.lr_start_frac))
    dt = torch.device(device).type
    scaler = torch.amp.GradScaler(dt, enabled=(dt == "cuda"))
    lossfn = torch.nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    mean = torch.as_tensor(IMAGENET_MEAN255, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD255, device=device).view(1, 3, 1, 1)

    def batch_loss(b, gen, epoch):
        res = _train_resolution(epoch, recipe)
        rows = train_rows[b.cpu().numpy()]
        raws = [torch.from_numpy(np.array(jpegs[offsets[r]:offsets[r + 1]]))
                for r in rows]
        imgs = _decode_jpeg_batch(raws, device)
        if recipe.augment:
            xb = _augment_batch(imgs, res, gen)
        else:                                   # ablation: deterministic val geometry
            xb = torch.stack([_center_crop_resize(im, res, recipe.crop_ratio)
                              for im in imgs])
        xb = ((xb - mean) / std).to(memory_format=torch.channels_last)
        with torch.amp.autocast(dt, enabled=(dt == "cuda")):
            return lossfn(model(xb), y[b])

    def backward_step(loss):
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    run_subset_sgd(model, idx, recipe.epochs, recipe.batch, seed, opt, sched, device,
                   batch_loss, backward_step, on_epoch_end=on_epoch_end)
    if verbose:
        print(f"    [train-imagenet] seed {seed} n_sub={int(idx.numel())} done",
              flush=True)
    return model


@torch.no_grad()
def margins(model, data, device="cuda", eval_batch: int = 256) -> np.ndarray:
    """Per-query correct-class log-odds margin on the query set -> ``(Q,)`` float64: the LDS
    ground-truth readout, TRAK's target (Eq. 19), no flip-TTA.  ``X_query`` was decoded at
    load (crop 224/256 x short side -> val_res^2); ``eval_batch`` bounds memory."""
    model.eval()
    dt = torch.device(device).type
    Xq, yq = data["X_query"], data["y_query"]
    outs = []
    for i in range(0, len(yq), eval_batch):
        xb = Xq[i:i + eval_batch].to(memory_format=torch.channels_last)
        with torch.amp.autocast(dt, enabled=(dt == "cuda")):
            logits = model(xb)
        outs.append(per_query_margin(logits.float(), yq[i:i + eval_batch]).double().cpu())
    return torch.cat(outs).numpy()
