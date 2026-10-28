"""Shared cell mechanics for every pool task: the training loop, the readout, the LR schedules.

:func:`run_subset_sgd` is the reproducibility contract (change its RNG use and every released
cell becomes unreproducible); :func:`per_query_margin` is both the margins readout and TRAK's
target, which makes ``g`` and ``y`` comparable; :func:`triangular_lr` is the DAWNBench/TRAK
schedule of the image recipe; :func:`linear_decay_lr` (text recipe) is shared so TracIn's
checkpoint weights read the multiplier the trainer stepped.
"""

import torch

__all__ = ["linear_decay_lr", "per_query_margin", "run_subset_sgd", "triangular_lr"]


def per_query_margin(logits, y_query, sentinel=float("-inf")):
    """Per-query correct-class margin ``L_c - logsumexp_{j != c} L_j`` (= log(p/(1-p)), does
    not saturate as p -> 1): ``logits`` (..., Q, C), ``y_query`` (Q,) -> (..., Q)."""
    ar = torch.arange(logits.shape[-2], device=logits.device)
    lc = logits[..., ar, y_query]
    other = logits.clone()
    other[..., ar, y_query] = sentinel
    return lc - torch.logsumexp(other, dim=-1)


def triangular_lr(epochs, peak_epoch, steps_per_epoch):
    """LambdaLR multiplier: 0 -> 1 at ``peak_epoch``, then linearly back to 0 at ``epochs``."""
    total = max(1, epochs * steps_per_epoch)
    peak = max(1, peak_epoch * steps_per_epoch)

    def f(step):
        if step < peak:
            return step / peak
        return max(0.0, (total - step) / max(1, total - peak))

    return f


def linear_decay_lr(epochs, steps_per_epoch):
    """LambdaLR multiplier: linear 1 -> 0 over all steps, no warmup (HF ``run_glue`` default)."""
    total = max(1, epochs * steps_per_epoch)

    def f(step):
        return max(0.0, 1.0 - step / total)

    return f


def run_subset_sgd(model, idx, epochs, batch, seed, opt, sched, device, batch_loss,
                   backward_step, on_epoch_end=None):
    """Per-subset SGD loop for the margins retrains: each epoch permutes the physical subset
    ``idx`` and steps over minibatches, all with one ``Generator(seed+1)``.  ``batch_loss(b, gen,
    epoch)`` owns forward + autocast (``epoch`` drives curricula such as the imagenet resolution
    ramp) and, for images, draws augmentation from the same ``gen``: that interleaving is the
    reproducibility contract.  ``backward_step(loss)`` owns backward + optimizer step, and
    ``on_epoch_end(epoch, model)`` (the TracIn checkpoint hook) must draw nothing from ``gen``, so
    hooked and bare runs are bit-identical."""
    n_sub = int(idx.numel())
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1)
    model.train()
    for epoch in range(epochs):
        perm = idx[torch.randperm(n_sub, device=device, generator=gen)]
        for s in range(0, n_sub, batch):
            b = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(b, gen, epoch)
            backward_step(loss)
            sched.step()
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
