"""The SOURCE paper's MNIST MLP pool task."""

import numpy as np
import torch
from torch import nn

from corrected_lds.banks.presets import MnistRecipe
from corrected_lds.pools.train import per_query_margin, run_subset_sgd

__all__ = ["make_model", "margins", "mnist_mlp", "train_mnist_model"]


def mnist_mlp(hidden=(512, 256, 128), num_classes=10) -> nn.Module:
    """Flatten -> three biased ReLU layers -> classifier (about 567k parameters)."""
    h1, h2, h3 = (int(v) for v in hidden)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, h3),
        nn.ReLU(),
        nn.Linear(h3, int(num_classes)),
    )


def make_model(recipe: MnistRecipe, num_classes=10, seed=None, device="cuda") -> nn.Module:
    if seed is not None:
        torch.manual_seed(int(seed))
    return mnist_mlp(recipe.hidden, num_classes).to(device=device, dtype=torch.float32)


def train_mnist_model(data, subset_idx, seed, recipe: MnistRecipe, device="cuda",
                      verbose=False, on_epoch_end=None) -> nn.Module:
    """Train one physical MNIST subset with the paper's constant-LR SGDm recipe."""
    model = make_model(recipe, data["meta"]["n_classes"], seed=seed, device=device)
    X, y = data["X_train"], data["y_train"]
    idx = torch.as_tensor(np.asarray(subset_idx, dtype=np.int64), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=recipe.peak_lr, momentum=recipe.momentum,
                          weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)
    lossfn = torch.nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)

    def batch_loss(batch, _gen, _epoch):
        return lossfn(model(X[batch]), y[batch])

    def backward_step(loss):
        loss.backward()
        opt.step()

    run_subset_sgd(model, idx, recipe.epochs, recipe.batch, seed, opt, sched, device,
                   batch_loss, backward_step, on_epoch_end=on_epoch_end)
    if verbose:
        with torch.no_grad():
            acc = (model(X[idx]).argmax(1) == y[idx]).float().mean().item()
        print(f"    [train] seed {seed} n_sub={len(idx)} -> train_acc {acc:.3f}", flush=True)
    return model


@torch.no_grad()
def margins(model, data, device="cuda", batch=2_000) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, int(data["meta"]["n_targets"]), int(batch)):
        stop = start + int(batch)
        logits = model(data["X_query"][start:stop])
        out.append(per_query_margin(logits, data["y_query"][start:stop]).double().cpu())
    return torch.cat(out).numpy()
