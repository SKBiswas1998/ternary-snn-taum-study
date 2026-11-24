"""Data: MNIST in memory with a FIXED stratified 50k/10k train/val split (split seed 0)."""
import numpy as np
import torch

SPLIT_SEED = 0


def stratified_split(y, n_val, seed=SPLIT_SEED):
    """Return (train_idx, val_idx). Deterministic, class-proportional, disjoint."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    n_val_c = {c: int(round(len(per_class[c]) * n_val / len(y))) for c in classes}
    diff = n_val - sum(n_val_c.values())          # fix rounding so the total is exact
    order = sorted(classes, key=lambda c: -len(per_class[c]))
    k = 0
    while diff != 0:
        c = order[k % len(order)]
        step = 1 if diff > 0 else -1
        n_val_c[c] += step
        diff -= step
        k += 1
    val, train = [], []
    for c in classes:
        val.append(per_class[c][: n_val_c[c]])
        train.append(per_class[c][n_val_c[c]:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def load_mnist(root, n_val=10000):
    from torchvision import datasets                      # imported lazily (not needed for smoke tests)
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    x = tr.data.float().div(255.0).unsqueeze(1)           # same scaling as transforms.ToTensor()
    y = tr.targets.long()
    tr_i, va_i = stratified_split(y.numpy(), n_val)
    xte = te.data.float().div(255.0).unsqueeze(1)
    yte = te.targets.long()
    return dict(xtr=x[tr_i], ytr=y[tr_i], xva=x[va_i], yva=y[va_i], xte=xte, yte=yte)


def load_synthetic(n_train, seed=0):
    """Learnable fake data, ONLY for smoke tests and timing (never for results)."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.rand(10, 1, 28, 28, generator=g)

    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        x = (0.7 * protos[y] + 0.3 * torch.rand(n, 1, 28, 28, generator=gg)).clamp(0, 1)
        return x, y

    n_small = max(n_train // 5, 64)
    xtr, ytr = make(n_train, 1)
    xva, yva = make(n_small, 2)
    xte, yte = make(n_small, 3)
    return dict(xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte)
