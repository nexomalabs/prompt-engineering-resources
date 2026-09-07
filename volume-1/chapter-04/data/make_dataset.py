"""Deterministic two-spiral dataset for Lab 4.

Two interleaved spirals are not linearly separable, which is the point: a
single-layer model cannot solve this, and a network with one hidden layer can.
That contrast is what the lab demonstrates.

Generated rather than downloaded so the lab runs offline and the numbers in the
README stay stable.
"""

from __future__ import annotations

import numpy as np

SEED = 20260904


def make_spirals(n_per_class: int = 300, noise: float = 0.22, seed: int = SEED):
    """Return (X, y) with X of shape (2n, 2) and y in {0, 1}."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * 2, 2))
    y = np.zeros(n_per_class * 2, dtype=int)
    for cls in (0, 1):
        t = np.linspace(0.4, 3.2, n_per_class)
        r = t
        theta = t * 2.4 + cls * np.pi + rng.normal(0, noise, n_per_class)
        idx = slice(cls * n_per_class, (cls + 1) * n_per_class)
        X[idx, 0] = r * np.cos(theta)
        X[idx, 1] = r * np.sin(theta)
        y[idx] = cls
    # Standardise. Unscaled inputs make gradient descent behave badly, which is
    # itself a lesson but not this lab's lesson.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    order = rng.permutation(len(X))
    return X[order], y[order]


def train_test_split(X, y, test_frac: float = 0.25, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


if __name__ == "__main__":
    X, y = make_spirals()
    print(f"{len(X)} points, {X.shape[1]} features, classes {sorted(set(y.tolist()))}")
    print(f"class balance: {int((y == 0).sum())} / {int((y == 1).sum())}")
