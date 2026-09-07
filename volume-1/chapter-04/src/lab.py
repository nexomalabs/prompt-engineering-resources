"""Lab 4 — Building a Neural Network (starter).

Implement a two-layer network with numpy only: forward pass, cross-entropy loss,
and backpropagation. No framework. The point is that none of it is magic.

This file RUNS as supplied. The network is untrained, so it scores about 0.50.
Complete the four TODOs and it should reach about 0.98.

    python src/lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from make_dataset import SEED, make_spirals, train_test_split  # noqa: E402

_FP = dict(divide="ignore", over="ignore", invalid="ignore")


def relu(z):
    # TODO 1: return z where positive, 0 elsewhere. One numpy call.
    return z


def softmax(z):
    # TODO 2: exponentiate and normalise across axis 1.
    # Subtract z.max(axis=1, keepdims=True) first, or large logits overflow
    # to inf and every loss becomes nan.
    return np.ones_like(z) / z.shape[1]


def cross_entropy(probs, y):
    n = len(y)
    return -np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean()


class TwoLayerNet:
    def __init__(self, n_in=2, n_hidden=32, n_out=2, seed=SEED):
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/fan_in stops activations shrinking to
        # zero as they pass through a ReLU layer.
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        with np.errstate(**_FP):
            z1 = X @ self.W1 + self.b1
            a1 = relu(z1)
            z2 = a1 @ self.W2 + self.b2
        return softmax(z2), (X, z1, a1)

    def backward(self, cache, probs, y):
        X, z1, a1 = cache
        n = len(y)
        # TODO 3: for softmax with cross-entropy, dL/dz2 collapses to
        # (probs - onehot(y)) / n. Build it, then propagate backwards.
        #
        #   dz2 = probs.copy(); dz2[np.arange(n), y] -= 1.0; dz2 /= n
        #   dW2 = a1.T @ dz2            ; db2 = dz2.sum(axis=0)
        #   da1 = dz2 @ self.W2.T
        #   dz1 = da1 * (z1 > 0)        # ReLU passes gradient only where active
        #   dW1 = X.T @ dz1             ; db1 = dz1.sum(axis=0)
        return (
            np.zeros_like(self.W1),
            np.zeros_like(self.b1),
            np.zeros_like(self.W2),
            np.zeros_like(self.b2),
        )

    def step(self, grads, lr):
        # TODO 4: subtract lr * gradient from each parameter.
        pass

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def train(model, X, y, epochs=400, lr=0.5, log_every=100, verbose=True):
    history = []
    for epoch in range(1, epochs + 1):
        probs, cache = model.forward(X)
        loss = cross_entropy(probs, y)
        if not np.isfinite(loss):
            raise FloatingPointError(f"loss became {loss} at epoch {epoch}; lower the learning rate")
        model.step(model.backward(cache, probs, y), lr)
        history.append(loss)
        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}   loss {loss:.4f}   train acc {accuracy(model, X, y):.3f}")
    return history


def main() -> int:
    X, y = make_spirals()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Dataset: two interleaved spirals, {len(X)} points")
    print(f"Training points: {len(X_train)}")
    print(f"Test points:     {len(X_test)}")
    print()
    net = TwoLayerNet(n_hidden=32)
    train(net, X_train, y_train)
    print()
    print(f"Train accuracy: {accuracy(net, X_train, y_train):.3f}")
    print(f"Test accuracy:  {accuracy(net, X_test, y_test):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
