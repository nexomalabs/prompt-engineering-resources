"""Lab 4 — Building a Neural Network (reference solution).

A two-layer network implemented with numpy only: forward pass, cross-entropy
loss, backpropagation, and gradient descent, all written out rather than called
from a framework. The point is that nothing here is magic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from make_dataset import SEED, make_spirals, train_test_split  # noqa: E402


# Some numpy/BLAS builds raise spurious divide-by-zero, overflow and invalid
# flags on ordinary matmuls; a plain random A @ B triggers them on numpy 2.0.2
# under Accelerate. Silencing them here would hide a genuine blow-up too, so
# train() asserts the loss stays finite instead. Suppress the false positive,
# keep the real check.
_FP = dict(divide="ignore", over="ignore", invalid="ignore")


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    # Subtract the row max before exponentiating. Without this, large logits
    # overflow to inf and the loss becomes nan.
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    n = len(y)
    # Clip to avoid log(0) when the model is confidently correct.
    return -np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean()


class LinearModel:
    """Softmax regression: no hidden layer, no non-linearity.

    Included as the baseline. It is the honest comparison for the two-layer
    network, and it cannot separate interleaved spirals however long it trains.
    """

    def __init__(self, n_in=2, n_out=2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, X):
        with np.errstate(**_FP):
            return softmax(X @ self.W + self.b), (X,)

    def backward(self, cache, probs, y):
        (X,) = cache
        n = len(y)
        dz = probs.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n
        with np.errstate(**_FP):
            return X.T @ dz, dz.sum(axis=0)

    def step(self, grads, lr):
        dW, db = grads
        self.W -= lr * dW
        self.b -= lr * db

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)


class TwoLayerNet:
    def __init__(self, n_in=2, n_hidden=32, n_out=2, seed=SEED):
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/fan_in keeps activations from shrinking
        # to zero through a ReLU layer.
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
        # dL/dz2 for softmax + cross-entropy collapses to (probs - onehot)/n.
        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n
        with np.errstate(**_FP):
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)      # ReLU passes gradient only where it was active
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

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
            raise FloatingPointError(
                f"loss became {loss} at epoch {epoch}. The model diverged — "
                f"lower the learning rate."
            )
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

    print("Baseline — softmax regression, no hidden layer:")
    linear = LinearModel()
    train(linear, X_train, y_train, epochs=400, lr=0.5, verbose=False)
    print(f"  test accuracy {accuracy(linear, X_test, y_test):.3f}")
    print()

    print("Two-layer network, 32 hidden units:")
    net = TwoLayerNet(n_hidden=32)
    history = train(net, X_train, y_train)
    print()
    print(f"Final training loss: {history[-1]:.4f}")
    print(f"Train accuracy:      {accuracy(net, X_train, y_train):.3f}")
    print(f"Test accuracy:       {accuracy(net, X_test, y_test):.3f}")
    print()
    print("The hidden layer is what makes the difference. A linear model cannot")
    print("separate interleaved spirals no matter how long it trains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
