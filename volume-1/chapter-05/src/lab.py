"""Lab 5 Part A — Convolution from scratch (starter).

Implement 2-D convolution, ReLU, and max-pooling with numpy, then measure what
convolutional features buy you over raw pixels.

This file RUNS as supplied. The operations are stubs, so convolutional features
are meaningless and score near chance. Complete the three TODOs.

    python src/lab.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260904
_FP = dict(divide="ignore", over="ignore", invalid="ignore")

KERNELS = {
    "vertical edge":   np.array([[-1.0, 0.0, 1.0]] * 3),
    "horizontal edge": np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    "blur":            np.ones((3, 3)) / 9.0,
    "sharpen":         np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
}


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """Valid 2-D cross-correlation. Output is (H + 2p - kh + 1, W + 2p - kw + 1)."""
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve2d expects 2-D image and kernel")
    if padding:
        image = np.pad(image, padding, mode="constant")
    kh, kw = kernel.shape
    h, w = image.shape
    if kh > h or kw > w:
        raise ValueError("kernel is larger than the padded image")
    # TODO 1: slide `kernel` over `image` and sum the elementwise products at
    # each position. Nested loops are fine and clearest to start with:
    #
    #   out = np.zeros((h - kh + 1, w - kw + 1))
    #   for i in range(out.shape[0]):
    #       for j in range(out.shape[1]):
    #           out[i, j] = (image[i:i+kh, j:j+kw] * kernel).sum()
    #   return out
    #
    # Once it works, try np.lib.stride_tricks.sliding_window_view with einsum.
    return np.zeros((h - kh + 1, w - kw + 1))


def relu(x: np.ndarray) -> np.ndarray:
    # TODO 2: zero out negatives.
    return x


def max_pool(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Non-overlapping max pooling. Trailing rows and columns are discarded."""
    if size < 1:
        raise ValueError("pool size must be at least 1")
    h, w = x.shape
    h2, w2 = h // size, w // size
    if h2 == 0 or w2 == 0:
        raise ValueError("pool size is larger than the input")
    # TODO 3: take the maximum of each non-overlapping size x size block.
    #
    #   trimmed = x[: h2 * size, : w2 * size]
    #   return trimmed.reshape(h2, size, w2, size).max(axis=(1, 3))
    return np.zeros((h2, w2))


def extract_features(image: np.ndarray) -> np.ndarray:
    """conv -> relu -> pool for each kernel, flattened into one feature vector."""
    maps = [max_pool(relu(convolve2d(image, k, padding=1)), 2) for k in KERNELS.values()]
    return np.concatenate([m.ravel() for m in maps])


def evaluate(X: np.ndarray, y: np.ndarray) -> float:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    with np.errstate(**_FP):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
        model.fit(X_tr, y_tr)
        return accuracy_score(y_te, model.predict(X_te))


def main() -> int:
    digits = load_digits()
    images, y = digits.images, digits.target
    raw = images.reshape(len(images), -1)
    conv = np.stack([extract_features(im) for im in images])
    print(f"raw pixels     test accuracy {evaluate(raw, y):.3f}")
    print(f"conv features  test accuracy {evaluate(conv, y):.3f}")
    print()
    print("Once the TODOs are done, run solution/lab.py to see the shifted-digit")
    print("comparison, which is where convolution earns its keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
