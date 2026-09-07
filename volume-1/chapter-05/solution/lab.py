"""Lab 5 Part A — Convolution from scratch (reference solution).

Implements 2-D convolution, ReLU, and max-pooling with numpy, then shows that
convolutional features beat raw pixels for image classification.

Part B (solution/resnet_demo.py) applies a real pre-trained ResNet-50. It needs
PyTorch and a network connection, so it is not part of the CI-gated suite.
Convolution is implemented here first so that Part B is not magic.
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

# Some numpy/BLAS builds raise spurious divide-by-zero, overflow and invalid
# flags on ordinary matmuls, including inside scikit-learn. See Lab 4.
_FP = dict(divide="ignore", over="ignore", invalid="ignore")

# Classic hand-designed kernels. A trained CNN learns kernels like these in its
# first layer rather than being given them.
KERNELS = {
    "vertical edge":   np.array([[-1.0, 0.0, 1.0]] * 3),
    "horizontal edge": np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    "blur":            np.ones((3, 3)) / 9.0,
    "sharpen":         np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
}


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """Valid 2-D cross-correlation, the operation deep learning calls convolution.

    Output size is (H + 2p - kh + 1, W + 2p - kw + 1).
    """
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve2d expects 2-D image and kernel")
    if padding:
        image = np.pad(image, padding, mode="constant")
    kh, kw = kernel.shape
    h, w = image.shape
    if kh > h or kw > w:
        raise ValueError("kernel is larger than the padded image")
    oh, ow = h - kh + 1, w - kw + 1
    # Build a sliding-window view, then contract it against the kernel in one
    # einsum. Equivalent to four nested loops, roughly two orders faster.
    windows = np.lib.stride_tricks.sliding_window_view(image, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)[:oh, :ow]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def max_pool(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Non-overlapping max pooling. Trailing rows and columns are discarded."""
    if size < 1:
        raise ValueError("pool size must be at least 1")
    h, w = x.shape
    h2, w2 = h // size, w // size
    if h2 == 0 or w2 == 0:
        raise ValueError("pool size is larger than the input")
    trimmed = x[: h2 * size, : w2 * size]
    return trimmed.reshape(h2, size, w2, size).max(axis=(1, 3))


def extract_features(image: np.ndarray) -> np.ndarray:
    """conv -> relu -> pool for each kernel, flattened into one feature vector.

    This is a single convolutional layer. A real CNN stacks many of these and
    learns the kernels instead of being handed them.
    """
    maps = []
    for kernel in KERNELS.values():
        maps.append(max_pool(relu(convolve2d(image, kernel, padding=1)), size=2))
    return np.concatenate([m.ravel() for m in maps])


def place_on_canvas(image: np.ndarray, canvas: int, dy: int, dx: int) -> np.ndarray:
    """Paste an 8x8 digit onto a larger canvas at a given offset."""
    out = np.zeros((canvas, canvas))
    h, w = image.shape
    out[dy : dy + h, dx : dx + w] = image
    return out


def shifted_dataset(images: np.ndarray, canvas: int = 12, seed: int = SEED):
    """Same digits, placed at random offsets on a larger canvas.

    Centred digits are the easy case: a raw-pixel classifier can memorise which
    pixel positions matter. Shift the digit and those positions stop meaning
    anything. Convolution slides the same kernel everywhere and pooling discards
    exact position, so convolutional features survive the shift. That invariance
    is what convolution actually buys, and it is invisible unless the data moves.
    """
    rng = np.random.default_rng(seed)
    margin = canvas - images.shape[1]
    out = np.zeros((len(images), canvas, canvas))
    for i, im in enumerate(images):
        out[i] = place_on_canvas(im, canvas, rng.integers(0, margin + 1), rng.integers(0, margin + 1))
    return out


def evaluate(X: np.ndarray, y: np.ndarray) -> float:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    # Standardise before fitting. Unscaled features make lbfgs converge slowly
    # or not at all, and the resulting comparison would measure the optimiser
    # rather than the features.
    with np.errstate(**_FP):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=SEED),
        )
        model.fit(X_tr, y_tr)
        return accuracy_score(y_te, model.predict(X_te))


def main() -> int:
    digits = load_digits()
    images, y = digits.images, digits.target
    print(f"Dataset: {len(images)} handwritten digits, 8x8 pixels, {len(set(y.tolist()))} classes")
    print()

    print("What each kernel responds to, on the first digit:")
    for name, kernel in KERNELS.items():
        out = relu(convolve2d(images[0], kernel, padding=1))
        print(f"  {name:<16} mean activation {out.mean():6.2f}   peak {out.max():6.2f}")
    print()

    shifted = shifted_dataset(images)

    rows = []
    for label, data in (("centred 8x8", images), ("shifted on 12x12 canvas", shifted)):
        raw = data.reshape(len(data), -1)
        conv = np.stack([extract_features(im) for im in data])
        rows.append((label, raw.shape[1], evaluate(raw, y), conv.shape[1], evaluate(conv, y)))

    print(f"{'dataset':<26}{'raw dims':>9}{'raw acc':>10}{'conv dims':>11}{'conv acc':>10}")
    print("-" * 66)
    for label, rd, ra, cd, ca in rows:
        print(f"{label:<26}{rd:>9}{ra:>10.3f}{cd:>11}{ca:>10.3f}")
    print("-" * 66)
    print()

    (_, _, ra_c, _, ca_c), (_, _, ra_s, _, ca_s) = rows
    lead_c, lead_s = ca_c - ra_c, ca_s - ra_s

    def verdict(lead: float) -> str:
        if abs(lead) < 0.02:
            return "the two are within noise of each other"
        return f"convolutional features lead by {lead:+.3f}" if lead > 0 else \
               f"raw pixels lead by {-lead:+.3f}"

    print(f"Centred digits: {verdict(lead_c)}.")
    print("  The digits never move, so a pixel index is already a reliable feature.")
    print("  Convolution has little to add when position is fixed.")
    print()
    print(f"Shifted digits: {verdict(lead_s)}.")
    print(f"  Raw pixels fall by {ra_c - ra_s:.3f}; convolutional features fall by {ca_c - ca_s:.3f}.")
    print()
    print("That difference is what convolution buys. The same kernel is applied at")
    print("every position and pooling discards exactly where it fired, so the")
    print("features barely change when the digit moves. A raw-pixel classifier has")
    print("to learn every position separately and mostly fails. Real images are")
    print("never perfectly centred, which is why CNNs displaced pixel classifiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
