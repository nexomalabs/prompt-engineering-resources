"""Lab 6 Part A — Attention from scratch (starter).

Implement scaled dot-product attention with numpy, then read the patterns that
hand-designed heads produce.

This file RUNS as supplied. Attention returns uniform weights, so every heatmap
is flat. Complete the three TODOs and the patterns appear.

    python src/lab.py
"""

from __future__ import annotations

import numpy as np

SEED = 20260904
SENTENCE = ["the", "cat", "sat", "on", "the", "mat"]
_FP = dict(divide="ignore", over="ignore", invalid="ignore")


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    # TODO 1: subtract the max along `axis`, exponentiate, divide by the sum.
    # Skipping the max-subtraction overflows to nan on large scores.
    return np.ones_like(z) / z.shape[axis]


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """Return (output, weights). weights[i, j] is how much token i attends to j."""
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must share their last dimension")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same number of tokens")
    d_k = Q.shape[-1]

    # TODO 2: build the score matrix.
    #   scores = Q @ K.T
    #   if scale: scores = scores / np.sqrt(d_k)
    #   if mask is not None: scores = np.where(mask, scores, -np.inf)
    with np.errstate(**_FP):
        scores = np.zeros((Q.shape[0], K.shape[0]))

    weights = softmax(scores, axis=-1)
    with np.errstate(**_FP):
        return weights @ V, weights


def causal_mask(n: int) -> np.ndarray:
    """True where attention is allowed: token i may see tokens j <= i."""
    # TODO 3: lower-triangular boolean matrix. One numpy call.
    return np.ones((n, n), dtype=bool)


def token_embeddings(tokens, d_model=8, seed=SEED):
    rng = np.random.default_rng(seed)
    vocab = {w: rng.normal(0, 1, d_model) for w in dict.fromkeys(tokens)}
    return np.stack([vocab[w] for w in tokens])


def heatmap(weights, tokens, width=6) -> str:
    shades = " .:-=+*#%@"
    lines = [" " * (width + 1) + "".join(f"{t[:width]:>{width}}" for t in tokens)]
    for i, row in enumerate(weights):
        cells = "".join(
            f"{shades[min(int(v * len(shades)), len(shades) - 1)] * 2:>{width}}" for v in row
        )
        lines.append(f"{tokens[i][:width]:>{width}} {cells}")
    return "\n".join(lines)


def main() -> int:
    tokens = SENTENCE
    n, d = len(tokens), 8
    X = token_embeddings(tokens, d)
    print(f"Sentence: {' '.join(tokens)}\n")

    print("Content-matching head:")
    _, w = scaled_dot_product_attention(X, X, X)
    print(heatmap(w, tokens))
    print(f"\n'the'(0) attends to 'the'(4) with weight {w[0, 4]:.3f}")
    print("Expect about 0.48 once TODO 1 and 2 are done. A flat 0.167 means")
    print("softmax or the scores are still stubbed.\n")

    print("With a causal mask:")
    _, wc = scaled_dot_product_attention(X, X, X, mask=causal_mask(n))
    print(heatmap(wc, tokens))
    print(f"\nWeight on future tokens: {wc[np.triu_indices(n, k=1)].sum():.6f}")
    print("Expect 0.000000 once TODO 3 is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
