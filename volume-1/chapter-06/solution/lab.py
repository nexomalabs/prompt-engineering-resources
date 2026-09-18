"""Lab 6 Part A — Attention from scratch (reference solution).

Implements scaled dot-product attention and multi-head attention in numpy, then
uses hand-designed heads to show that attention patterns are readable.

Part B (solution/bert_demo.py) extracts real attention from a pre-trained BERT.
It needs transformers and torch and downloads weights, so it is not CI-gated.
Attention is built by hand here first so that Part B is not magic.
"""

from __future__ import annotations

import numpy as np

SEED = 20260904
SENTENCE = ["the", "cat", "sat", "on", "the", "mat"]
_FP = dict(divide="ignore", over="ignore", invalid="ignore")


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """Return (output, weights).

    weights[i, j] is how much token i attends to token j. Each row sums to 1.
    """
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must share their last dimension")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same number of tokens")
    d_k = Q.shape[-1]
    with np.errstate(**_FP):
        scores = Q @ K.T
    # Divide by sqrt(d_k). Without it, dot products grow with dimension, softmax
    # saturates, and gradients vanish. Demonstrated in demo_scaling().
    if scale:
        scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    with np.errstate(**_FP):
        return weights @ V, weights


def causal_mask(n: int) -> np.ndarray:
    """True where attention is allowed: token i may see tokens j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))


def multi_head_attention(X, head_projections):
    """Run several heads and concatenate. head_projections is a list of (Wq, Wk, Wv)."""
    outputs, all_weights = [], []
    for Wq, Wk, Wv in head_projections:
        out, w = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
        outputs.append(out)
        all_weights.append(w)
    return np.concatenate(outputs, axis=-1), np.stack(all_weights)


def token_embeddings(tokens, d_model=8, seed=SEED):
    """One fixed vector per distinct word, so repeated words are identical."""
    rng = np.random.default_rng(seed)
    vocab = {w: rng.normal(0, 1, d_model) for w in dict.fromkeys(tokens)}
    return np.stack([vocab[w] for w in tokens])


def positional_encoding(n, d_model):
    """Sinusoidal positions, as in the original Transformer."""
    pos = np.arange(n)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((n, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def heatmap(weights, tokens, width=6) -> str:
    """ASCII attention heatmap. Rows attend to columns."""
    shades = " .:-=+*#%@"
    lines = [" " * (width + 1) + "".join(f"{t[:width]:>{width}}" for t in tokens)]
    for i, row in enumerate(weights):
        cells = "".join(
            f"{shades[min(int(v * len(shades)), len(shades) - 1)] * 2:>{width}}" for v in row
        )
        lines.append(f"{tokens[i][:width]:>{width}} {cells}")
    return "\n".join(lines)


def demo_scaling(d_k=64, n=6, seed=SEED):
    """Show what dividing by sqrt(d_k) is for."""
    rng = np.random.default_rng(seed)
    Q, K, V = (rng.normal(0, 1, (n, d_k)) for _ in range(3))
    _, scaled = scaled_dot_product_attention(Q, K, V, scale=True)
    _, unscaled = scaled_dot_product_attention(Q, K, V, scale=False)
    return scaled, unscaled


def entropy(weights):
    """Mean row entropy in bits. Low means peaked attention, high means diffuse."""
    w = np.clip(weights, 1e-12, None)
    return float((-(w * np.log2(w)).sum(axis=-1)).mean())


def main() -> int:
    tokens = SENTENCE
    n, d = len(tokens), 8
    X = token_embeddings(tokens, d)

    print(f"Sentence: {' '.join(tokens)}")
    print(f"Tokens: {n}   d_model: {d}")
    print()

    print("=" * 62)
    print("Head 1 — content matching (Wq = Wk = identity)")
    print("=" * 62)
    eye = np.eye(d)
    _, w_content = scaled_dot_product_attention(X @ eye, X @ eye, X)
    print(heatmap(w_content, tokens))
    print()
    print(f"'the' at position 0 attends to position 4 with weight {w_content[0, 4]:.3f}")
    print("Identical words have identical embeddings, so a content-matching head")
    print("links the two occurrences of 'the'. This is the mechanism behind")
    print("induction heads, which copy from earlier matching context.")
    print()

    print("=" * 62)
    print("Head 2 — previous token (built from positional encoding)")
    print("=" * 62)
    P = positional_encoding(n, d)
    # Query from position i carries the encoding of position i-1, so it matches
    # the key of the preceding token. np.roll wraps, which would let position 0
    # attend to the last token; the causal mask removes that artefact and leaves
    # position 0 attending to itself, which is the only thing it can do.
    Q_prev = np.roll(P, 1, axis=0) * 4.0
    _, w_prev = scaled_dot_product_attention(Q_prev, P * 4.0, X, mask=causal_mask(n))
    print(heatmap(w_prev, tokens))
    print()
    diag = float(np.mean([w_prev[i, i - 1] for i in range(1, n)]))
    print(f"Mean weight on the immediately preceding token: {diag:.3f}")
    print("Positional information alone produces a positional pattern. Real")
    print("models learn heads that look very like this one.")
    print()

    print("=" * 62)
    print("Causal masking")
    print("=" * 62)
    _, w_causal = scaled_dot_product_attention(X, X, X, mask=causal_mask(n))
    print(heatmap(w_causal, tokens))
    upper = w_causal[np.triu_indices(n, k=1)]
    print()
    print(f"Total weight on future tokens: {upper.sum():.6f}")
    print("A decoder cannot attend forwards. That constraint is what makes")
    print("next-token prediction a valid training objective.")
    print()

    print("=" * 62)
    print("Why divide by sqrt(d_k)")
    print("=" * 62)
    scaled, unscaled = demo_scaling()
    print(
        f"  scaled    mean row entropy {entropy(scaled):.3f} bits   max weight {scaled.max():.3f}"
    )
    print(
        f"  unscaled  mean row entropy {entropy(unscaled):.3f} bits   max weight {unscaled.max():.3f}"  # noqa: E501 — column alignment
    )
    print()
    print("At d_k = 64 the unscaled dot products are large, softmax saturates")
    print("towards one-hot, and the gradient through it nearly vanishes. Scaling")
    print("keeps attention distributed and trainable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
