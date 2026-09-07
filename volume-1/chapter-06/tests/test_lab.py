"""Tests for Lab 6 Part A. Run against solution/ in CI. No network, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solution"))

from lab import (  # noqa: E402
    SENTENCE, causal_mask, demo_scaling, entropy, heatmap, multi_head_attention,
    positional_encoding, scaled_dot_product_attention, softmax, token_embeddings,
)


@pytest.fixture(scope="module")
def X():
    return token_embeddings(SENTENCE, 8)


class TestSoftmax:
    def test_rows_sum_to_one(self):
        assert np.allclose(softmax(np.array([[1.0, 2.0, 3.0]])).sum(axis=-1), 1.0)

    def test_overflow_safe(self):
        p = softmax(np.array([[1000.0, 1001.0]]))
        assert np.all(np.isfinite(p)) and np.allclose(p.sum(), 1.0)

    def test_handles_negative_infinity(self):
        """Masked positions arrive as -inf and must become exactly zero."""
        p = softmax(np.array([[1.0, -np.inf, 2.0]]))
        assert p[0, 1] == 0.0 and np.allclose(p.sum(), 1.0)


class TestAttention:
    def test_weights_are_a_distribution(self, X):
        _, w = scaled_dot_product_attention(X, X, X)
        assert np.allclose(w.sum(axis=-1), 1.0)
        assert (w >= 0).all()

    def test_output_shape_follows_v(self, X):
        V = np.ones((len(X), 3))
        out, _ = scaled_dot_product_attention(X, X, V)
        assert out.shape == (len(X), 3)

    def test_output_is_a_weighted_average_of_v(self, X):
        V = np.arange(len(X) * 2.0).reshape(len(X), 2)
        out, w = scaled_dot_product_attention(X, X, V)
        assert np.allclose(out, w @ V)

    def test_identical_tokens_attend_to_each_other(self, X):
        """'the' appears at positions 0 and 4 with identical embeddings."""
        _, w = scaled_dot_product_attention(X, X, X)
        assert w[0, 4] > 0.3
        assert w[0, 4] == pytest.approx(w[0, 0], abs=1e-9)

    def test_rejects_mismatched_dimensions(self, X):
        with pytest.raises(ValueError):
            scaled_dot_product_attention(X, np.ones((6, 5)), X)

    def test_rejects_mismatched_k_and_v(self, X):
        with pytest.raises(ValueError):
            scaled_dot_product_attention(X, X, np.ones((3, 8)))


class TestCausalMask:
    def test_shape_and_triangularity(self):
        m = causal_mask(4)
        assert m.shape == (4, 4)
        assert m[3, 0] and not m[0, 3]

    def test_no_weight_leaks_forward(self, X):
        n = len(X)
        _, w = scaled_dot_product_attention(X, X, X, mask=causal_mask(n))
        assert w[np.triu_indices(n, k=1)].sum() == pytest.approx(0.0, abs=1e-12)

    def test_first_token_attends_only_to_itself(self, X):
        _, w = scaled_dot_product_attention(X, X, X, mask=causal_mask(len(X)))
        assert w[0, 0] == pytest.approx(1.0)


class TestScaling:
    def test_scaling_keeps_attention_diffuse(self):
        """The reason for the sqrt(d_k) divisor, measured rather than asserted."""
        scaled, unscaled = demo_scaling(d_k=64)
        assert entropy(scaled) > entropy(unscaled) * 3
        assert unscaled.max() > 0.99   # saturated to one-hot
        assert scaled.max() < 0.95

    def test_scaling_is_irrelevant_at_low_dimension(self):
        s, u = demo_scaling(d_k=1)
        assert abs(entropy(s) - entropy(u)) < 1.0


class TestPositional:
    def test_encoding_shape_and_range(self):
        pe = positional_encoding(6, 8)
        assert pe.shape == (6, 8)
        assert np.abs(pe).max() <= 1.0

    def test_positions_are_distinct(self):
        pe = positional_encoding(6, 8)
        assert len({tuple(np.round(r, 6)) for r in pe}) == 6


class TestMultiHead:
    def test_concatenates_head_outputs(self, X):
        d = X.shape[1]
        heads = [(np.eye(d), np.eye(d), np.eye(d)[:, :4]) for _ in range(3)]
        out, w = multi_head_attention(X, heads)
        assert out.shape == (len(X), 12)
        assert w.shape == (3, len(X), len(X))


class TestHeatmap:
    def test_renders_a_row_per_token(self, X):
        _, w = scaled_dot_product_attention(X, X, X)
        assert len(heatmap(w, SENTENCE).splitlines()) == len(SENTENCE) + 1
