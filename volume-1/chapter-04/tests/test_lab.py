"""Tests for Lab 4. Run against solution/ in CI. No network, no API key."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "solution"))

from lab import (  # noqa: E402
    LinearModel,
    TwoLayerNet,
    accuracy,
    cross_entropy,
    relu,
    softmax,
    train,
)
from make_dataset import make_spirals, train_test_split  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y = make_spirals()
    return train_test_split(X, y)


class TestPrimitives:
    def test_relu_clips_negatives(self):
        assert np.array_equal(relu(np.array([-2.0, 0.0, 3.0])), np.array([0.0, 0.0, 3.0]))

    def test_softmax_rows_sum_to_one(self):
        p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_softmax_is_overflow_safe(self):
        """Without the max-subtraction trick this returns nan."""
        p = softmax(np.array([[1000.0, 1001.0]]))
        assert np.all(np.isfinite(p))
        assert np.allclose(p.sum(), 1.0)

    def test_cross_entropy_rewards_confidence(self):
        confident = cross_entropy(np.array([[0.99, 0.01]]), np.array([0]))
        unsure = cross_entropy(np.array([[0.50, 0.50]]), np.array([0]))
        assert confident < unsure

    def test_cross_entropy_finite_when_certain_and_wrong(self):
        assert np.isfinite(cross_entropy(np.array([[1.0, 0.0]]), np.array([1])))


class TestGradients:
    def test_backward_matches_numerical_gradient(self):
        """The check that actually proves backpropagation is correct."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 2))
        y = rng.integers(0, 2, 8)
        net = TwoLayerNet(n_hidden=5)

        probs, cache = net.forward(X)
        dW1 = net.backward(cache, probs, y)[0]

        eps = 1e-6
        i, j = 1, 3
        original = net.W1[i, j]
        net.W1[i, j] = original + eps
        loss_hi = cross_entropy(net.forward(X)[0], y)
        net.W1[i, j] = original - eps
        loss_lo = cross_entropy(net.forward(X)[0], y)
        net.W1[i, j] = original

        numerical = (loss_hi - loss_lo) / (2 * eps)
        assert numerical == pytest.approx(dW1[i, j], abs=1e-6)


class TestTraining:
    def test_loss_decreases(self, data):
        X_train, _, y_train, _ = data
        history = train(TwoLayerNet(), X_train, y_train, epochs=200, verbose=False)
        assert history[-1] < history[0] * 0.5

    def test_network_learns_the_spirals(self, data):
        X_train, X_test, y_train, y_test = data
        net = TwoLayerNet(n_hidden=32)
        train(net, X_train, y_train, verbose=False)
        assert accuracy(net, X_test, y_test) > 0.90

    def test_linear_model_cannot(self, data):
        """The lab's whole point: this must fail where the network succeeds."""
        X_train, X_test, y_train, y_test = data
        lin = LinearModel()
        train(lin, X_train, y_train, verbose=False)
        assert accuracy(lin, X_test, y_test) < 0.70

    def test_reproducible(self, data):
        X_train, _, y_train, _ = data
        a = train(TwoLayerNet(), X_train, y_train, epochs=50, verbose=False)
        b = train(TwoLayerNet(), X_train, y_train, epochs=50, verbose=False)
        assert a == b

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_divergence_raises_rather_than_producing_nan(self, data):
        X_train, _, y_train, _ = data
        with pytest.raises(FloatingPointError):
            train(TwoLayerNet(), X_train, y_train, epochs=500, lr=5e4, verbose=False)
