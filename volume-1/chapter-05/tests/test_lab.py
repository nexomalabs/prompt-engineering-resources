"""Tests for Lab 5 Part A. Run against solution/ in CI. No network, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solution"))

from lab import (  # noqa: E402
    KERNELS,
    convolve2d,
    evaluate,
    extract_features,
    max_pool,
    relu,
    shifted_dataset,
)


@pytest.fixture(scope="module")
def digits():
    d = load_digits()
    return d.images, d.target


class TestConvolve:
    def test_output_shape_valid(self):
        assert convolve2d(np.ones((8, 8)), np.ones((3, 3))).shape == (6, 6)

    def test_output_shape_with_padding(self):
        assert convolve2d(np.ones((8, 8)), np.ones((3, 3)), padding=1).shape == (8, 8)

    def test_identity_kernel_returns_the_image(self):
        img = np.arange(25.0).reshape(5, 5)
        identity = np.zeros((3, 3))
        identity[1, 1] = 1.0
        assert np.allclose(convolve2d(img, identity, padding=1), img)

    def test_matches_a_naive_loop(self):
        rng = np.random.default_rng(0)
        img, k = rng.normal(size=(9, 7)), rng.normal(size=(3, 3))
        got = convolve2d(img, k)
        want = np.zeros((7, 5))
        for i in range(7):
            for j in range(5):
                want[i, j] = (img[i : i + 3, j : j + 3] * k).sum()
        assert np.allclose(got, want)

    def test_vertical_kernel_finds_a_vertical_edge(self):
        img = np.zeros((7, 7))
        img[:, 4:] = 1.0  # a vertical step
        out = np.abs(convolve2d(img, KERNELS["vertical edge"]))
        assert out.max() > np.abs(convolve2d(img, KERNELS["horizontal edge"])).max()

    def test_rejects_oversized_kernel(self):
        with pytest.raises(ValueError):
            convolve2d(np.ones((3, 3)), np.ones((5, 5)))

    def test_rejects_wrong_dimensions(self):
        with pytest.raises(ValueError):
            convolve2d(np.ones(9), np.ones((3, 3)))


class TestReluAndPool:
    def test_relu(self):
        assert np.array_equal(relu(np.array([-1.0, 0.0, 2.0])), np.array([0.0, 0.0, 2.0]))

    def test_max_pool_halves_dimensions(self):
        assert max_pool(np.ones((8, 8)), 2).shape == (4, 4)

    def test_max_pool_takes_the_block_maximum(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert max_pool(x, 2)[0, 0] == 4.0

    def test_max_pool_discards_trailing_rows(self):
        assert max_pool(np.ones((7, 7)), 2).shape == (3, 3)

    def test_max_pool_rejects_oversized_window(self):
        with pytest.raises(ValueError):
            max_pool(np.ones((2, 2)), 4)


class TestFeatures:
    def test_feature_vector_length_is_stable(self, digits):
        images, _ = digits
        assert len({len(extract_features(im)) for im in images[:20]}) == 1

    def test_features_are_translation_tolerant(self):
        """The property the whole lab exists to demonstrate.

        Compared as *relative* change, because raw pixels and convolutional
        features are on different scales and their absolute L1 distances are
        not comparable. A single 2x2 pool gives partial invariance, not total:
        a two-pixel shift still moves the pooled map by one cell, so the
        features change — just less than the pixels do.
        """
        img = np.zeros((12, 12))
        img[3:8, 3:8] = np.arange(25.0).reshape(5, 5)
        moved = np.zeros((12, 12))
        moved[5:10, 5:10] = np.arange(25.0).reshape(5, 5)

        def relative(a, b):
            return np.abs(a - b).sum() / (np.abs(a).sum() + np.abs(b).sum())

        raw_change = relative(img.ravel(), moved.ravel())
        feat_change = relative(extract_features(img), extract_features(moved))
        assert feat_change < raw_change, (
            f"features moved {feat_change:.3f} against pixels {raw_change:.3f}"
        )


class TestConclusion:
    def test_convolution_wins_once_digits_move(self, digits):
        """If this ever fails, the lab no longer makes its point."""
        images, y = digits
        shifted = shifted_dataset(images)
        raw_acc = evaluate(shifted.reshape(len(shifted), -1), y)
        conv_acc = evaluate(np.stack([extract_features(im) for im in shifted]), y)
        assert conv_acc > raw_acc + 0.15, (
            f"raw {raw_acc:.3f} vs conv {conv_acc:.3f} — translation invariance "
            f"is no longer visible"
        )

    def test_raw_pixels_collapse_when_shifted(self, digits):
        images, y = digits
        centred = evaluate(images.reshape(len(images), -1), y)
        shifted = shifted_dataset(images)
        moved = evaluate(shifted.reshape(len(shifted), -1), y)
        assert centred - moved > 0.3
