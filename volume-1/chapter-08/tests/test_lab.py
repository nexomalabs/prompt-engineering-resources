"""Tests for Lab 8. Run against solution/ in CI, replaying recorded fixtures.
No API key, no network — see fixtures/chapter-08/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT.parent / "shared"))

os.environ.setdefault("NEXOMA_LAB_MODE", "fixture")

from lab import (  # noqa: E402
    FIXTURES_DIR, MODEL, PROMPT, SAMPLES_PER_TEMPERATURE, TEMPERATURES,
    cross_sample_diversity, distinct_count, run_experiment,
)
from nexoma_labs.client import LabClient  # noqa: E402


@pytest.fixture(scope="module")
def results():
    client = LabClient(FIXTURES_DIR, MODEL)
    return run_experiment(client)


class TestDiversityMetric:
    def test_identical_samples_have_zero_diversity(self):
        assert cross_sample_diversity(["the cat sat"] * 5) == 0.0

    def test_disjoint_samples_have_diversity_one(self):
        assert cross_sample_diversity(["aa bb", "cc dd"]) == pytest.approx(1.0)

    def test_single_sample_is_defined_as_zero(self):
        assert cross_sample_diversity(["only one"]) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        d = cross_sample_diversity(["the cat sat", "the dog sat"])
        assert 0.0 < d < 1.0

    def test_is_case_insensitive(self):
        assert cross_sample_diversity(["The Cat", "the cat"]) == 0.0


class TestFixtureReplay:
    def test_all_five_temperatures_present(self, results):
        assert set(results.keys()) == set(TEMPERATURES)

    def test_five_samples_per_temperature(self, results):
        for samples in results.values():
            assert len(samples) == SAMPLES_PER_TEMPERATURE

    def test_zero_temperature_is_deterministic(self, results):
        """The lab's first claim: T=0.0 should give identical output."""
        assert distinct_count(results[0.0]) == 1

    def test_replay_is_exact_and_repeatable(self):
        """Fixture mode must return the same content on every run."""
        client = LabClient(FIXTURES_DIR, MODEL)
        first = [client.ask(PROMPT, temperature=0.7, max_tokens=100, seed=i).content
                 for i in range(5)]
        second = [client.ask(PROMPT, temperature=0.7, max_tokens=100, seed=i).content
                  for i in range(5)]
        assert first == second


class TestDiversityTrend:
    def test_diversity_rises_overall(self, results):
        """The lab's central claim. Checked end-to-end, not step-to-step,
        because 5 samples is too few for perfect monotonicity — the README
        says so and a stricter test here would be dishonest about that."""
        lo, hi = TEMPERATURES[0], TEMPERATURES[-1]
        diversity = {t: cross_sample_diversity(s) for t, s in results.items()}
        assert diversity[hi] > diversity[lo] + 0.5

    def test_zero_temperature_has_lowest_diversity(self, results):
        diversity = {t: cross_sample_diversity(s) for t, s in results.items()}
        assert diversity[0.0] == min(diversity.values())

    def test_no_fixture_is_an_empty_string(self, results):
        for samples in results.values():
            assert all(s.strip() for s in samples)
