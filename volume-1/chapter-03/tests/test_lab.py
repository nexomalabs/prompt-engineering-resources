"""Tests for Lab 3. Run against solution/ in CI (LAB_PROJECT_STANDARD §6, tier 2).

No network and no API key: the corpus is generated locally from a fixed seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "solution"))

from build_corpus import HEAVY, VOCAB, build  # noqa: E402
from lab import top_features, train_and_evaluate  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return build()


@pytest.fixture(scope="module")
def result(corpus):
    return train_and_evaluate(*corpus)


class TestCorpus:
    def test_is_deterministic(self):
        assert build() == build()

    def test_seed_changes_output(self, corpus):
        assert build(seed=1)[0] != corpus[0]

    def test_balanced(self, corpus):
        _, labels, names = corpus
        counts = [labels.count(i) for i in range(len(names))]
        assert len(set(counts)) == 1, f"sections are unbalanced: {counts}"

    def test_no_section_exclusive_vocabulary(self):
        """The teaching property: every section draws from one shared pool."""
        for section, heavy in HEAVY.items():
            assert set(heavy) <= set(VOCAB), f"{section} uses words outside VOCAB"

    def test_headlines_are_short(self, corpus):
        texts, _, _ = corpus
        assert max(len(t.split()) for t in texts) <= 7


class TestModel:
    def test_beats_random_substantially(self, result):
        assert result["accuracy"] > 0.80, "pipeline is broken"

    def test_is_not_perfect(self, result):
        """A perfect score means the task is too easy to teach evaluation from."""
        assert result["accuracy"] < 0.99, (
            "task has become trivial; the confusion matrix will be empty "
            "and the chapter's precision/recall material will have nothing to work with"
        )

    def test_confusion_matrix_has_real_errors(self, result):
        cm = result["confusion"]
        off_diagonal = cm.sum() - cm.trace()
        assert off_diagonal >= 4, f"only {off_diagonal} errors — too few to examine"

    def test_train_test_split_is_disjoint_in_size(self, result, corpus):
        assert result["n_train"] + result["n_test"] == len(corpus[0])

    def test_reproducible(self, corpus):
        a = train_and_evaluate(*corpus)["accuracy"]
        b = train_and_evaluate(*corpus)["accuracy"]
        assert a == b

    def test_learned_terms_are_plausible(self, result):
        """The model should key on each section's heavy words, not noise."""
        top = top_features(result, k=5)
        for section, terms in top.items():
            overlap = set(terms) & set(HEAVY[section])
            assert overlap, f"{section} learned {terms}, none of which are its heavy terms"
