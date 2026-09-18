"""Tests for Lab 7 Part A. Run against solution/ in CI. Standard library only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solution"))

from lab import (  # noqa: E402
    CORPUS,
    END,
    SAMPLES,
    count_pairs,
    decode,
    encode,
    encode_word,
    merge_pair,
    train_bpe,
    word_frequencies,
)


@pytest.fixture(scope="module")
def trained():
    return train_bpe(CORPUS, 40)


class TestWordFrequencies:
    def test_appends_end_marker(self):
        assert word_frequencies(["ab"]) == {("a", "b", END): 1}

    def test_counts_repeats(self):
        assert word_frequencies(["ab", "ab"])[("a", "b", END)] == 2


class TestCountPairs:
    def test_counts_adjacent_pairs(self):
        pairs = count_pairs({("a", "b", "c"): 1})
        assert pairs[("a", "b")] == 1 and pairs[("b", "c")] == 1

    def test_weights_by_word_frequency(self):
        assert count_pairs({("a", "b"): 7})[("a", "b")] == 7

    def test_single_symbol_word_has_no_pairs(self):
        assert count_pairs({("a",): 5}) == {}


class TestMergePair:
    def test_joins_the_pair(self):
        assert merge_pair(("a", "b"), {("a", "b", "c"): 1}) == {("ab", "c"): 1}

    def test_does_not_remerge_its_own_output(self):
        """Advancing by 1 instead of 2 would produce ('aaa',) here."""
        assert merge_pair(("a", "a"), {("a", "a", "a", "a"): 1}) == {("aa", "aa"): 1}

    def test_leaves_unrelated_symbols_alone(self):
        assert merge_pair(("x", "y"), {("a", "b"): 1}) == {("a", "b"): 1}

    def test_combines_words_that_become_identical(self):
        out = merge_pair(("a", "b"), {("a", "b"): 2, ("ab",): 3})
        assert out == {("ab",): 5}


class TestTraining:
    def test_is_deterministic(self):
        assert train_bpe(CORPUS, 20)[0] == train_bpe(CORPUS, 20)[0]

    def test_more_merges_never_increases_token_count(self):
        sample = SAMPLES["common English"]
        counts = [len(encode(sample, train_bpe(CORPUS, n)[0])) for n in (0, 5, 10, 20, 40)]
        assert counts == sorted(counts, reverse=True)

    def test_learns_whole_frequent_words(self, trained):
        """'the' and 'design' are frequent enough to become single tokens."""
        merges, vocab = trained
        assert "the" + END in vocab
        assert any(v.startswith("design") for v in vocab)

    def test_stops_when_no_pair_repeats(self):
        merges, _ = train_bpe(["abc"], 100)
        assert merges == []

    def test_rejects_negative_merges(self):
        with pytest.raises(ValueError):
            train_bpe(CORPUS, -1)


class TestEncoding:
    def test_untrained_encoding_is_per_character(self):
        assert encode_word("cat", []) == ["c", "a", "t", END]

    def test_frequent_word_becomes_one_token(self, trained):
        merges, _ = trained
        assert encode_word("the", merges) == ["the" + END]

    def test_unseen_word_falls_back_to_characters(self, trained):
        merges, _ = trained
        assert len(encode_word("zzzz", merges)) >= 4

    def test_trained_text_compresses_better_than_unseen(self, trained):
        """The lab's central claim, measured."""
        merges, _ = trained
        seen = SAMPLES["common English"]
        unseen = SAMPLES["unseen words"]
        seen_ratio = len(seen) / len(encode(seen, merges))
        unseen_ratio = len(unseen) / len(encode(unseen, merges))
        assert seen_ratio > unseen_ratio * 3


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "the engineer designs the system",
            "quixotic zephyr",
            "a",
            "design design design",
        ],
    )
    def test_decode_inverts_encode(self, trained, text):
        merges, _ = trained
        assert decode(encode(text, merges)) == text

    def test_whitespace_is_normalised_not_preserved(self, trained):
        """A documented limitation: encode() splits on whitespace, so runs of
        spaces collapse. Real tokenizers encode whitespace as part of a token.
        Part B shows the difference."""
        merges, _ = trained
        assert decode(encode("the     engineer", merges)) == "the engineer"
