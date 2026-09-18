"""Tests for Lab 10. Run against solution/ in CI, replaying recorded fixtures.
No API key, no network — see fixtures/chapter-10/.
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
    EX1_DOC,
    EX1_STRONG,
    EX1_WEAK,
    EX2_STRONG,
    EX2_USER,
    EX2_WEAK,
    EX3_STRONG,
    EX3_USER,
    EX3_WEAK,
    EX4_STRONG,
    EX4_WEAK,
    EX5_STRONG,
    EX5_TEXT,
    EX5_WEAK,
    EXERCISES,
    FIXTURES_DIR,
    MODEL,
    call,
    check_ex1,
    check_ex2,
    check_ex3,
    check_ex4,
    check_ex5,
    is_valid_json,
    sentence_count,
)
from nexoma_labs.client import LabClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return LabClient(FIXTURES_DIR, MODEL)


class TestSentenceCount:
    def test_counts_three_sentences(self):
        assert sentence_count("One. Two. Three.") == 3

    def test_dollar_figures_do_not_create_false_sentence_breaks(self):
        """The bug this function exists to avoid: '$4.2M' is not two sentences."""
        assert sentence_count("Revenue was $4.2M this quarter. Costs rose.") == 2

    def test_empty_string_is_zero(self):
        assert sentence_count("") == 0

    def test_no_terminal_punctuation_is_one_sentence(self):
        assert sentence_count("just some words") == 1


class TestIsValidJson:
    def test_valid_object(self):
        assert is_valid_json('{"a": 1}')

    def test_prose_is_not_json(self):
        assert not is_valid_json("The answer is yes.")

    def test_trailing_prose_after_json_is_not_valid(self):
        """A common near-miss: JSON followed by an explanatory sentence."""
        assert not is_valid_json('{"a": 1} — that is the classification.')


class TestExerciseChecks:
    """Each check function against hand-built weak/strong pairs, independent
    of the fixtures, so a fixture change cannot silently break the checker."""

    def test_ex1_detects_three_sentence_compliance(self):
        good = check_ex1("one sentence only", "First. Second. Third with $4.2M and $1.1M.")
        assert good["demonstrates_fix"]
        bad = check_ex1("x", "Only one sentence with $4.2M and $1.1M.")
        assert not bad["demonstrates_fix"]

    def test_ex2_requires_weak_to_fail_and_strong_to_pass(self):
        assert check_ex2("prose response", '{"intent": "request"}')["demonstrates_fix"]
        assert not check_ex2('{"intent": "x"}', '{"intent": "request"}')["demonstrates_fix"]

    def test_ex3_requires_grounding_and_no_lucky_guess(self):
        good = check_ex3("Standard plan at $12.99", "The Professional plan costs $20 per month.")
        assert good["demonstrates_fix"]

    def test_ex4_requires_weak_over_and_strong_under_the_limit(self):
        assert check_ex4("word " * 50, "word " * 30)["demonstrates_fix"]
        assert not check_ex4("word " * 30, "word " * 30)["demonstrates_fix"]

    def test_ex5_requires_full_schema_and_weak_non_json(self):
        strong = '{"action_items": [1,2,3], "attendees": [1,2,3], "decisions": ["a"]}'
        assert check_ex5("a bulleted list", strong)["demonstrates_fix"]


class TestFixtureReplay:
    def test_all_five_exercises_produce_output(self, client):
        for title, weak_prompt, strong_prompt, user, check in EXERCISES:
            weak = call(client, weak_prompt, user)
            strong = call(client, strong_prompt, user)
            assert weak.strip() and strong.strip()

    def test_replay_is_repeatable(self, client):
        a = call(client, EX1_WEAK, EX1_DOC)
        b = call(client, EX1_WEAK, EX1_DOC)
        assert a == b


class TestAllFiveExercisesDemonstrateTheirFix:
    """The lab's actual claim, checked end to end against the bundled fixtures."""

    def test_exercise_1(self, client):
        weak, strong = call(client, EX1_WEAK, EX1_DOC), call(client, EX1_STRONG, EX1_DOC)
        assert check_ex1(weak, strong)["demonstrates_fix"]

    def test_exercise_2(self, client):
        weak, strong = call(client, EX2_WEAK, EX2_USER), call(client, EX2_STRONG, EX2_USER)
        assert check_ex2(weak, strong)["demonstrates_fix"]

    def test_exercise_3(self, client):
        weak, strong = call(client, EX3_WEAK, EX3_USER), call(client, EX3_STRONG, EX3_USER)
        assert check_ex3(weak, strong)["demonstrates_fix"]

    def test_exercise_4(self, client):
        weak, strong = call(client, EX4_WEAK), call(client, EX4_STRONG)
        assert check_ex4(weak, strong)["demonstrates_fix"]

    def test_exercise_5(self, client):
        weak, strong = call(client, EX5_WEAK, EX5_TEXT), call(client, EX5_STRONG, EX5_TEXT)
        assert check_ex5(weak, strong)["demonstrates_fix"]
