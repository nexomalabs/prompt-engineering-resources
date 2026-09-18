"""Lab 9 — Comparing LLM vs. Reasoning Models (starter).

Run three multi-step math problems under three conditions and compare accuracy.

This file RUNS as supplied against recorded fixtures, no API key needed.
`check_accuracy` is a stub that always returns False, so every condition
reports 0/3. Complete the one TODO.

    python src/lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from nexoma_labs.client import LabClient, Message, mode_banner  # noqa: E402

MODEL_STANDARD = "gpt-4o-mini"
MODEL_REASONING = "o1-mini"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "chapter-09"

PROBLEMS = [
    {
        "id": "P1",
        "problem": (
            "A train leaves Chicago at 9:00 AM traveling toward New York at 60 mph. "
            "A second train leaves New York at 10:00 AM traveling toward Chicago at "
            "80 mph. The cities are 790 miles apart. At what time do the two trains "
            "meet? Give your answer as a specific time (e.g., 3:13 PM)."
        ),
        "answer": "3:13 PM",
    },
    {
        "id": "P2",
        "problem": (
            "A bag contains 4 red balls and 6 blue balls. You draw two balls without "
            "replacement. What is the probability that both balls are red? Express "
            "your answer as a fraction in lowest terms."
        ),
        "answer": "2/15",
    },
    {
        "id": "P3",
        "problem": (
            "Alice can complete a project in 12 days. Bob can complete the same "
            "project in 8 days. If they work together, how many days will it take "
            "to complete the project? Round to two decimal places."
        ),
        "answer": "4.80 days",
    },
]


def check_accuracy(response: str, answer: str) -> bool:
    """Does the answer's digits/punctuation appear anywhere in the response?"""
    # TODO 1: strip `answer` down to digits and ./: characters, then check
    # whether that substring (or the full lowercased answer) appears in the
    # lowercased response.
    #
    #   key = "".join(c for c in answer if c.isdigit() or c in "./:")
    #   return key in response.lower() or answer.lower() in response.lower()
    return False


def call_standard(client: LabClient, problem: str, chain_of_thought: bool):
    suffix = (
        "\n\nThink step by step before giving your final answer."
        if chain_of_thought
        else "\n\nGive only the final answer with no explanation."
    )
    return client.complete([Message("user", problem + suffix)], temperature=0.0, max_tokens=500)


def call_reasoning(client: LabClient, problem: str) -> dict:
    request = {
        "model": MODEL_REASONING,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": 1500,
    }

    def call_live(req: dict) -> dict:
        raise NotImplementedError("record mode not implemented in the starter")

    return client.store.call(request, call_live)


def main() -> int:
    print(mode_banner("OpenAI"))
    client = LabClient(FIXTURES_DIR, MODEL_STANDARD)
    tally = {"direct": 0, "cot": 0, "reasoning": 0}

    for prob in PROBLEMS:
        direct = call_standard(client, prob["problem"], chain_of_thought=False)
        cot = call_standard(client, prob["problem"], chain_of_thought=True)
        reasoning = call_reasoning(client, prob["problem"])

        tally["direct"] += check_accuracy(direct.content, prob["answer"])
        tally["cot"] += check_accuracy(cot.content, prob["answer"])
        tally["reasoning"] += check_accuracy(reasoning["content"], prob["answer"])

        print(f"{prob['id']}: direct={direct.content[:50]!r}")

    print(f"\nAccuracy over {len(PROBLEMS)} problems:")
    for mode, correct in tally.items():
        print(f"  {mode:<10} {correct}/{len(PROBLEMS)}")
    print("\nExpect direct=0/3, cot=2/3, reasoning=3/3 once TODO 1 is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
