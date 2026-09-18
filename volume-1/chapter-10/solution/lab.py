"""Lab 10 — Writing Your First Professional Prompts (reference solution).

Volume I capstone. Five exercises, each demonstrating one failure mode from
the taxonomy in Chapter 10 section 9: ambiguity, format, hallucination,
constraint violation, and missing structure. Each pairs a WEAK prompt against
a STRONG one on the same task, and a check function verifies the failure
actually happens and the fix actually fixes it.

Calls replay through the shared LabClient (LAB_PROJECT_STANDARD.md section 6):
no API key needed by default. See "Recording Your Own Fixtures" in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from nexoma_labs.client import LabClient, Message, mode_banner  # noqa: E402

MODEL = "gpt-4o-mini"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "chapter-10"


def call(client: LabClient, system: str, user: str = "") -> str:
    """Send a system+user pair, or just `system` alone as a user message if
    `user` is empty — matching how the chapter's own lab code is structured."""
    messages = (
        [Message("user", system)]
        if not user
        else [Message("system", system), Message("user", user)]
    )
    return client.complete(messages, temperature=0.0, max_tokens=600).content


# ─────────────────────────────────────────────────────────────────────
# Exercise 1: Ambiguity — Document Summarization
# ─────────────────────────────────────────────────────────────────────
EX1_DOC = (
    "The Q2 revenue for Nexoma Labs was $4.2M, up 18% year-over-year. "
    "Operating expenses were $3.1M, driven primarily by R&D ($1.8M) and "
    "sales and marketing ($0.9M). Net income was $1.1M, compared to $0.6M "
    "in Q2 of last year. Cash on hand is $6.4M. Headcount grew from 42 to 51."
)
EX1_WEAK = "Summarize this."
EX1_STRONG = (
    "You are a financial analyst writing a one-paragraph executive summary.\n\n"
    "Summarize the following financial report extract in exactly 3 sentences.\n"
    "Sentence 1: State the revenue and revenue growth.\n"
    "Sentence 2: State net income and the key expense driver.\n"
    "Sentence 3: Comment on cash position and headcount change.\n\n"
    "Use professional financial language. Do not add analysis not present in the data."
)


def sentence_count(text: str) -> int:
    """Count sentences by splitting on '. ' (period-then-space) or a trailing
    period at the end of the text, not on every bare period. A naive split on
    '.' breaks on any dollar figure like $4.2M, which this text is full of.
    """
    import re

    text = text.replace("\n", " ").strip()
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return len([p for p in parts if p.strip()])


def check_ex1(weak: str, strong: str) -> dict:
    strong_ok = sentence_count(strong) == 3 and "$4.2M" in strong and "$1.1M" in strong
    return {
        "weak_sentences": sentence_count(weak),
        "strong_sentences": sentence_count(strong),
        "demonstrates_fix": strong_ok,
    }


# ─────────────────────────────────────────────────────────────────────
# Exercise 2: Format — Intent Classification (JSON)
# ─────────────────────────────────────────────────────────────────────
EX2_USER = (
    "My invoice from last month has the wrong address and I need it corrected before I can pay."
)
EX2_WEAK = "Classify the intent of the user message."
EX2_STRONG = (
    "You are an intent classification API. Classify the user's message and return ONLY "
    "a JSON object with no additional text, prose, or code fences.\n\n"
    "Schema:\n"
    '{\n  "intent": "question" | "complaint" | "request" | "feedback" | "other",\n'
    '  "confidence": "high" | "medium" | "low",\n  "topic": string\n}\n\n'
    "Begin your response with { and end with }."
)


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def check_ex2(weak: str, strong: str) -> dict:
    weak_ok, strong_ok = is_valid_json(weak), is_valid_json(strong)
    return {
        "weak_is_valid_json": weak_ok,
        "strong_is_valid_json": strong_ok,
        "demonstrates_fix": (not weak_ok) and strong_ok,
    }


# ─────────────────────────────────────────────────────────────────────
# Exercise 3: Hallucination — Document-Grounded Q&A
# ─────────────────────────────────────────────────────────────────────
EX3_DOC = (
    "Nexoma Cloud Storage plans:\n"
    "- Starter: 50 GB, $5/month, no API access\n"
    "- Professional: 500 GB, $20/month, API access included\n"
    "- Enterprise: Unlimited storage, $150/month, dedicated support, SLA 99.99%"
)
EX3_USER = "What is the cheapest plan that includes API access and what does it cost?"
EX3_WEAK = "What is the cheapest plan with API access?"
EX3_STRONG = (
    "You are a product information assistant. Answer the user's question using ONLY "
    "the information in the document below. Do not use any external knowledge. "
    'If the answer is not in the document, say: "This information is not available '
    'in the provided documentation."\n\n'
    f"<document>\n{EX3_DOC}\n</document>"
)


def check_ex3(weak: str, strong: str) -> dict:
    strong_grounded = "professional" in strong.lower() and "$20" in strong
    weak_ungrounded = "professional" not in weak.lower() or "$20" not in weak
    return {
        "strong_is_grounded_in_document": strong_grounded,
        "weak_did_not_happen_to_match_the_real_answer": weak_ungrounded,
        "demonstrates_fix": strong_grounded and weak_ungrounded,
    }


# ─────────────────────────────────────────────────────────────────────
# Exercise 4: Constraint Violation — Length-Limited Description
# ─────────────────────────────────────────────────────────────────────
EX4_WEAK = "Write a product description for an AI-powered note-taking app."
EX4_STRONG = (
    "Write a product description for an AI-powered note-taking app.\n\n"
    "Requirements:\n"
    "- Maximum 40 words. Count carefully. Do not exceed 40 words.\n"
    "- Lead with the primary benefit.\n"
    "- Include one specific feature.\n"
    "- End with a call to action.\n"
    "- Do not include a headline — start directly with the description text."
)


def check_ex4(weak: str, strong: str) -> dict:
    weak_words, strong_words = len(weak.split()), len(strong.split())
    within, exceeds = strong_words <= 40, weak_words > 40
    return {
        "weak_word_count": weak_words,
        "strong_word_count": strong_words,
        "strong_within_limit": within,
        "weak_exceeds_limit": exceeds,
        "demonstrates_fix": within and exceeds,
    }


# ─────────────────────────────────────────────────────────────────────
# Exercise 5: Structured Extraction — Meeting Notes
# ─────────────────────────────────────────────────────────────────────
EX5_TEXT = (
    "Meeting Notes — Project Phoenix Kickoff\n"
    "Date: July 24, 2026\n"
    "Attendees: Sarah Chen (PM), David Okafor (Engineering Lead), Maria Santos (Design)\n\n"
    "Key decisions:\n"
    "- Launch target: October 15, 2026\n"
    "- MVP scope: Authentication, dashboard, and reporting modules only\n"
    "- Design system: Adopt Nexoma Design System v2\n\n"
    "Action items:\n"
    "- David to provide technical architecture draft by August 1\n"
    "- Maria to share initial wireframes by July 31\n"
    "- Sarah to schedule weekly sync starting July 28"
)
EX5_WEAK = "Extract the action items from this meeting note."
EX5_STRONG = (
    "Extract structured data from the following meeting notes. Return ONLY a JSON "
    "object with this exact schema. No prose, no code fences.\n\n"
    "{\n"
    '  "meeting_date": "YYYY-MM-DD",\n  "project": string,\n'
    '  "attendees": [{"name": string, "role": string}],\n'
    '  "decisions": [string],\n'
    '  "action_items": [\n    {\n      "owner": string,\n      "task": string,\n'
    '      "due_date": "YYYY-MM-DD or null if not specified"\n    }\n  ]\n}\n\n'
    "Begin with { and end with }."
)


def check_ex5(weak: str, strong: str) -> dict:
    try:
        parsed = json.loads(strong)
        strong_ok = (
            len(parsed.get("action_items", [])) == 3
            and len(parsed.get("attendees", [])) == 3
            and "decisions" in parsed
        )
    except json.JSONDecodeError:
        strong_ok = False
    weak_is_json = is_valid_json(weak)
    return {
        "weak_is_valid_json": weak_is_json,
        "strong_has_full_schema": strong_ok,
        "demonstrates_fix": strong_ok and not weak_is_json,
    }


EXERCISES = [
    ("Exercise 1: Ambiguity — Document Summarization", EX1_WEAK, EX1_STRONG, EX1_DOC, check_ex1),
    (
        "Exercise 2: Format — Intent Classification (JSON)",
        EX2_WEAK,
        EX2_STRONG,
        EX2_USER,
        check_ex2,
    ),
    (
        "Exercise 3: Hallucination — Document-Grounded Q&A",
        EX3_WEAK,
        EX3_STRONG,
        EX3_USER,
        check_ex3,
    ),
    (
        "Exercise 4: Constraint Violation — Length-Limited Description",
        EX4_WEAK,
        EX4_STRONG,
        "",
        check_ex4,
    ),
    (
        "Exercise 5: Structured Extraction — Meeting Notes",
        EX5_WEAK,
        EX5_STRONG,
        EX5_TEXT,
        check_ex5,
    ),
]


def main() -> int:
    print(mode_banner("OpenAI"))
    print("=" * 72)
    print("Lab 10: Writing Your First Professional Prompts")
    print(f"Model: {MODEL}")
    print("=" * 72)

    client = LabClient(FIXTURES_DIR, MODEL)
    passed = 0

    for title, weak_prompt, strong_prompt, user, check in EXERCISES:
        print(f"\n### {title}")
        print("-" * 72)
        weak = call(client, weak_prompt, user)
        strong = call(client, strong_prompt, user)
        print(f"[WEAK]\n{weak[:200]}")
        print(f"\n[STRONG]\n{strong[:200]}")
        result = check(weak, strong)
        ok = result["demonstrates_fix"]
        passed += ok
        print(f"\nCheck: {result}")
        print(f"Fix demonstrated: {'yes' if ok else 'no'}")

    print(f"\n{'=' * 72}")
    print(f"{passed}/{len(EXERCISES)} exercises show the weak-to-strong fix working as designed.")
    print()
    print("Every failure here has the same shape: the weak prompt left something")
    print("unspecified — structure, format, grounding, a limit, a schema — and the")
    print("model filled the gap with its own judgment. The strong prompt did not")
    print("make the model smarter. It removed the gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
