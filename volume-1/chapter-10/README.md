# Volume I · Chapter 10 — Lab: Writing Your First Professional Prompts

**Volume I capstone.**

## What You Will Build

Five exercises. Each pairs a WEAK prompt against a STRONG one on the identical task, and
each demonstrates one specific failure mode from the taxonomy in §10.9: ambiguity,
format, hallucination, constraint violation, and missing structure. You will write a
checker for each exercise that proves — rather than just asserts — that the strong prompt
actually fixed what the weak one broke.

## Learning Outcomes

- Apply the prompt anatomy from §10.3 to remove ambiguity from a task.
- Force a structured, parseable output format and verify it parses.
- Ground a model in a provided document and detect when it is not grounded.
- Enforce a hard constraint (a word limit) and verify compliance.
- Extract a full schema from unstructured text and verify every field arrived.
- Recognise that a "weak" prompt is not a bad prompt — it is an underspecified one.

## Prerequisites

- Python 3.12
- Chapter 10 read through §10.9
- **No API key required by default.** This lab replays recorded model responses.
  Estimated API cost to complete: **$0.00**.

## Setup

```bash
cd volumes/volume-1/labs/chapter-10
uv sync
```

## Running It

```bash
uv run python src/lab.py
```

`src/lab.py` runs as supplied against the fixtures, but all five `check_exN` functions are
stubs returning `False`, so every exercise reports **0/5**. Complete the five TODOs — one
check function per exercise.

Reference implementation:

```bash
uv run python solution/lab.py
```

## Expected Result

Full output is in `expected/output.txt`. Summary: **5/5**.

| Exercise | Failure mode | Weak result | Strong result |
|---|---|---|---|
| 1 — Summarization | Ambiguity | 2 loose sentences, no figures anchored to structure | Exactly 3 sentences, each figure in its assigned slot |
| 2 — Intent classification | Format | Prose explanation, not parseable | `{"intent": "request", "confidence": "high", "topic": "invoice correction"}` |
| 3 — Document Q&A | Hallucination | Invents "Standard plan, $12.99/month" — a plan that does not exist | Correctly reports "Professional, $20/month" from the document |
| 4 — Product description | Constraint violation | 76 words, has a headline it was told to omit | 38 words, no headline, ends on the call to action |
| 5 — Meeting notes | Missing structure | A bulleted list — not valid JSON, drops attendees and decisions entirely | Valid JSON, all 3 attendees, all 3 action items, decisions included |

**Exercise 3 is the one to read closely.** "Nexoma Cloud Storage" is fictional — invented
for this exercise — so there is no real answer for an ungrounded model to happen to
recall. The weak prompt's confident, specific, wrong answer (a "Standard" plan at "$12.99"
that does not exist anywhere in the source material) is not a one-off glitch. It is what
happens by default when a model is asked a factual question with nothing to ground it:
it produces something plausible-shaped rather than admitting it does not know.

**Exercise 1 also taught something during authoring.** A naive sentence counter that
splits on every `.` character counts **8** sentences in the strong response, not 3 —
because the text contains `$4.2M`, `$1.1M`, `$1.8M`, and `$3.1M`, each with a period the
splitter mistook for a sentence break. `sentence_count()` splits only on a
period/question mark/exclamation point that is followed by whitespace and a capital
letter, which fixes it. `TestSentenceCount.test_dollar_figures_do_not_create_false_sentence_breaks`
pins this so it cannot regress.

## Verifying Your Work

```bash
uv run pytest
```

Nineteen tests. `TestExerciseChecks` verifies each check function against hand-built
weak/strong strings, independent of the fixtures — so a future fixture change cannot
silently break the checker itself. `TestAllFiveExercisesDemonstrateTheirFix` is the
end-to-end claim: all five, against the bundled recordings.

## Recording Your Own Fixtures

The bundled fixtures are hand-authored, representative responses — not a live
transcript. With an `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=sk-...
NEXOMA_LAB_MODE=record uv run python solution/lab.py
```

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Every exercise reports "fail" | All five `check_exN` are still stubs | Complete the five TODOs |
| Exercise 1 sentence count looks wrong | Splitting on every `.`, breaking on `$4.2M` | Use `sentence_count()`, not `text.split(".")` |
| Exercise 2/5 "fix demonstrated" is `False` even though output looks right | `demonstrates_fix` requires the **weak** response to fail too | Check both sides, not just the strong response |
| `FixtureMissing` error | Prompt text edited — even whitespace changes the fixture key | Use the constants exactly as given |

## Going Further

1. Write a stricter Exercise 3 check that also verifies the response does *not* contain
   fabricated numbers (`$12.99`, `250 GB`) even when the correct numbers are also present.
2. Add a sixth exercise for a failure mode of your own choosing — refusal, verbosity, or
   inconsistent formatting across repeated calls — and write its checker.
3. Run Exercise 1's weak prompt five times in `record` mode at temperature 0.7. Does the
   length and structure vary run to run? That variance is the ambiguity failure showing up
   a second way — not just wrong once, but *inconsistent* every time.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Your starting point — runs against fixtures, with five TODOs |
| `solution/lab.py` | Complete reference implementation |
| `fixtures/chapter-10/` | Hand-authored recorded responses, replayed in fixture mode |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Dependencies; `live` group only needed for `record`/`live` mode |
