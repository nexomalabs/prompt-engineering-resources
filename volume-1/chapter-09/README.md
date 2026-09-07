# Volume I · Chapter 9 — Lab: Comparing LLM vs. Reasoning Models

## What You Will Build

The same three multi-step math problems run under three conditions — a standard model
answering directly, the same model with chain-of-thought prompting, and a reasoning
model — scored for accuracy and compared for token cost.

## Learning Outcomes

- Explain why a direct answer to a multi-step problem fails silently.
- Explain what chain-of-thought prompting changes, and what it does not fix.
- Explain why a reasoning model's token count and latency are far higher.
- Build a simple automated accuracy check and know its limits.
- State the actual cost/accuracy trade-off between the three approaches, in numbers.

## Prerequisites

- Python 3.12
- Chapter 9 read through §9.2
- **No API key required by default.** This lab replays recorded model responses.
  Estimated API cost to complete: **$0.00**.

## Setup

```bash
cd volumes/volume-1/labs/chapter-09
uv sync
```

## Running It

```bash
uv run python src/lab.py
```

`src/lab.py` runs as supplied against the fixtures, but `check_accuracy` always returns
`False`, so every condition reports 0/3. Complete the one TODO.

Reference implementation:

```bash
uv run python solution/lab.py
```

## Expected Result

Full output is in `expected/output.txt`. The summary:

```text
Accuracy over 3 problems:
  direct     0/3
  cot        2/3
  reasoning  3/3
```

**Direct fails on all three, every time.** Asked for "only the final answer, no
explanation," the model produces a confident, specific, wrong number — 2:39 PM instead of
3:13 PM, 4/25 instead of 2/15, 10.00 days instead of 4.80. It never shows work, so there
is nothing to check it against, and no way for it to catch its own mistake.

**Chain-of-thought recovers two of three — not three.** On the probability problem, the
model reasons through the right method (4/10 × 3/9 = 12/90) and then simplifies the
fraction wrong, landing on 2/19 instead of 2/15. This is the honest failure mode CoT does
not fix: it makes the *method* checkable, not the arithmetic. A correct-looking derivation
can still end on a wrong number.

**The reasoning model gets all three right**, but at real cost. On problem P1, the visible
answer is under 100 words — the fixture records **1,032 hidden reasoning tokens** behind
it, more than ten times the length of the answer itself. That is the trade-off: fewer
wrong answers, several times the tokens, and correspondingly higher latency and cost.

## Verifying Your Work

```bash
uv run pytest
```

Thirteen tests. `test_direct_answers_are_all_wrong` and
`test_reasoning_model_gets_everything_right` check the lab's two clean claims directly.
`test_chain_of_thought_beats_direct_but_is_not_perfect` asserts CoT's accuracy is strictly
between 0 and 3 — if it ever hits a perfect 3/3, this lab has stopped demonstrating what a
step can still get wrong.

## A Note on Model Names

`o1-mini` is used as a placeholder for "the current reasoning model." OpenAI's
reasoning-model lineup has changed release to release, and by the time you read this it
may again. The bundled fixtures do not depend on which specific model produced them —
they demonstrate the *category* of behaviour (visible answer, large hidden reasoning
cost, higher accuracy on multi-step problems), which is durable even when the model name
is not.

## Recording Your Own Fixtures

The bundled fixtures are hand-authored, representative responses — not a live transcript.
With an `OPENAI_API_KEY` and access to a current reasoning model:

```bash
export OPENAI_API_KEY=sk-...
NEXOMA_LAB_MODE=record uv run python solution/lab.py
```

You will need to update `MODEL_REASONING` in `solution/lab.py` to whatever reasoning
model your account currently has access to before recording.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Every condition reports 0/3 | `check_accuracy` still returns `False` | Complete TODO 1 |
| `FixtureMissing` error | Problem text or mode does not match a bundled fixture exactly | Use the three `PROBLEMS` as given; do not edit their text |
| `check_accuracy` reports a false positive | Matching on a digit substring that also appears elsewhere | See `test_partial_digit_overlap_is_not_a_false_positive` for the boundary case |
| `record` mode: `NotImplementedError` | The starter's `call_reasoning` has no live path | Use `solution/lab.py` for recording, not `src/lab.py` |

## Going Further

1. Write a stricter `check_accuracy` that requires the exact answer format rather than a
   digit substring. Does it change any of the three tallies?
2. Add a fourth problem of your choosing. Predict which condition will fail before running
   it, in `record` mode, against a real account.
3. Estimate the dollar cost of the reasoning condition across all three problems using
   `nexoma_labs.costs`, and compare it against running all three problems through the
   chain-of-thought condition three times each, majority-voting the answer. Which is
   cheaper for the same accuracy?

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Your starting point — runs against fixtures, with one TODO |
| `solution/lab.py` | Complete reference implementation |
| `fixtures/chapter-09/` | Hand-authored recorded responses, replayed in fixture mode |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Dependencies; `live` group only needed for `record`/`live` mode |
