# Volume I · Chapter 8 — Lab: Exploring Generative Models

## What You Will Build

An experiment that asks the same question five times at five different temperatures,
and measures how much the independent answers agree with each other. You will discover
that word-diversity *within* one sentence does not move with temperature — what moves is
how much five separate samples *diverge from each other*.

## Learning Outcomes

- Explain what temperature controls in sampling, and what it does not control.
- Measure output diversity across independent samples using Jaccard similarity.
- Explain why 0.0 is "closest to deterministic" rather than "deterministic."
- Read a cost report and connect sample count to API spend.
- Recognise when a within-sample metric is measuring the wrong thing.

## Prerequisites

- Python 3.12
- Chapter 8 read through §8.2
- **No API key required by default.** This lab replays recorded model responses.
  Estimated API cost to complete: **$0.00**.
- To run it against the real API instead, see "Recording Your Own Fixtures" below.

## Setup

```bash
cd volumes/volume-1/labs/chapter-08
uv sync
```

## Running It

```bash
uv run python src/lab.py
```

`src/lab.py` runs as supplied against the recorded fixtures, but the diversity metric is
a stub and `generate_samples` returns only one sample, so every row reads 0.000 with
1 distinct sample. Complete the two TODOs.

Reference implementation:

```bash
uv run python solution/lab.py
```

## Expected Result

Full output is in `expected/output.txt`:

```text
[OpenAI: replaying recorded fixtures, no API key needed]

  temp  distinct    cross-sample diversity
------------------------------------------
   0.0         1                     0.000
   0.3         5                     0.312
   0.7         5                     0.803
   1.0         5                     0.892
   1.4         5                     0.876
------------------------------------------

At temperature 0.0, all 5 samples are identical.
At temperature 1.4, 5 of 5 samples are distinct.

Cross-sample diversity rises from 0.000 at T=0.0 to 0.876 at T=1.4.
```

**Read the trend, not the last row.** Diversity rises sharply from 0.0 to 1.0, then dips
very slightly at 1.4. That dip is not an error — five samples is a small enough number
that the estimate has real variance, and the README says so rather than smoothing it away.
The direction across the full range is what the lab is asking you to demonstrate.

**The metric itself is the harder lesson.** An earlier version of this lab measured
unique-word ratio *within* each single sentence, and that number stayed flat around
0.82–0.86 regardless of temperature — because any one well-formed sentence is lexically
dense whether it was sampled hot or cold. Temperature does not make a sentence use more
distinct words; it makes independent samples *disagree with each other more*. Only a
metric computed *across* samples — pairwise Jaccard similarity here — can see that.

## Verifying Your Work

```bash
uv run pytest
```

Twelve tests. `test_zero_temperature_is_deterministic` checks the lab's first claim
directly. `test_diversity_rises_overall` checks the trend end-to-end rather than
step-to-step, because a stricter test would demand perfect monotonicity from only five
samples — which the expected output itself does not show.

## Recording Your Own Fixtures

The bundled fixtures in `fixtures/chapter-08/` are hand-authored, representative
responses — not a live API transcript. If you have an `OPENAI_API_KEY`, you can generate
your own:

```bash
export OPENAI_API_KEY=sk-...
NEXOMA_LAB_MODE=record uv run python solution/lab.py
```

This calls the real API, prints real results, and saves them to `fixtures/chapter-08/` as
new recordings, keyed by request hash. Your original bundled fixtures are untouched unless
a request matches one exactly. Re-run with `NEXOMA_LAB_MODE=fixture` afterwards to replay
what you just recorded.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Every row reads 0.000 | `cross_sample_diversity` still returns `0.0` | Complete TODO 1 |
| Every row shows 1 distinct sample | `generate_samples` returns a single-item list | Complete TODO 2, with a distinct `seed` per call |
| `FixtureMissing` error | A temperature or seed outside the bundled set | Only `TEMPERATURES` and `seed=0..4` are recorded |
| `ModuleNotFoundError: openai` | Only needed for `record`/`live` mode | `uv sync --group live`, or stay in fixture mode |
| `record` mode fails with an auth error | No `OPENAI_API_KEY` set | Export it, or skip recording — fixture mode needs nothing |

## Going Further

1. Raise `SAMPLES_PER_TEMPERATURE` to 20 in a `record` run. Does the dip at T=1.4 persist,
   or was it noise from having only 5 samples?
2. Compute the within-sentence unique-word ratio yourself and confirm it stays flat. This
   is the metric the lab deliberately does not use — verify why for yourself.
3. Add a sixth temperature above 1.4 (some APIs allow up to 2.0). Does diversity keep
   rising, or does output quality visibly break down first?

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Your starting point — runs against fixtures, with two TODOs |
| `solution/lab.py` | Complete reference implementation |
| `fixtures/chapter-08/` | Hand-authored recorded responses, replayed in fixture mode |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Dependencies; `live` group only needed for `record`/`live` mode |
