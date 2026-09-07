# Volume I · Chapter 6 — Lab: Visualizing Attention Weights

## What You Will Build

**Part A** — scaled dot-product attention in numpy, plus an ASCII heatmap renderer, used
to show that attention patterns are readable rather than mysterious. You will build a
content-matching head, a previous-token head, and a causal mask, and you will measure why
the `sqrt(d_k)` divisor exists.

**Part B** — the same operation in a real model: attention extracted from a pre-trained
BERT, all 144 heads.

## Learning Outcomes

- Implement scaled dot-product attention and explain each term.
- Explain why softmax must subtract its row maximum, and what `-inf` does to a masked score.
- Explain the `sqrt(d_k)` divisor by measuring what happens without it.
- Implement a causal mask and explain why decoders need one.
- Read an attention heatmap and identify what a head is doing.
- Explain how multi-head attention differs from a single wider head.

## Prerequisites

- Python 3.12
- Chapter 6 read through §6.5
- **Part A: no API key, no network. Estimated API cost: $0.00.**
- Part B: torch and transformers, and about 440 MB of weights on first run. Also $0.00,
  but not free of disk or bandwidth.

## Setup

```bash
cd volumes/volume-1/labs/chapter-06
uv sync                      # Part A
uv sync --group bert         # Part B, optional
```

## Running It

```bash
uv run python src/lab.py                                        # Part A, your work
uv run python solution/lab.py                                   # Part A, reference
uv run python solution/bert_demo.py --text "the cat sat on the mat"   # Part B
```

`src/lab.py` runs as supplied, but `softmax`, the score matrix, and `causal_mask` are
stubs, so every heatmap is flat at 0.167. Complete the three TODOs.

## Expected Result

Full output is in `expected/output.txt`. Four things to look at.

**Content matching.** `Wq = Wk = identity`, so tokens attend by embedding similarity:

```text
          the   cat   sat    on   the   mat
   the     ==                      ==
   cat           %%
   sat                 @@
    on           --          ++
   the     ==                      ==
   mat                                   @@
```

`'the'` at position 0 attends to position 4 with weight **0.482**, and to itself with the
same weight. Identical words have identical embeddings, so the head links both
occurrences. That is the mechanism behind induction heads, which copy from earlier
matching context.

**Previous token.** Built only from positional encoding, no content at all:

```text
          the   cat   sat    on   the   mat
   the     @@
   cat     @@
   sat           %%
    on                 %%
   the                       %%
   mat                             %%
```

Mean weight on the immediately preceding token: **0.886**. Position alone produces a
positional pattern, and trained models learn heads that look very like this.

**Causal masking.** Total weight on future tokens: **0.000000**. A decoder cannot attend
forwards, and that constraint is what makes next-token prediction a valid objective.

**Why divide by sqrt(d_k).** At `d_k = 64`:

| | mean row entropy | max weight |
|---|---:|---:|
| scaled | 2.146 bits | 0.715 |
| unscaled | 0.131 bits | 1.000 |

Unscaled, the dot products grow with dimension, softmax saturates to one-hot, and the
gradient through it nearly vanishes. The divisor is not cosmetic — it is what keeps
attention trainable.

## Verifying Your Work

```bash
uv run pytest
```

Eighteen tests. `test_output_is_a_weighted_average_of_v` checks the defining property.
`test_handles_negative_infinity` catches the masking bug where `-inf` produces `nan`
instead of zero. `test_scaling_keeps_attention_diffuse` asserts the entropy gap stays at
least 3× — measuring the claim rather than repeating it.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Every weight is 0.167 | `softmax` still returns a uniform stub | Complete TODO 1 |
| Weights are `nan` | Softmax without the max-subtraction, or `-inf - -inf` | Subtract the row max first |
| Heatmap looks right but is transposed | Used `K @ Q.T` | Scores are `Q @ K.T`; row *i* is what token *i* attends to |
| Future weight is not 0 | `causal_mask` returns all-`True` | Complete TODO 3; `np.tril` |
| `'the'` does not match itself | Embeddings regenerated per occurrence | One vector per distinct word |
| Part B: `ModuleNotFoundError` | Optional group not installed | `uv sync --group bert` |
| Part B: `output_attentions` empty | Model used a fused attention kernel | Load with `attn_implementation="eager"` |

## Going Further

1. Run `demo_scaling` at `d_k` of 4, 16, 64, 256. Plot entropy against dimension. At what
   point does the divisor start to matter?
2. Build a head that attends to the *next* token. Then apply a causal mask and explain why
   it becomes useless.
3. In Part B, print entropy for all 12 heads in layer 0, then in layer 11. Do later layers
   look more focused or more diffuse?
4. Feed Part B a sentence with an ambiguous pronoun — "the trophy did not fit in the
   suitcase because it was too big". Find a head that links "it" to the right noun.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Part A starting point — runs, with three TODOs |
| `solution/lab.py` | Part A reference implementation |
| `solution/bert_demo.py` | Part B — real BERT attention, not CI-gated |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Dependencies; `bert` group is optional |

> **Why Part A exists.** The chapter's lab loads BERT directly. That needs torch and
> transformers and a 440 MB download, so it cannot run in the offline suite that gates
> every push (`LAB_PROJECT_STANDARD.md` §6). Part A is the CI-tested core, and building
> attention by hand first means Part B is not magic: each of BERT's 144 heads is the
> function you just wrote, with learned projections instead of hand-picked ones.
