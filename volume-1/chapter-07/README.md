# Volume I · Chapter 7 — Lab: Exploring Tokenization

## What You Will Build

**Part A** — a byte-pair encoding tokenizer from scratch, in the standard library. You
will train it, watch it learn whole words out of characters, and measure what
tokenization costs in real money.

**Part B** — the same text through `cl100k_base`, a production tokenizer, for comparison.

## Learning Outcomes

- Explain the BPE training loop: count pairs, merge the most frequent, repeat.
- Implement encoding and decoding, and state where the round trip is lossy.
- Explain the vocabulary-size against sequence-length trade-off.
- Predict which inputs tokenize badly and explain why.
- Convert a token count into a cost, and a cost into a daily bill.

## Prerequisites

- Python 3.12
- Chapter 7 read through §7.3
- **Part A: no dependencies, no API key, no network. Estimated API cost: $0.00.**
- Part B: `tiktoken`, which downloads vocabulary files on first use. Also $0.00.

## Setup

```bash
cd volumes/volume-1/labs/chapter-07
uv sync                          # Part A needs nothing but Python
uv sync --group tiktoken         # Part B, optional
```

## Running It

```bash
uv run python src/lab.py                       # Part A, your work
uv run python solution/lab.py                  # Part A, reference
uv run python solution/tiktoken_demo.py        # Part B
```

`src/lab.py` runs as supplied, but `count_pairs`, `merge_pair`, and `encode_word` are
stubs, so training learns **0 merges** and the sample encodes to **32 tokens**, one per
character. Complete the three TODOs and it should reach 40 merges and 5 tokens.

## Expected Result

Full output is in `expected/output.txt`. Three things to look at.

**The trade-off.** More merges means a bigger vocabulary and shorter sequences:

| merges | vocab | tokens for the sample |
|---:|---:|---:|
| 0 | 17 | 32 |
| 5 | 21 | 27 |
| 10 | 21 | 19 |
| 20 | 23 | 13 |
| 40 | 27 | 5 |

That is the only dial a tokenizer really has.

**What it learns.** The merges are readable, and you can watch a word assemble:

```text
   1. 'e' + 's'    -> 'es'
   4. 'd' + 'es'   -> 'des'
   5. 'des' + 'i'  -> 'desi'
   6. 'desi' + 'g' -> 'desig'
   7. 'desig' + 'n'-> 'design'
```

**What it costs.** Same tokenizer, different input:

| input type | chars | tokens | chars/token |
|---|---:|---:|---:|
| common English | 31 | 5 | 6.20 |
| unseen words | 28 | 29 | 0.97 |
| numbers | 21 | 22 | 0.95 |

Then, holding the **text volume fixed** at 20,000 characters:

```text
common English      3226 input tokens    $0.014178
unseen words       20714 input tokens    $0.066642

unseen words costs 4.7x more, because it needs 6.4x the tokens to say it.
At 50,000 calls a day the difference is $2,623.
```

**That is the lesson.** Identical amounts of text, a 4.7× difference in the bill,
determined entirely by what the tokenizer happened to be trained on. It is why the same
prompt in a less-represented language can cost several times more than in English, and
why token count — not word count — is the unit you budget in.

## Verifying Your Work

```bash
uv run pytest
```

Twenty-three tests. `test_does_not_remerge_its_own_output` catches the classic BPE bug
where advancing by one instead of two turns `aaaa` into `aaa`.
`test_more_merges_never_increases_token_count` asserts the trade-off holds.
`test_trained_text_compresses_better_than_unseen` asserts the cost gap stays at least 3×,
so the lab cannot quietly stop making its point.

## Known Limitation

`encode()` splits on whitespace, so runs of spaces collapse and `decode(encode(x))`
normalises whitespace rather than preserving it. This is deliberate — it keeps Part A
readable — and `test_whitespace_is_normalised_not_preserved` pins the behaviour so it
cannot change silently. Real tokenizers encode whitespace as part of the token, which is
why `"hello"` and `" hello"` are different tokens. Part B shows that directly.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| 0 merges learned | `count_pairs` still returns an empty Counter | Complete TODO 1 |
| Merges learned but tokens unchanged | `encode_word` does not apply them | Complete TODO 3 |
| A symbol like `aaa` appears from `aa` | Advancing by 1 after a merge | Advance by 2 |
| Merges differ between runs | `max()` over an unsorted dict | Sort the pairs before `max()` |
| `'the'` never becomes one token | End marker missing | Append `END` in `word_frequencies` |
| Round trip loses spaces | Expected — see Known Limitation | Nothing to fix |
| Part B: `ModuleNotFoundError` | Optional group not installed | `uv sync --group tiktoken` |

## Going Further

1. Train on 200 merges instead of 40. Where does the token count stop improving, and what
   does that tell you about choosing a vocabulary size?
2. Add a second language to `CORPUS` and re-measure the chars/token table. How much does
   English degrade to make room?
3. In Part B, compare `"1000"` against `"1,000"`. Explain why digit grouping changes the
   token count and what that implies for prompts containing figures.
4. Take a real prompt from your own work, run it through Part B, and compute its monthly
   cost at your expected volume. Chapter 9 of the v2.0.0 blueprint builds on this.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Part A starting point — runs, with three TODOs |
| `solution/lab.py` | Part A reference implementation |
| `solution/tiktoken_demo.py` | Part B — production tokenizer, not CI-gated |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | No runtime dependencies; `tiktoken` group is optional |

> **Why Part A exists.** The chapter's lab uses `tiktoken`, which downloads vocabulary
> files and so cannot run in the offline suite that gates every push
> (`LAB_PROJECT_STANDARD.md` §6). Building BPE by hand first means Part B is not magic: a
> production tokenizer is this algorithm, trained on far more text with a far larger merge
> table.
