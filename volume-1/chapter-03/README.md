# Volume I · Chapter 3 — Lab: Training Your First Machine Learning Model

## What You Will Build

A text classifier that routes short news headlines into one of three sections —
sports, health, or politics — using TF-IDF features and logistic regression. You will
train it, evaluate it on data it has never seen, and read a confusion matrix to find
out *which* headlines it gets wrong and why.

## Learning Outcomes

- Explain the difference between training, validation, and test data, and why fitting
  a vectoriser on test data invalidates a result.
- Convert raw text into numeric features using TF-IDF.
- Train a supervised classifier and evaluate it on held-out data.
- Interpret accuracy, precision, recall, and F1, and explain when accuracy alone misleads.
- Read a confusion matrix and identify which classes a model confuses.
- Inspect a model's learned weights to check it is keying on signal rather than noise.

## Prerequisites

- Python 3.12
- Chapter 3 read through §3.12
- **No API key required. Estimated API cost: $0.00** — this lab runs entirely locally.
- No network required. The corpus is generated deterministically from a fixed seed.

## Setup

```bash
cd volumes/volume-1/labs/chapter-03
uv sync
```

## Running It

```bash
uv run python src/lab.py
```

`src/lab.py` runs as supplied but the model is not trained: it predicts the majority
class and scores about **0.333**. Complete the three `TODO` blocks and re-run.

The reference implementation is in `solution/lab.py` if you get stuck:

```bash
uv run python solution/lab.py
```

## Expected Result

With all three TODOs complete, your output should match `expected/output.txt`:

```text
Corpus: 600 headlines, 3 sections (bundled)
Training headlines: 450
Test headlines:     150
Vocabulary size:    56

Test accuracy: 0.940

Classification report:
              precision    recall  f1-score   support

      health      0.923     0.960     0.941        50
    politics      0.939     0.920     0.929        50
      sports      0.959     0.940     0.949        50

    accuracy                          0.940       150
   macro avg      0.940     0.940     0.940       150
weighted avg      0.940     0.940     0.940       150

Confusion matrix (rows = actual, columns = predicted):
                health   politics     sports
health              48          1          1
politics             3         46          1
sports               1          2         47

Most informative terms per section:
  health     patient, therapy, clinic, referral, screening
  politics   policy, reform, panel, budget, appeal
  sports     season, contract, training, schedule, performance
```

**94% is the target, not 100%.** The corpus is built so that every section draws from
the same shared vocabulary and differs only in word frequency — which is how real text
behaves. The nine errors are the point of the exercise: examine them.

Notice the asymmetry. Three *politics* headlines were routed to *health*, but only one
went the other way. A shared word such as `panel` or `review` is doing that, and no
single accuracy number would have told you.

## Verifying Your Work

```bash
uv run pytest
```

Eleven tests. They check that the corpus is deterministic and balanced, that the
pipeline beats a random baseline, that the result is reproducible, and — deliberately —
that accuracy stays **below** 0.99, because a perfect score would empty the confusion
matrix and leave the chapter's precision and recall material with nothing to work on.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: sklearn` | Dependencies not installed | Run `uv sync` from this directory |
| `ModuleNotFoundError: build_corpus` | Running from the wrong directory | `cd` into `chapter-03` first; the script resolves `data/` relative to itself |
| Accuracy stuck at 0.333 | A TODO is incomplete — `model` is still `None` | Check all three TODO blocks are done |
| Accuracy is 1.000 | The vectoriser was fitted on the full dataset before splitting | Fit on `X_train` only, then `transform` the test set |
| `ValueError: empty vocabulary` | `min_df` is higher than the corpus supports | Keep `min_df=2` |
| Your numbers differ slightly | `random_state` not set | Pass `random_state=SEED` to both the split and the model |

## Going Further

1. Swap the bundled corpus for the real thing: `uv run python solution/lab.py --dataset newsgroups`.
   It downloads about 14 MB. Compare the accuracy and the shape of the confusion matrix.
2. Replace logistic regression with `MultinomialNB` or `LinearSVC`. Which wins, and does
   the winner change when you shorten the headlines further?
3. Set `doc_len=(10, 20)` in `build()`. Accuracy goes to 1.000. Explain why longer text
   makes this task trivial, and what that implies for short-text classification in production.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Your starting point — runs, with three TODOs |
| `solution/lab.py` | Complete reference implementation |
| `data/build_corpus.py` | Deterministic corpus generator; read the docstring |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Pinned dependencies |
