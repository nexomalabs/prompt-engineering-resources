# Volume I · Chapter 4 — Lab: Building a Neural Network

## What You Will Build

A two-layer neural network in numpy — forward pass, cross-entropy loss, and
backpropagation written out by hand, with no framework. You will train it to separate
two interleaved spirals, a problem no linear model can solve, and prove your gradients
are correct by checking them numerically.

## Learning Outcomes

- Implement a forward pass through a network with a non-linear hidden layer.
- Explain why softmax must subtract the row maximum before exponentiating.
- Derive and implement backpropagation for softmax with cross-entropy.
- Verify a gradient implementation against a numerical estimate.
- Explain what a hidden layer buys you, using a case where a linear model fails.
- Recognise divergence and connect it to the learning rate.

## Prerequisites

- Python 3.12
- Chapter 4 read through §4.8
- **No API key required. Estimated API cost: $0.00** — this lab runs entirely locally.
- No network required. The dataset is generated from a fixed seed.

## Setup

```bash
cd volumes/volume-1/labs/chapter-04
uv sync
```

## Running It

```bash
uv run python src/lab.py
```

`src/lab.py` runs as supplied but the network is untrained, scoring about **0.50** —
chance, on a balanced two-class problem. Complete the four TODOs and re-run.

Reference implementation:

```bash
uv run python solution/lab.py
```

## Expected Result

Your completed `src/lab.py` should reach roughly 0.98 test accuracy. The reference
solution also prints the linear baseline; full output is in `expected/output.txt`:

```text
Dataset: two interleaved spirals, 600 points
Training points: 450
Test points:     150

Baseline — softmax regression, no hidden layer:
  test accuracy 0.507

Two-layer network, 32 hidden units:
  epoch    1   loss 0.6795   train acc 0.509
  epoch  100   loss 0.2956   train acc 0.909
  epoch  200   loss 0.1804   train acc 0.940
  epoch  300   loss 0.1239   train acc 0.978
  epoch  400   loss 0.0846   train acc 0.989

Final training loss: 0.0846
Train accuracy:      0.989
Test accuracy:       0.980
```

**0.507 against 0.980 is the result that matters.** The linear model is not badly
tuned — it is trying to draw one straight boundary through two spirals that wind around
each other. No amount of training fixes that. The hidden layer changes the class of
function the model can represent, and that is the whole argument of Chapter 4.

## Verifying Your Work

```bash
uv run pytest
```

Eleven tests. The important one is `test_backward_matches_numerical_gradient`: it
perturbs a single weight, measures how the loss actually changes, and compares that
against what your `backward()` claims. Backpropagation that passes this is correct;
backpropagation that trains but fails it is subtly wrong and will bite you later.

`test_linear_model_cannot` asserts the baseline stays **below** 0.70. If a future change
makes the dataset easy enough for a linear model, the lab has lost its point.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Accuracy stuck near 0.50 | `step()` is still `pass` | Complete TODO 4 |
| Loss is `nan` from epoch 1 | `softmax` does not subtract the row max | Complete TODO 2 as written |
| `FloatingPointError: loss became nan` | Learning rate too high — the model diverged | Lower `lr`; 0.5 works here |
| Loss falls then rises | Same cause, milder | Lower `lr` |
| Gradient test fails but training works | `dz1` is missing the ReLU mask | `dz1 = da1 * (z1 > 0)` |
| Gradient test off by a constant factor | Forgot `dz2 /= n` | Cross-entropy averages over the batch |

## Going Further

1. Set `n_hidden=2`. How small can the hidden layer get before the spirals stop separating?
2. Replace ReLU with `tanh`. Does it train faster, and does the initialisation still suit
   it? He initialisation assumes ReLU.
3. Add a second hidden layer. Does depth help here, or only cost more?
4. Raise `noise` in `make_spirals` to 0.6 and watch train and test accuracy diverge. That
   gap is overfitting, which Chapter 3 §3.10 named before you could see it.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Your starting point — runs, with four TODOs |
| `solution/lab.py` | Complete reference implementation |
| `data/make_dataset.py` | Deterministic two-spiral generator |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Pinned dependencies |
