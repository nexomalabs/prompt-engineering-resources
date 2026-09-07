# Volume I · Chapter 5 — Lab: Convolution and Image Classification

## What You Will Build

**Part A** — 2-D convolution, ReLU, and max-pooling implemented in numpy, then used to
show *what convolution actually buys you*: not raw accuracy, but tolerance to the image
moving.

**Part B** — the same operation at production scale: a pre-trained ResNet-50 classifying
a real photograph.

## Learning Outcomes

- Implement 2-D convolution and explain the output-size formula.
- Explain what a kernel detects and why edge kernels respond to edges.
- Implement max-pooling and explain what information it discards on purpose.
- Demonstrate translation tolerance rather than asserting it.
- Explain why convolution beat fully-connected networks on images.
- Apply a pre-trained model and explain why preprocessing must match training.

## Prerequisites

- Python 3.12
- Chapter 5 read through §5.2
- **Part A: no API key, no network. Estimated API cost: $0.00.**
- Part B: PyTorch, and about 100 MB of weights downloaded on first run. Still $0.00 —
  it runs locally, but it is not free of disk or bandwidth.

## Setup

```bash
cd volumes/volume-1/labs/chapter-05
uv sync                        # Part A
uv sync --group resnet         # Part B, only if you want it — torch is large
```

## Running It

```bash
uv run python src/lab.py                                   # Part A, your work
uv run python solution/lab.py                              # Part A, reference
uv run python solution/resnet_demo.py --image photo.jpg    # Part B
```

`src/lab.py` runs as supplied, but `convolve2d`, `relu`, and `max_pool` are stubs, so the
convolutional features are meaningless. Complete the three TODOs.

## Expected Result

Full output is in `expected/output.txt`:

```text
dataset                    raw dims   raw acc  conv dims  conv acc
------------------------------------------------------------------
centred 8x8                      64     0.967         64     0.978
shifted on 12x12 canvas         144     0.309        144     0.707
------------------------------------------------------------------
```

**Read the two rows against each other; that comparison is the entire lab.**

On centred digits the two are within noise. Every digit sits in the same place, so a
pixel index is already a reliable feature and convolution adds almost nothing. If the lab
stopped here it would suggest convolution is not worth the trouble.

Shift the same digits to random positions on a larger canvas and raw pixels collapse from
0.967 to **0.309** — barely above chance for ten classes. Convolutional features fall much
less, to **0.707**, and now lead by roughly 0.40.

That gap is the point. The same kernel is applied at every position and pooling discards
exactly where it fired, so the features barely notice the digit moving. A raw-pixel
classifier has to learn each position separately and mostly fails. Real images are never
perfectly centred, which is why CNNs displaced pixel classifiers.

## Verifying Your Work

```bash
uv run pytest
```

Sixteen tests. `test_matches_a_naive_loop` checks your vectorised convolution against
four nested loops. `test_identity_kernel_returns_the_image` catches off-by-one padding
errors. `test_convolution_wins_once_digits_move` asserts the shifted-digit gap stays above
0.15 — if that ever fails, the lab has stopped making its point.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Both accuracies near 0.10 | `convolve2d` still returns zeros | Complete TODO 1 |
| Output is the wrong shape | Padding applied after measuring `h, w` | Pad first, then read the shape |
| Identity-kernel test fails | Off-by-one in the window slice | Slice `image[i:i+kh, j:j+kw]` |
| Conv features no better when shifted | `max_pool` still returns zeros | Complete TODO 3 |
| `ConvergenceWarning` from lbfgs | Features not standardised | Keep the `StandardScaler` in the pipeline |
| Part B: `ModuleNotFoundError: torch` | Optional group not installed | `uv sync --group resnet` |
| Part B: confident but absurd predictions | Preprocessing does not match training | Keep the ImageNet `Normalize` values |

## Going Further

1. Raise pooling to `size=4`. Accuracy on centred digits drops but shifted digits improve.
   Explain the trade-off between invariance and discrimination.
2. Replace the hand-designed kernels with random ones. How much of the benefit survives?
   That result is why learned kernels matter.
3. Stack a second conv-pool layer. Does depth help at 8×8, or is there nothing left to pool?
4. In Part B, feed the same photo rotated 90°. ResNet-50 is translation-tolerant but not
   rotation-tolerant. Explain why, from the architecture.

## Files

| Path | Purpose |
|------|---------|
| `src/lab.py` | Part A starting point — runs, with three TODOs |
| `solution/lab.py` | Part A reference implementation |
| `solution/resnet_demo.py` | Part B — pre-trained ResNet-50, not CI-gated |
| `tests/test_lab.py` | Runs against `solution/` in CI |
| `expected/output.txt` | Captured reference output |
| `pyproject.toml` | Dependencies; `resnet` group is optional |

> **Why Part A exists.** The chapter's lab applies ResNet-50 directly. That needs PyTorch
> and a weights download, so it cannot run in the offline suite that gates every push
> (`LAB_PROJECT_STANDARD.md` §6). Part A is the CI-tested core, and implementing
> convolution by hand first means Part B is not magic: `model.conv1` is the function you
> just wrote, with 64 learned kernels instead of four hand-picked ones.
