# Volume I — Laboratories

*Applied AI Engineering · Volume I: Foundations*

Every lab runs from a clean checkout and is tested in CI on every push.
See [`.npf/LAB_PROJECT_STANDARD.md`](../../../.npf/LAB_PROJECT_STANDARD.md).

**Python 3.12.** Install [uv](https://docs.astral.sh/uv/), then from any lab directory:

```bash
uv sync
uv run python src/lab.py     # your starting point
uv run pytest                # verify your work
```

---

## Index

| Ch | Lab | Type | Time | API cost | Status |
|:--:|-----|------|:----:|:--------:|:------:|
| 1 | [Mapping the AI Landscape](chapter-01/) | Paper | 30 min | $0.00 | ✅ |
| 2 | [AI Timeline Analysis](chapter-02/) | Paper | 30 min | $0.00 | ✅ |
| 3 | [Training Your First Machine Learning Model](chapter-03/) | Engineering | 45 min | $0.00 | ✅ |
| 4 | [Building a Neural Network](chapter-04/) | Engineering | 60 min | $0.00 | ✅ |
| 5 | [Convolution and Image Classification](chapter-05/) | Engineering | 60 min | $0.00 | ✅ |
| 6 | [Visualizing Attention Weights](chapter-06/) | Engineering | 50 min | $0.00 | ✅ |
| 7 | [Exploring Tokenization](chapter-07/) | Engineering | 45 min | $0.00 | ✅ |
| 8 | [Exploring Generative Models](chapter-08/) | Engineering | 45 min | $0.00 (fixtures) | ✅ |
| 9 | [Comparing LLM vs. Reasoning Models](chapter-09/) | Engineering | 50 min | $0.00 (fixtures) | ✅ |
| 10 | [Writing Your First Professional Prompts](chapter-10/) | Engineering | 60 min | $0.00 (fixtures) | ✅ |

**Estimated API cost for the whole volume: under $1.00.** Labs 1 through 7 run entirely
locally with no model calls.

Legend — ✅ complete and passing CI · ⬜ not built yet.

Rows follow the **current manuscript**, which has 10 chapters. The v2.0.0 blueprint
([`.npf/blueprints/volume-1.md`](../../../.npf/blueprints/volume-1.md)) grows Volume I to
15 chapters; those labs are specified there but their chapters are not written yet.

---

## `shared/`

Helpers imported by labs across the volume. Standard library only, so it stays
importable in a bare environment.

| Module | Purpose |
|--------|---------|
| `nexoma_labs.costs` | `Price`, `CostLedger`, `estimate` — token cost accounting, introduced in Chapter 9 |
| `nexoma_labs.fixtures` | `FixtureStore`, `lab_mode` — recorded model responses so labs test without an API key |

```bash
cd shared && python3 -m unittest discover -s tests -t .   # 17 tests
```

### Lab modes

Labs that call a model honour `NEXOMA_LAB_MODE`:

| Mode | Behaviour | Used by |
|------|-----------|---------|
| `fixture` *(default)* | Replay recorded responses. No key, no cost. | CI tier 2 |
| `record` | Call the real API and save responses to `fixtures/` | Author, when writing a lab |
| `live` | Call the real API, save nothing | CI tier 3, nightly |

A changed prompt produces a different fixture key and fails loudly rather than
replaying a stale answer.

---

## Recording the video companion

Record against a **tagged release** of the companion repository and put the tag in the
video description. Each lab's `README.md` is the recording outline: *What You Will Build*
is the intro, *Setup* and *Running It* are the demo, *Expected Result* is the payoff, and
*If It Does Not Work* pre-empts the comments.
