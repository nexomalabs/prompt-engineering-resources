"""Lab 8 — Exploring Generative Models (reference solution).

Generates text at five temperatures and measures how output diversity changes.
The generation calls go through the shared LabClient, which replays recorded
fixtures in CI (LAB_PROJECT_STANDARD.md section 6) and calls the real API when
you have a key.

    NEXOMA_LAB_MODE=fixture  (default) replay recordings, $0.00, no key
    NEXOMA_LAB_MODE=record   call the real API and save what comes back
    NEXOMA_LAB_MODE=live     call the real API, save nothing
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from nexoma_labs.client import LabClient, mode_banner  # noqa: E402
from nexoma_labs.costs import CostLedger, Price  # noqa: E402

MODEL = "gpt-4o-mini"
PROMPT = (
    "In exactly one sentence, describe the main difference between "
    "a generative model and a discriminative model."
)
TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.4]
SAMPLES_PER_TEMPERATURE = 5
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "chapter-08"


def cross_sample_diversity(samples: list[str]) -> float:
    """1 minus the mean pairwise Jaccard similarity of the samples' vocabularies.

    Word-overlap *within* one sentence turns out not to move with temperature:
    a single sentence is lexically dense whether or not it was sampled hot.
    What actually changes is how much independent samples *agree with each
    other*. At T=0 they are the same sentence; at higher T they increasingly
    diverge in wording. Jaccard similarity between each pair of samples
    captures exactly that, and its complement is a diversity score in [0, 1].
    """
    from itertools import combinations

    sets = [set(s.lower().split()) for s in samples]
    if len(sets) < 2:
        return 0.0
    sims = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return 1.0 - (sum(sims) / len(sims))


def generate_samples(client: LabClient, temperature: float, n: int) -> list[str]:
    """n independent completions at one temperature. Each gets its own seed so
    they don't collide on a single fixture key — five real API calls would
    give five different samples too."""
    return [
        client.ask(PROMPT, temperature=temperature, max_tokens=100, seed=i).content
        for i in range(n)
    ]


def distinct_count(samples: list[str]) -> int:
    """How many distinct strings appear among the samples."""
    return len(set(samples))


def run_experiment(client: LabClient) -> dict[float, list[str]]:
    return {t: generate_samples(client, t, SAMPLES_PER_TEMPERATURE) for t in TEMPERATURES}


def main() -> int:
    print(mode_banner("OpenAI"))
    print("=" * 70)
    print("Lab 8: Temperature and Output Variation")
    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT}")
    print(f"Samples per temperature: {SAMPLES_PER_TEMPERATURE}")
    print("=" * 70)

    client = LabClient(FIXTURES_DIR, MODEL)
    results = run_experiment(client)

    print(f"\n{'temp':>6}{'distinct':>10}{'cross-sample diversity':>26}")
    print("-" * 42)
    diversity = {}
    for temp, samples in results.items():
        distinct = distinct_count(samples)
        diversity[temp] = cross_sample_diversity(samples)
        print(f"{temp:>6.1f}{distinct:>10}{diversity[temp]:>26.3f}")
    print("-" * 42)

    print(f"\nAt temperature 0.0, all {SAMPLES_PER_TEMPERATURE} samples are ", end="")
    print("identical." if distinct_count(results[0.0]) == 1 else "NOT identical.")
    print(f"At temperature {TEMPERATURES[-1]}, {distinct_count(results[TEMPERATURES[-1]])} "
          f"of {SAMPLES_PER_TEMPERATURE} samples are distinct.")

    lo, hi = TEMPERATURES[0], TEMPERATURES[-1]
    print(f"\nCross-sample diversity rises from {diversity[lo]:.3f} at T={lo} to "
          f"{diversity[hi]:.3f} at T={hi}. Five samples is a small enough number that")
    print("the trend is not perfectly monotonic — that variance is itself real and")
    print("worth noticing, not a flaw in the measurement.")
    print("Temperature does not change what the model knows. It changes how it")
    print("samples from what it knows, and 0.0 is the closest thing to a")
    print("deterministic answer this API offers — not a guarantee of one.")

    print()
    print("What running this experiment costs:")
    price = Price(input_per_mtok=0.15, output_per_mtok=0.60)  # gpt-4o-mini list rates
    ledger = CostLedger(price)
    for temp, samples in results.items():
        for s in samples:
            ledger.record(f"T={temp}", input_tokens=24, output_tokens=len(s.split()) * 2)
    print(ledger.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
