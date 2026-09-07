"""Lab 8 — Exploring Generative Models (starter).

Generate text at five temperatures and measure how much independent samples
at the same temperature agree with each other.

This file RUNS as supplied against recorded fixtures, no API key needed. The
diversity metric is a stub, so every temperature reports 0.000. Complete the
two TODOs.

    python src/lab.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from nexoma_labs.client import LabClient, mode_banner  # noqa: E402

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

    Jaccard similarity of two sets is |intersection| / |union|: 1.0 means
    identical vocabulary, 0.0 means no words in common. Averaging it over
    every pair of samples and subtracting from 1 gives a diversity score in
    [0, 1], where 0 means all samples used exactly the same words.
    """
    # TODO 1: build one set of lowercased words per sample, then average the
    # pairwise Jaccard similarity across all pairs. `combinations(sets, 2)`
    # gives you every pair once.
    #
    #   sets = [set(s.lower().split()) for s in samples]
    #   sims = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    #   return 1.0 - (sum(sims) / len(sims))
    return 0.0


def generate_samples(client: LabClient, temperature: float, n: int) -> list[str]:
    # TODO 2: return a list of n completions. Give each call a distinct seed
    # (0, 1, 2, ...) so they don't collide on a single fixture key.
    #
    #   return [client.ask(PROMPT, temperature=temperature, max_tokens=100,
    #                       seed=i).content for i in range(n)]
    return [client.ask(PROMPT, temperature=temperature, max_tokens=100, seed=0).content]


def main() -> int:
    print(mode_banner("OpenAI"))
    client = LabClient(FIXTURES_DIR, MODEL)
    print(f"{'temp':>6}{'distinct':>10}{'cross-sample diversity':>26}")
    print("-" * 42)
    for temp in TEMPERATURES:
        samples = generate_samples(client, temp, SAMPLES_PER_TEMPERATURE)
        print(f"{temp:>6.1f}{len(set(samples)):>10}{cross_sample_diversity(samples):>26.3f}")
    print("-" * 42)
    print("\nExpect diversity to rise from 0.000 at T=0.0 toward roughly 0.9 by T=1.0.")
    print("All zeros means TODO 1 is still a stub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
