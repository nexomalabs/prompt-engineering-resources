"""Lab 7 Part A — Byte-pair encoding from scratch (starter).

Train a BPE tokenizer, encode text with it, and measure what tokenization costs.

This file RUNS as supplied. Training learns no merges, so every word encodes one
character at a time. Complete the three TODOs.

    python src/lab.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

END = "</w>"

CORPUS = """
the engineer designs the system and the engineer tests the system
engineering requires design testing and redesign of the design
a designer designs designs and an engineer engineers engineering
testing the tested design tests the designer and the engineering
the systems engineer redesigned the tested engineering system
""".split()


def word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    return {tuple(word) + (END,): count for word, count in Counter(corpus).items()}


def count_pairs(freqs: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    # TODO 1: for every word, count each adjacent symbol pair, weighted by how
    # often the word occurs.
    #
    #   for symbols, count in freqs.items():
    #       for a, b in zip(symbols, symbols[1:]):
    #           pairs[(a, b)] += count
    return pairs


def merge_pair(pair: tuple[str, str], freqs: dict[tuple[str, ...], int]):
    """Replace every adjacent occurrence of `pair` with the joined symbol."""
    a, b = pair
    joined = a + b
    out = {}
    for symbols, count in freqs.items():
        # TODO 2: walk the symbol list and join `a` followed by `b`.
        # Advance by 2 when you merge, or you will re-merge the result.
        merged = list(symbols)
        out[tuple(merged)] = out.get(tuple(merged), 0) + count
    return out


def train_bpe(corpus: list[str], num_merges: int):
    if num_merges < 0:
        raise ValueError("num_merges must not be negative")
    freqs = word_frequencies(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(freqs)
        if not pairs:
            break
        # Sort before max() so ties break deterministically.
        best = max(sorted(pairs), key=lambda p: pairs[p])
        if pairs[best] < 2:
            break
        merges.append(best)
        freqs = merge_pair(best, freqs)
    vocab = sorted({s for symbols in freqs for s in symbols})
    return merges, vocab


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word) + [END]
    # TODO 3: apply each learned merge in order, exactly as merge_pair does.
    return symbols


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    return [tok for word in text.split() for tok in encode_word(word, merges)]


def decode(tokens: list[str]) -> str:
    return "".join(tokens).replace(END, " ").strip()


def main() -> int:
    merges, vocab = train_bpe(CORPUS, 40)
    sample = "the engineer designs the system"
    tokens = encode(sample, merges)
    print(f"merges learned: {len(merges)}   vocabulary: {len(vocab)}")
    print(f"{sample!r}")
    print(f"  -> {len(tokens)} tokens: {tokens}")
    print()
    print("Expect 40 merges and 5 tokens once the TODOs are done.")
    print("0 merges and 32 tokens means count_pairs or merge_pair is still a stub.")
    print()
    print(f"round trip: {'ok' if decode(tokens) == sample else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
