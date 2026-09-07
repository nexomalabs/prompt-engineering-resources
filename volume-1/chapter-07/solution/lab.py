"""Lab 7 Part A — Byte-pair encoding from scratch (reference solution).

Trains a BPE tokenizer on a small corpus, encodes and decodes text with it, and
measures the consequences that matter to a prompt engineer: how token count
varies by input type, and what that costs.

Standard library only. Part B (solution/tiktoken_demo.py) compares against a
production tokenizer; it downloads vocabulary files, so it is not CI-gated.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from nexoma_labs.costs import CostLedger, Price  # noqa: E402

END = "</w>"  # marks a word boundary, so "in" inside a word differs from "in" alone

CORPUS = """
the engineer designs the system and the engineer tests the system
engineering requires design testing and redesign of the design
a designer designs designs and an engineer engineers engineering
testing the tested design tests the designer and the engineering
the systems engineer redesigned the tested engineering system
""".split()

SAMPLES = {
    "common English": "the engineer designs the system",
    "repeated words": "the the the the the",
    "unseen words": "quixotic zephyr onomatopoeia",
    "numbers": "1234567890 42 3.14159",
    "whitespace": "the     engineer",
    "punctuation": "the engineer, designs; the system.",
}


def word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """Each word becomes a tuple of characters plus an end-of-word marker."""
    return {tuple(word) + (END,): count for word, count in Counter(corpus).items()}


def count_pairs(freqs: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, count in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs


def merge_pair(pair: tuple[str, str], freqs: dict[tuple[str, ...], int]):
    """Replace every adjacent occurrence of `pair` with the joined symbol."""
    a, b = pair
    joined = a + b
    out = {}
    for symbols, count in freqs.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(joined)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        out[tuple(merged)] = out.get(tuple(merged), 0) + count
    return out


def train_bpe(corpus: list[str], num_merges: int):
    """Return (merges, vocab). Deterministic: ties broken by sorting the pair."""
    if num_merges < 0:
        raise ValueError("num_merges must not be negative")
    freqs = word_frequencies(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(freqs)
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        if pairs[best] < 2:
            break  # nothing left worth merging
        merges.append(best)
        freqs = merge_pair(best, freqs)
    vocab = sorted({s for symbols in freqs for s in symbols})
    return merges, vocab


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word) + [END]
    for a, b in merges:
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    return [tok for word in text.split() for tok in encode_word(word, merges)]


def decode(tokens: list[str]) -> str:
    """Inverse of encode, up to whitespace normalisation. See test_lab.py."""
    return "".join(tokens).replace(END, " ").strip()


def main() -> int:
    print("Training BPE on a small corpus")
    print(f"  corpus: {len(CORPUS)} words, {len(set(CORPUS))} distinct")
    print()

    print(f"{'merges':>7}{'vocab':>8}{'tokens for the sample':>24}")
    print("-" * 39)
    sample = SAMPLES["common English"]
    for n in (0, 5, 10, 20, 40):
        merges, vocab = train_bpe(CORPUS, n)
        print(f"{n:>7}{len(vocab):>8}{len(encode(sample, merges)):>24}")
    print("-" * 39)
    print("More merges means a larger vocabulary and fewer tokens per text.")
    print("That trade-off is the only dial a tokenizer really has.")
    print()

    merges, vocab = train_bpe(CORPUS, 40)
    print(f"First 12 merges learned (most frequent pair first):")
    for i, (a, b) in enumerate(merges[:12], 1):
        print(f"  {i:>2}. {a!r} + {b!r} -> {a + b!r}")
    print()

    print("How the same trained tokenizer handles different input:")
    print(f"{'input type':<18}{'chars':>7}{'tokens':>8}{'chars/token':>13}")
    print("-" * 46)
    for label, text in SAMPLES.items():
        toks = encode(text, merges)
        ratio = len(text) / len(toks) if toks else 0.0
        print(f"{label:<18}{len(text):>7}{len(toks):>8}{ratio:>13.2f}")
    print("-" * 46)
    print()
    print("Text the tokenizer was trained on compresses well. Unseen words fall")
    print("back to near one token per character, and numbers fragment badly.")
    print("This is why a prompt in a rare language can cost several times more")
    print("than the same content in English.")
    print()

    print("What that costs, for the same amount of text")
    print("-" * 58)
    # Hold the character count fixed and let the token count vary. This is the
    # comparison that matters: identical volumes of text, different bills.
    CHARS = 20_000
    price = Price(input_per_mtok=3.0, output_per_mtok=15.0)
    ledger = CostLedger(price)
    for label in ("common English", "unseen words"):
        text = SAMPLES[label]
        tokens_per_char = len(encode(text, merges)) / len(text)
        ledger.record(label, input_tokens=round(CHARS * tokens_per_char), output_tokens=300)
    print(ledger.report())
    print()
    cheap, dear = ledger.calls
    print(f"Same {CHARS:,} characters. {dear.label} costs "
          f"{dear.cost / cheap.cost:.1f}x more than {cheap.label},")
    print(f"because it needs {dear.input_tokens / cheap.input_tokens:.1f}x the tokens to say it.")
    print(f"At 50,000 calls a day the difference is "
          f"${(dear.cost - cheap.cost) * 50_000:,.0f}.")
    print()
    print("Token count is not a detail. It is the unit you are billed in, and")
    print("it depends on what the tokenizer was trained on rather than on how")
    print("much you actually said.")

    print()
    print("Round-trip check:")
    for label in ("common English", "unseen words"):
        text = SAMPLES[label]
        ok = decode(encode(text, merges)) == text
        print(f"  {label:<18} {'ok' if ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
