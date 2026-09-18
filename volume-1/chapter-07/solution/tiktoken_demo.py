"""Lab 7 Part B — Comparing against a production tokenizer.

NOT part of the CI suite. tiktoken downloads its vocabulary files on first use,
so it cannot run in the offline tier that gates every push.

    pip install tiktoken
    python solution/tiktoken_demo.py

Part A builds BPE from scratch precisely so this is not magic: a production
tokenizer is the same algorithm, trained on far more text with a far larger
merge table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import CORPUS, SAMPLES, encode, train_bpe  # noqa: E402


def main() -> int:
    try:
        import tiktoken
    except ImportError:
        print("missing dependency: tiktoken", file=sys.stderr)
        print("install with:  pip install tiktoken", file=sys.stderr)
        return 1

    enc = tiktoken.get_encoding("cl100k_base")
    merges, vocab = train_bpe(CORPUS, 40)

    print(
        f"Ours:       {len(vocab):>7,} vocabulary, {len(merges)} merges, trained on {len(CORPUS)} words"  # noqa: E501 — column alignment
    )
    print(f"cl100k_base:{enc.n_vocab:>7,} vocabulary, trained on a large web corpus")
    print()

    print(f"{'input type':<18}{'chars':>7}{'ours':>7}{'cl100k':>8}{'ratio':>8}")
    print("-" * 48)
    for label, text in SAMPLES.items():
        ours = len(encode(text, merges))
        theirs = len(enc.encode(text))
        print(f"{label:<18}{len(text):>7}{ours:>7}{theirs:>8}{ours / max(theirs, 1):>8.2f}")
    print("-" * 48)
    print()
    print("A production tokenizer wins everywhere except on the narrow corpus")
    print("ours was trained on. That is the whole argument for training on a lot")
    print("of text: coverage, not cleverness.")
    print()

    print("Edge cases worth knowing about:")
    for text in ["hello", " hello", "Hello", "HELLO", "1000", "1,000", "🙂", "  spaced  out  "]:
        ids = enc.encode(text)
        print(f"  {text!r:<20} {len(ids):>2} tokens  {[enc.decode([i]) for i in ids]}")
    print()
    print("Leading whitespace, capitalisation and digit grouping all change the")
    print("token count. Part A's simple word split hides this; real tokenizers")
    print("encode whitespace as part of the token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
