"""Lab 6 Part B — Attention from a real pre-trained BERT.

NOT part of the CI suite. Needs transformers and torch and downloads about
440 MB of weights, so it cannot run in the offline tier that gates every push.

    pip install torch transformers
    python solution/bert_demo.py --text "the cat sat on the mat"

Part A implements attention from scratch precisely so this is not magic: every
one of BERT's 144 heads is the function you already wrote, with learned
projections instead of hand-picked ones.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import entropy, heatmap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="the cat sat on the mat")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("install with:  pip install torch transformers", file=sys.stderr)
        return 1

    name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation="eager")
    model.eval()  # inference mode

    enc = tokenizer(a.text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    print(f"{name}: {n_layers} layers x {n_heads} heads = {n_layers * n_heads} attention heads")
    print(f"Tokens: {tokens}\n")

    w = out.attentions[a.layer][0, a.head].numpy()
    print(f"Layer {a.layer}, head {a.head}:")
    print(heatmap(w, tokens))
    print(f"\nMean row entropy: {entropy(w):.3f} bits")
    print("Compare against Part A. Low entropy means a focused head; high means")
    print("it is averaging over the sequence. Both kinds exist in a trained model.\n")

    print("Entropy of every head in this layer:")
    for h in range(n_heads):
        e = entropy(out.attentions[a.layer][0, h].numpy())
        print(f"  head {h:>2}  {e:.3f} bits  {'#' * int(e * 8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
