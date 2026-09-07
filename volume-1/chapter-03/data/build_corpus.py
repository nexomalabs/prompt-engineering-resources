"""Deterministic synthetic headline corpus for Lab 3.

The chapter discusses the 20 Newsgroups dataset, which needs a 14 MB network
download. A lab that requires the network cannot run in tier-2 CI
(LAB_PROJECT_STANDARD §6), and a lab whose accuracy drifts with an upstream
dataset cannot have a stable "Expected Result" section. So the default corpus is
generated here, deterministically, from a fixed seed.

The task is section routing: given a short news headline, predict whether it
belongs to sports, health, or politics. Headlines are chosen deliberately —
short-text classification (headlines, support tickets, search queries, log lines)
is both a common production task and a harder one, because each item carries
little evidence.

Design note, and the reason this file is worth reading:

Sections do NOT have exclusive vocabularies. Every topic draws from the same word
pool and differs only in how *often* it uses each word. That is how real text
behaves, and it is what makes the task learnable but not trivial. Two earlier versions of this generator were discarded. The first gave each
section private vocabulary and classified at 100%. The second shared vocabulary
but used 18–34 word documents and still reached 99.3%. Both produced an almost
empty confusion matrix, which teaches nothing about precision, recall, or the
trade-off between them — the entire point of the chapter. Short headlines over a
shared vocabulary land near 87%, which gives the reader real errors to examine.
"""

from __future__ import annotations

import random

# One shared pool. Topics differ by weight, not by membership.
VOCAB = """team season injury recovery therapy treatment patient clinic panel review
    policy committee vote result report record schedule official decision analysis
    risk assessment outcome response pressure defence support case study evidence
    trial screening budget reform debate ruling appeal coverage referral contract
    penalty performance training staff facility programme standard guideline
    inquiry statement measure benefit rate access delay reserve""".split()

# Words a topic uses often (weight 6) and occasionally (weight 2).
# The heavy sets overlap deliberately: injury/recovery/therapy serve both sport
# and medicine; panel/review/policy serve both medicine and politics.
HEAVY = {
    "sports":   "team season injury recovery penalty training contract schedule performance".split(),
    "health":   "patient therapy treatment clinic injury recovery referral screening trial".split(),
    "politics": "policy vote committee reform debate ruling budget panel appeal".split(),
}
MEDIUM = {
    "sports":   "staff facility record result defence support coverage".split(),
    "health":   "evidence study case assessment guideline standard outcome".split(),
    "politics": "statement inquiry measure access rate official decision".split(),
}


def _weights(topic: str) -> list[float]:
    heavy, medium = set(HEAVY[topic]), set(MEDIUM[topic])
    return [6.0 if w in heavy else 2.0 if w in medium else 0.6 for w in VOCAB]


def build(docs_per_topic: int = 200, seed: int = 20260904, doc_len=(3, 7)):
    """Return (headlines, labels, section_names). Deterministic for a given seed."""
    rng = random.Random(seed)
    names = sorted(HEAVY)
    weights = {n: _weights(n) for n in names}
    texts, labels = [], []
    for idx, name in enumerate(names):
        w = weights[name]
        for _ in range(docs_per_topic):
            n = rng.randint(*doc_len)
            texts.append(" ".join(rng.choices(VOCAB, weights=w, k=n)))
            labels.append(idx)
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order], names


if __name__ == "__main__":
    t, y, names = build()
    print(f"{len(t)} headlines, {len(names)} sections: {', '.join(names)}")
    print(f"shared vocabulary of {len(VOCAB)} words — no section-exclusive terms")
    print(f"example ({names[y[0]]}): {t[0]}")
