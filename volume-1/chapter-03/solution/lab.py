"""Lab 3 — Training Your First Machine Learning Model (reference solution).

Routes short news headlines to a section (sports, health, politics) using TF-IDF
features and logistic regression, then reports the metrics the chapter introduces:
accuracy, precision, recall, F1, and a confusion matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from build_corpus import build  # noqa: E402

SEED = 20260904


def load(dataset: str):
    """Return (texts, labels, names). 'bundled' is offline and deterministic."""
    if dataset == "bundled":
        return build()
    from sklearn.datasets import fetch_20newsgroups

    cats = ["rec.sport.hockey", "sci.med", "talk.politics.misc"]
    data = fetch_20newsgroups(
        subset="all", categories=cats, remove=("headers", "footers", "quotes")
    )
    return list(data.data), list(data.target), list(data.target_names)


def train_and_evaluate(texts, labels, names, test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=SEED, stratify=labels
    )

    # Step 1 — turn text into numbers. TF-IDF weights a term by how often it
    # appears in a document against how rare it is across the corpus.
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    # Step 2 — fit the model. Note fit_transform on train, transform on test:
    # fitting the vectorizer on test data leaks information and is the single
    # most common mistake in this pipeline.
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train_v, y_train)

    # Step 3 — evaluate on data the model has never seen.
    y_pred = model.predict(X_test_v)
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(vectorizer.vocabulary_),
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=names, digits=3),
        "confusion": confusion_matrix(y_test, y_pred),
        "model": model,
        "vectorizer": vectorizer,
        "names": names,
    }


def top_features(result, k=5):
    """The k highest-weighted terms per class — what the model actually learned."""
    inv = {i: t for t, i in result["vectorizer"].vocabulary_.items()}
    out = {}
    for idx, name in enumerate(result["names"]):
        coefs = result["model"].coef_[idx]
        top = sorted(range(len(coefs)), key=lambda i: coefs[i], reverse=True)[:k]
        out[name] = [inv[i] for i in top]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["bundled", "newsgroups"], default="bundled",
                    help="'newsgroups' downloads ~14 MB on first run")
    a = ap.parse_args()

    texts, labels, names = load(a.dataset)
    print(f"Corpus: {len(texts)} headlines, {len(names)} sections ({a.dataset})")

    r = train_and_evaluate(texts, labels, names)
    print(f"Training headlines: {r['n_train']}")
    print(f"Test headlines:     {r['n_test']}")
    print(f"Vocabulary size:    {r['n_features']}")
    print()
    print(f"Test accuracy: {r['accuracy']:.3f}")
    print()
    print("Classification report:")
    print(r["report"])
    print("Confusion matrix (rows = actual, columns = predicted):")
    header = "".join(f"{n[:9]:>11}" for n in names)
    print(f"{'':<11}{header}")
    for name, row in zip(names, r["confusion"]):
        print(f"{name[:10]:<11}" + "".join(f"{v:>11}" for v in row))
    print()
    print("Most informative terms per section:")
    for name, terms in top_features(r).items():
        print(f"  {name:<10} {', '.join(terms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
