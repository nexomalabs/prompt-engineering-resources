"""Lab 3 — Training Your First Machine Learning Model (starter).

Route short news headlines to a section: sports, health, or politics.

This file RUNS as-is. It loads the data and prints a baseline, but the model is
not trained yet. Complete the three TODOs and the accuracy will rise from roughly
33% (random) to roughly 94%.

    python src/lab.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from build_corpus import build  # noqa: E402

SEED = 20260904


def main() -> int:
    texts, labels, names = build()
    print(f"Corpus: {len(texts)} headlines, {len(names)} sections")
    print(f"Example: {texts[0]!r} -> {names[labels[0]]}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=SEED, stratify=labels
    )
    print(f"Training headlines: {len(X_train)}")
    print(f"Test headlines:     {len(X_test)}")

    # ---------------------------------------------------------------- TODO 1
    # Turn text into numeric features with TfidfVectorizer.
    # Fit on the TRAINING data only, then transform both sets.
    # Fitting on test data leaks information and is the most common mistake here.
    #
    #   vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
    #   X_train_v = vectorizer.fit_transform(X_train)
    #   X_test_v  = vectorizer.transform(X_test)
    vectorizer = None
    X_train_v = X_test_v = None

    # ---------------------------------------------------------------- TODO 2
    # Fit a LogisticRegression model on the vectorised training data.
    # Pass random_state=SEED so your result is reproducible.
    model = None

    # ---------------------------------------------------------------- TODO 3
    # Predict on the test set and replace the baseline below.
    if model is None or X_test_v is None:
        majority = Counter(y_train).most_common(1)[0][0]
        y_pred = [majority] * len(y_test)
        print("\n[baseline] model not trained yet — predicting the majority class")
    else:
        y_pred = model.predict(X_test_v)

    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=names, digits=3, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"{'':<11}" + "".join(f"{n[:9]:>11}" for n in names))
    for name, row in zip(names, confusion_matrix(y_test, y_pred)):
        print(f"{name[:10]:<11}" + "".join(f"{v:>11}" for v in row))

    if vectorizer is not None:
        print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
