

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ml.complexity_features import (
    CLASS_ORDER,
    FEATURE_NAMES,
    extract_features,
    rule_based_estimate,
)
from ml.evaluate_guard import find_data_file, load_data
from ml.extra_features import EXTRA_NAMES, extra_features


THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


def build_dataset(rows):
    features = [extract_features(r["src"]) for r in rows]
    extras = [extra_features(r["src"]) for r in rows]

    y = np.array([
        CLASS_ORDER.index(r["complexity"])
        for r in rows
    ])

    groups = np.array([
        r["problem"]
        for r in rows
    ])

    rule = np.array([
        CLASS_ORDER.index(rule_based_estimate(f))
        for f in features
    ])

    lines = np.array([
        sum(1 for line in r["src"].splitlines() if line.strip())
        for r in rows
    ])

    X = np.hstack([
        np.array(
            [[f[n] for n in FEATURE_NAMES] for f in features],
            dtype=float
        ),
        rule.reshape(-1, 1),
        np.array(
            [[e[n] for n in EXTRA_NAMES] for e in extras],
            dtype=float
        ),
    ])

    return X, y, groups, rule, lines


def train_predict(X_train, y_train, X_test):
    model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    raw = model.predict_proba(X_test)

    proba = np.zeros(
        (len(X_test), len(CLASS_ORDER))
    )

    for column, cls in enumerate(model.classes_):
        proba[:, int(cls)] = raw[:, column]

    predictions = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    return predictions, confidence


def evaluate_threshold(y, predictions, confidence, threshold):
    accepted = confidence >= threshold
    n = int(accepted.sum())

    if n == 0:
        return 0.0, 0.0, 0

    accuracy = (predictions[accepted] == y[accepted]).mean()
    coverage = n / len(y)

    return accuracy, coverage, n


def choose_threshold(
    y,
    predictions,
    confidence,
    target_coverage=0.40,
):
    """
    Choose the threshold using ONLY validation data.

    We select the highest accuracy threshold among thresholds
    whose coverage is at least target_coverage.
    """

    candidates = []

    for threshold in THRESHOLDS:
        accuracy, coverage, n = evaluate_threshold(
            y,
            predictions,
            confidence,
            threshold,
        )

        if coverage >= target_coverage:
            candidates.append(
                (accuracy, threshold, coverage, n)
            )

    if not candidates:
        return min(
            THRESHOLDS,
            key=lambda t: abs(
                evaluate_threshold(
                    y,
                    predictions,
                    confidence,
                    t
                )[1] - target_coverage
            )
        )

    candidates.sort(
        key=lambda x: (x[0], x[2]),
        reverse=True
    )

    return candidates[0][1]


def main():
    rows = load_data(find_data_file(None))

    print("Samples:", len(rows))

    X, y, groups, rule, lines = build_dataset(rows)

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026,
    )

    results = []

    for fold, (train_val, test) in enumerate(
        outer.split(X, y, groups),
        1,
    ):
        print()
        print(f"Outer fold {fold}/5")

        X_train_val = X[train_val]
        y_train_val = y[train_val]
        groups_train_val = groups[train_val]

        X_test = X[test]
        y_test = y[test]

        # ---------------------------------------------------------
        # Inner split:
        # used ONLY to select the confidence threshold.
        # ---------------------------------------------------------

        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=1000 + fold,
        )

        inner_train, validation = next(
            inner.split(
                X_train_val,
                y_train_val,
                groups_train_val,
            )
        )

        ml_val, conf_val = train_predict(
            X_train_val[inner_train],
            y_train_val[inner_train],
            X_train_val[validation],
        )

        threshold = choose_threshold(
            y_train_val[validation],
            ml_val,
            conf_val,
            target_coverage=0.40,
        )

        # ---------------------------------------------------------
        # Train a fresh model on ALL outer training data.
        # ---------------------------------------------------------

        ml_test, conf_test = train_predict(
            X_train_val,
            y_train_val,
            X_test,
        )

        # ---------------------------------------------------------
        # Evaluate the threshold chosen WITHOUT seeing the test set.
        # ---------------------------------------------------------

        accuracy, coverage, n = evaluate_threshold(
            y_test,
            ml_test,
            conf_test,
            threshold,
        )

        all_accuracy = (ml_test == y_test).mean()

        results.append({
            "fold": fold,
            "threshold": threshold,
            "accuracy": accuracy,
            "coverage": coverage,
            "samples": n,
            "ml_accuracy": all_accuracy,
        })

        print(f"Chosen threshold: {threshold:.2f}")
        print(f"ML accuracy:       {100 * all_accuracy:.1f}%")
        print(f"Accepted samples:  {n}/{len(test)}")
        print(f"Coverage:          {100 * coverage:.1f}%")
        print(f"Accepted accuracy: {100 * accuracy:.1f}%")

    print()
    print("=" * 70)
    print("NESTED VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"{'Fold':<8}"
        f"{'Threshold':>12}"
        f"{'Coverage':>12}"
        f"{'Accepted':>12}"
        f"{'Accuracy':>12}"
        f"{'ML all':>12}"
    )

    for r in results:
        print(
            f"{r['fold']:<8}"
            f"{r['threshold']:>12.2f}"
            f"{100 * r['coverage']:>11.1f}%"
            f"{r['samples']:>12}"
            f"{100 * r['accuracy']:>11.1f}%"
            f"{100 * r['ml_accuracy']:>11.1f}%"
        )

    thresholds = np.array([
        r["threshold"] for r in results
    ])

    accuracies = np.array([
        r["accuracy"] for r in results
    ])

    coverages = np.array([
        r["coverage"] for r in results
    ])

    ml_accuracies = np.array([
        r["ml_accuracy"] for r in results
    ])

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Mean selected threshold: "
        f"{thresholds.mean():.2f}"
    )

    print(
        f"Threshold std:           "
        f"{thresholds.std():.2f}"
    )

    print(
        f"Mean coverage:           "
        f"{100 * coverages.mean():.1f}%"
    )

    print(
        f"Mean accepted accuracy:  "
        f"{100 * accuracies.mean():.1f}%"
    )

    print(
        f"Mean ML accuracy:        "
        f"{100 * ml_accuracies.mean():.1f}%"
    )

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if accuracies.mean() > ml_accuracies.mean():
        print(
            "Abstention improves accuracy on accepted predictions."
        )
    else:
        print(
            "Abstention does not improve accepted-prediction accuracy."
        )

    if thresholds.std() <= 0.05:
        print(
            "Threshold selection is reasonably stable across folds."
        )
    else:
        print(
            "Threshold selection varies substantially across folds."
        )

    if coverages.mean() >= 0.30:
        print(
            "Coverage remains substantial."
        )
    else:
        print(
            "Coverage is low; the model may be too uncertain."
        )

    print()
    print(
        "IMPORTANT: The outer test folds were never used "
        "to select thresholds."
    )
    print(
        "The complexity agent was not modified."
    )


if __name__ == "__main__":
    main()