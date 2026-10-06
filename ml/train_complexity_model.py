"""
Time-complexity classification on CodeComplex (Python split).

Run from the project root:
    python -m ml.train_complexity_model
    python -m ml.train_complexity_model --data path/to/python_data.jsonl

Compares, on the SAME problem-level splits:
  1. Rule-based estimate (hand-written logic, no learning)
  2. Random Forest on your original 8 structural features
  3. Random Forest on the new complexity-oriented features
  4. Random Forest on the new features + the rule-based estimate (hybrid)

Evaluation:
  * Single split: GroupShuffleSplit(test_size=0.2, random_state=42) on the
    problem id, which reproduces the 3995 / 905 split you already use.
  * 5-fold StratifiedGroupKFold, repeated with 3 seeds, for a stable estimate.
  * Metrics: accuracy, macro-F1 and "within one class" accuracy. The classes
    are ordered (constant < logn < linear < nlogn < quadratic < cubic < np),
    so predicting quadratic for cubic is a smaller mistake than predicting
    constant for cubic.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ml.complexity_features import (
    CLASS_ORDER,
    FEATURE_NAMES,
    extract_features,
    rule_based_estimate,
)

BASELINE_FEATURES = [
    "loop_count",
    "condition_count",
    "call_count",
    "subscript_access_count",
    "max_nesting_depth",
    "max_loop_nesting",
    "is_recursive",
    "function_length",
]

DATA_CANDIDATES = [
    "dataset/python_data.jsonl",
    "dataset/codecomplex/python_data.jsonl",
    "data/python_data.jsonl",
    "python_data.jsonl",
]


def find_data_file(path):
    if path:
        if os.path.exists(path):
            return path
        sys.exit(f"Data file not found: {path}")

    for candidate in DATA_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    sys.exit(
        "Could not find python_data.jsonl. Pass it explicitly:\n"
        "  python -m ml.train_complexity_model --data path/to/python_data.jsonl"
    )


def load_data(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def within_one(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def make_model():
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def summarize(label, y_true, y_pred):
    print(
        f"{label:<34} accuracy {accuracy_score(y_true, y_pred):.4f} | "
        f"macro-F1 {f1_score(y_true, y_pred, average='macro'):.4f} | "
        f"within-one {within_one(y_true, y_pred):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="path to python_data.jsonl")
    args = parser.parse_args()

    data_path = find_data_file(args.data)
    rows = load_data(data_path)

    print("Data file:", data_path)
    print("Total samples:", len(rows))

    features = [extract_features(row["src"]) for row in rows]
    parse_failures = sum(1 for f in features if f["parse_ok"] == 0)
    print("Programs that failed to parse:", parse_failures)

    y = np.array([CLASS_ORDER.index(row["complexity"]) for row in rows])
    groups = np.array([row["problem"] for row in rows])
    print("Problems:", len(set(groups)))
    print()

    def matrix(names):
        return np.array([[f[name] for name in names] for f in features], dtype=float)

    rule_pred = np.array(
        [CLASS_ORDER.index(rule_based_estimate(f)) for f in features]
    )

    X_baseline = matrix(BASELINE_FEATURES)
    X_new = matrix(FEATURE_NAMES)
    X_hybrid = np.hstack([X_new, rule_pred.reshape(-1, 1)])

    experiments = [
        ("RF: original 8 features", X_baseline),
        ("RF: new features", X_new),
        ("RF: new features + rule estimate", X_hybrid),
    ]

    # -----------------------------------------------------------
    # Single problem-level split (same as your earlier runs)
    # -----------------------------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_new, y, groups))

    print("=== Single problem-level split (test_size=0.2, random_state=42) ===")
    print("Training samples:", len(train_idx), "| Testing samples:", len(test_idx))
    print(
        "Training problems:", len(set(groups[train_idx])),
        "| Testing problems:", len(set(groups[test_idx])),
    )
    print()

    summarize("Rule-based (no learning)", y[test_idx], rule_pred[test_idx])

    best_pred = None
    best_model = None
    for label, X in experiments:
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])
        summarize(label, y[test_idx], predictions)

        if label == "RF: new features":
            best_pred = predictions
            best_model = model

    print()
    print("Per-class report (RF: new features):")
    print(
        classification_report(
            y[test_idx], best_pred,
            labels=list(range(len(CLASS_ORDER))),
            target_names=CLASS_ORDER,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = true, columns = predicted):")
    print("Order:", CLASS_ORDER)
    print(confusion_matrix(y[test_idx], best_pred, labels=list(range(len(CLASS_ORDER)))))
    print()

    print("Top feature importances (RF: new features):")
    ranked = sorted(
        zip(FEATURE_NAMES, best_model.feature_importances_),
        key=lambda item: -item[1],
    )
    for name, importance in ranked[:12]:
        print(f"  {name:<28} {importance:.4f}")
    print()

    # -----------------------------------------------------------
    # Repeated grouped cross-validation
    # -----------------------------------------------------------
    print("=== 5-fold grouped cross-validation (3 seeds, split by problem) ===")

    rule_scores = []
    for seed in (0, 1, 2):
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for _, test in folds.split(X_new, y, groups):
            rule_scores.append(accuracy_score(y[test], rule_pred[test]))
    print(f"{'Rule-based (no learning)':<34} accuracy {np.mean(rule_scores):.4f} +/- {np.std(rule_scores):.4f}")

    for label, X in experiments:
        accuracies, macro_f1s, within_ones = [], [], []

        for seed in (0, 1, 2):
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for train, test in folds.split(X, y, groups):
                model = make_model()
                model.fit(X[train], y[train])
                predictions = model.predict(X[test])

                accuracies.append(accuracy_score(y[test], predictions))
                macro_f1s.append(f1_score(y[test], predictions, average="macro"))
                within_ones.append(within_one(y[test], predictions))

        print(
            f"{label:<34} accuracy {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f} | "
            f"macro-F1 {np.mean(macro_f1s):.4f} | within-one {np.mean(within_ones):.4f}"
        )

    print()
    print("Chance level for 7 balanced classes is about 0.143.")


if __name__ == "__main__":
    main()