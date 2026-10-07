

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.complexity_features import (
    CLASS_ORDER,
    FEATURE_NAMES,
    extract_features,
    rule_based_estimate,
)

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
    sys.exit("Could not find python_data.jsonl. Pass it with --data.")


def load_data(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def full_proba(model, X):
    """Probabilities as a (n, 7) matrix in CLASS_ORDER order."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASS_ORDER)))
    for column, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, column]
    return out


def decode_argmax(proba):
    return proba.argmax(axis=1)


def decode_median(proba):
    return (np.cumsum(proba, axis=1) >= 0.5).argmax(axis=1)


def metrics(y_true, y_pred):
    distance = np.abs(y_true - y_pred)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "within1": float(np.mean(distance <= 1)),
        "mae": float(np.mean(distance)),
    }


MODELS = {
    "RF 300 trees": lambda: RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1),
    "RF balanced, min_leaf 2": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "ExtraTrees balanced": lambda: ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=42),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    rows = load_data(find_data_file(args.data))
    print("Samples:", len(rows))

    features = [extract_features(row["src"]) for row in rows]
    y = np.array([CLASS_ORDER.index(row["complexity"]) for row in rows])
    groups = np.array([row["problem"] for row in rows])

    def matrix(names):
        return np.array([[f[n] for n in names] for f in features], dtype=float)

    rule_pred = np.array([CLASS_ORDER.index(rule_based_estimate(f)) for f in features])
    X_new = matrix(FEATURE_NAMES)
    X_hybrid = np.hstack([X_new, rule_pred.reshape(-1, 1)])

    experiments = [("RF 300 trees | features only", X_new, MODELS["RF 300 trees"])]
    for name, factory in MODELS.items():
        experiments.append((f"{name} | features + rule", X_hybrid, factory))

    results = {"Rule-based (baseline)": []}
    for label, _, _ in experiments:
        results[(label, "argmax")] = []
        results[(label, "median")] = []

    for seed in range(args.seeds):
        splitter = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X_new, y, groups), 1):
            print(f"seed {seed} fold {fold}/{args.folds}", flush=True)
            results["Rule-based (baseline)"].append(metrics(y[test], rule_pred[test]))

            for label, X, factory in experiments:
                model = factory()
                model.fit(X[train], y[train])
                proba = full_proba(model, X[test])
                results[(label, "argmax")].append(metrics(y[test], decode_argmax(proba)))
                results[(label, "median")].append(metrics(y[test], decode_median(proba)))

    def mean(entries, key):
        return float(np.mean([e[key] for e in entries]))

    baseline = results["Rule-based (baseline)"]
    base_acc = np.array([e["acc"] for e in baseline])

    print()
    header = f"{'model | decoding':<52}{'acc':>7}{'macroF1':>9}{'within1':>9}{'MAE':>7}{'vs rule':>9}{'wins':>7}"
    print(header)
    print("-" * len(header))
    print(
        f"{'Rule-based (baseline)':<52}{mean(baseline, 'acc'):>7.3f}"
        f"{mean(baseline, 'f1'):>9.3f}{mean(baseline, 'within1'):>9.3f}"
        f"{mean(baseline, 'mae'):>7.3f}{'':>9}{'':>7}"
    )

    ranked = []
    for key, entries in results.items():
        if key == "Rule-based (baseline)":
            continue
        acc = np.array([e["acc"] for e in entries])
        ranked.append((acc.mean(), key, entries, acc))
    ranked.sort(key=lambda item: -item[0])

    for _, (label, decode), entries, acc in ranked:
        diff = acc - base_acc
        print(
            f"{label + ' | ' + decode:<52}{acc.mean():>7.3f}"
            f"{mean(entries, 'f1'):>9.3f}{mean(entries, 'within1'):>9.3f}"
            f"{mean(entries, 'mae'):>7.3f}{diff.mean():>+9.3f}{np.mean(diff > 0):>7.0%}"
        )

    print()
    print("'vs rule' = mean accuracy difference to the rule baseline over the same folds;")
    print("'wins'    = share of folds where the model beat the rules.")
    print("Lower MAE (average class distance of mistakes) is better.")


if __name__ == "__main__":
    main()