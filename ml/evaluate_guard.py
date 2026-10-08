
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.complexity_features import (
    CLASS_ORDER,
    FEATURE_NAMES,
    extract_features,
    rule_based_estimate,
)
from ml.extra_features import EXTRA_NAMES, extra_features

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


def metrics(y_true, y_pred):
    distance = np.abs(y_true - y_pred)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        float(np.mean(distance <= 1)),
        float(np.mean(distance)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = load_data(find_data_file(args.data))
    print("Samples:", len(rows))

    features = [extract_features(row["src"]) for row in rows]
    y = np.array([CLASS_ORDER.index(row["complexity"]) for row in rows])
    groups = np.array([row["problem"] for row in rows])
    rule = np.array([CLASS_ORDER.index(rule_based_estimate(f)) for f in features])
    lines = np.array([
        sum(1 for line in row["src"].splitlines() if line.strip()) for row in rows
    ])
    print("Program length (non-empty lines): median", int(np.median(lines)),
          "| shortest", int(lines.min()),
          "| programs under 12 lines:", int((lines < 12).sum()))
    extras = [extra_features(row["src"]) for row in rows]
    X = np.hstack([
        np.array([[f[n] for n in FEATURE_NAMES] for f in features], dtype=float),
        rule.reshape(-1, 1),
        np.array([[e[n] for n in EXTRA_NAMES] for e in extras], dtype=float),
    ])

    variants = {
        "Rule engine alone": lambda p, c, r, n: r,
        "ML alone": lambda p, c, r, n: p,
        "ML, fallback to rules if conf < 0.40": lambda p, c, r, n: np.where(c < 0.40, r, p),
        "ML, fallback if conf < 0.40 or under 8 lines": lambda p, c, r, n: np.where((c < 0.40) | (n < 8), r, p),
        "ML, fallback if conf < 0.40 or under 12 lines (agent)": lambda p, c, r, n: np.where((c < 0.40) | (n < 12), r, p),
        "ML, fallback if conf < 0.40 or under 16 lines": lambda p, c, r, n: np.where((c < 0.40) | (n < 16), r, p),
    }
    buckets = [("up to 8 lines", 0, 8), ("9-15 lines", 9, 15), ("16-30 lines", 16, 30), ("31+ lines", 31, 10 ** 9)]
    bucket_stats = {name: [0, 0, 0] for name, _, _ in buckets}  # count, rule correct, ML correct
    scores = {name: [] for name in variants}

    for seed in range(args.seeds):
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
            print(f"seed {seed} fold {fold}/{args.folds}", flush=True)
            model = ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            model.fit(X[train], y[train])

            raw = model.predict_proba(X[test])
            proba = np.zeros((len(test), len(CLASS_ORDER)))
            for column, cls in enumerate(model.classes_):
                proba[:, int(cls)] = raw[:, column]
            pred = proba.argmax(axis=1)
            conf = proba.max(axis=1)

            for name, choose in variants.items():
                scores[name].append(metrics(y[test], choose(pred, conf, rule[test], lines[test])))

            for name, low, high in buckets:
                mask = (lines[test] >= low) & (lines[test] <= high)
                bucket_stats[name][0] += int(mask.sum())
                bucket_stats[name][1] += int((rule[test][mask] == y[test][mask]).sum())
                bucket_stats[name][2] += int((pred[mask] == y[test][mask]).sum())

    base = np.array([s[0] for s in scores["Rule engine alone"]])

    print()
    header = f"{'variant':<50}{'acc':>7}{'macroF1':>9}{'within1':>9}{'MAE':>7}{'vs rule':>9}"
    print(header)
    print("-" * len(header))
    for name, entries in scores.items():
        arr = np.array(entries)
        diff = arr[:, 0] - base
        print(
            f"{name:<50}{arr[:, 0].mean():>7.3f}{arr[:, 1].mean():>9.3f}"
            f"{arr[:, 2].mean():>9.3f}{arr[:, 3].mean():>7.3f}{diff.mean():>+9.3f}"
        )
    print()
    print("'vs rule' = accuracy difference to the rule engine on the same folds.")

    print()
    print(f"{'program length':<18}{'programs':>10}{'rule acc':>10}{'ML acc':>9}")
    print("-" * 47)
    for name, _, _ in buckets:
        count, rule_ok, ml_ok = bucket_stats[name]
        if count:
            print(f"{name:<18}{count:>10}{rule_ok / count:>10.3f}{ml_ok / count:>9.3f}")
        else:
            print(f"{name:<18}{0:>10}")


if __name__ == "__main__":
    main()