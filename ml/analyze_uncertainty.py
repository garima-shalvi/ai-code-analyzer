
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


def collect(rows, folds=5, seeds=2):
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

    out = {
        "y": [],
        "ml": [],
        "conf": [],
        "rule": [],
        "lines": [],
        "index": [],
    }

    for seed in range(seeds):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed
        )

        for fold, (train, test) in enumerate(
            splitter.split(X, y, groups), 1
        ):
            print(
                f"seed {seed} fold {fold}/{folds}",
                flush=True
            )

            model = ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X[train], y[train])

            raw = model.predict_proba(X[test])

            proba = np.zeros(
                (len(test), len(CLASS_ORDER))
            )

            for column, cls in enumerate(model.classes_):
                proba[:, int(cls)] = raw[:, column]

            out["y"].append(y[test])
            out["ml"].append(proba.argmax(axis=1))
            out["conf"].append(proba.max(axis=1))
            out["rule"].append(rule[test])
            out["lines"].append(lines[test])
            out["index"].append(test)

    return {
        key: np.concatenate(value)
        for key, value in out.items()
    }


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    rows = load_data(find_data_file(None))

    print("Samples:", len(rows))

    d = collect(rows)

    y = d["y"]
    ml = d["ml"]
    rule = d["rule"]
    conf = d["conf"]
    lines = d["lines"]

    ml_ok = ml == y
    rule_ok = rule == y

    disagreement = ml != rule
    both_wrong = disagreement & ~ml_ok & ~rule_ok

    print()
    print("=" * 70)
    print("CONFIDENCE VS COVERAGE")
    print("=" * 70)

    print(
        f"{'Threshold':<14}"
        f"{'Coverage':>12}"
        f"{'Samples':>10}"
        f"{'Accuracy':>12}"
    )

    for threshold in [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]:
        accepted = conf >= threshold
        n = int(accepted.sum())

        if n == 0:
            continue

        coverage = n / len(y)
        accuracy = ml_ok[accepted].mean()

        print(
            f"{threshold:<14.2f}"
            f"{pct(coverage):>12}"
            f"{n:>10}"
            f"{pct(accuracy):>12}"
        )

    print()
    print("=" * 70)
    print("CONFIDENCE VS ACCURACY")
    print("=" * 70)

    buckets = [
        ("< 0.30", 0.0, 0.30),
        ("0.30-0.40", 0.30, 0.40),
        ("0.40-0.50", 0.40, 0.50),
        ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90+", 0.90, 1.01),
    ]

    print(
        f"{'Confidence':<16}"
        f"{'Samples':>10}"
        f"{'Mean conf':>12}"
        f"{'ML accuracy':>14}"
    )

    for label, lo, hi in buckets:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())

        if n == 0:
            print(f"{label:<16}{0:>10}")
            continue

        print(
            f"{label:<16}"
            f"{n:>10}"
            f"{conf[mask].mean():>12.3f}"
            f"{pct(ml_ok[mask].mean()):>14}"
        )

    print()
    print("=" * 70)
    print("BOTH-WRONG CASES")
    print("=" * 70)

    n = int(both_wrong.sum())

    print("Both wrong:", n)
    print("Percentage of all predictions:", pct(n / len(y)))
    print(
        "Percentage of disagreements:",
        pct(n / max(int(disagreement.sum()), 1))
    )

    print()
    print("True complexity distribution:")
    for idx, label in enumerate(CLASS_ORDER):
        mask = both_wrong & (y == idx)
        count = int(mask.sum())

        if count:
            print(
                f"  {label:<12}"
                f"{count:>6}"
                f"  {pct(count / max(n, 1))}"
            )

    print()
    print("ML prediction distribution:")
    for idx, label in enumerate(CLASS_ORDER):
        mask = both_wrong & (ml == idx)
        count = int(mask.sum())

        if count:
            print(
                f"  {label:<12}"
                f"{count:>6}"
                f"  {pct(count / max(n, 1))}"
            )

    print()
    print("Rule prediction distribution:")
    for idx, label in enumerate(CLASS_ORDER):
        mask = both_wrong & (rule == idx)
        count = int(mask.sum())

        if count:
            print(
                f"  {label:<12}"
                f"{count:>6}"
                f"  {pct(count / max(n, 1))}"
            )

    print()
    print("=" * 70)
    print("BOTH-WRONG BY ML CONFIDENCE")
    print("=" * 70)

    print(
        f"{'Confidence':<16}"
        f"{'Samples':>10}"
        f"{'Both wrong':>12}"
        f"{'Rate':>12}"
    )

    for label, lo, hi in buckets:
        bucket = (conf >= lo) & (conf < hi)
        total = int(bucket.sum())
        wrong = int((bucket & both_wrong).sum())

        if total == 0:
            continue

        print(
            f"{label:<16}"
            f"{total:>10}"
            f"{wrong:>12}"
            f"{pct(wrong / total):>12}"
        )

    print()
    print("=" * 70)
    print("BOTH-WRONG BY PROGRAM LENGTH")
    print("=" * 70)

    length_buckets = [
        ("<= 8", 0, 8),
        ("9-15", 9, 15),
        ("16-30", 16, 30),
        ("31-50", 31, 50),
        ("51+", 51, 10**9),
    ]

    print(
        f"{'Length':<16}"
        f"{'Samples':>10}"
        f"{'Both wrong':>12}"
        f"{'Rate':>12}"
    )

    for label, lo, hi in length_buckets:
        bucket = (lines >= lo) & (lines <= hi)
        total = int(bucket.sum())
        wrong = int((bucket & both_wrong).sum())

        if total == 0:
            continue

        print(
            f"{label:<16}"
            f"{total:>10}"
            f"{wrong:>12}"
            f"{pct(wrong / total):>12}"
        )

    print()
    print("=" * 70)
    print("BOTH-WRONG: MOST COMMON TRUE → ML → RULE PATTERNS")
    print("=" * 70)

    patterns = {}

    for i in np.where(both_wrong)[0]:
        key = (
            CLASS_ORDER[y[i]],
            CLASS_ORDER[ml[i]],
            CLASS_ORDER[rule[i]],
        )
        patterns[key] = patterns.get(key, 0) + 1

    for (true_label, ml_label, rule_label), count in sorted(
        patterns.items(),
        key=lambda item: -item[1]
    )[:20]:
        print(
            f"{true_label:<12}"
            f" ML={ml_label:<12}"
            f" Rule={rule_label:<12}"
            f"{count:>6}"
        )

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE BOTH-WRONG CASES")
    print("=" * 70)

    for threshold in [0.70, 0.80, 0.90]:
        mask = both_wrong & (conf >= threshold)

        print(
            f"Confidence >= {threshold:.2f}:"
            f" {int(mask.sum())} cases"
        )

    print()
    print("This experiment is evaluation-only.")
    print("No agent logic was changed.")


if __name__ == "__main__":
    main()