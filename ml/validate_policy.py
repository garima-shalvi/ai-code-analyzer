

import argparse

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from agents.complexity_agent import ACCEPT_AGREEMENT, ML_MIN_CONFIDENCE, decide
from ml.complexity_features import (
    CLASS_ORDER,
    FEATURE_NAMES,
    extract_features,
    rule_based_estimate,
)
from ml.evaluate_guard import find_data_file, load_data
from ml.extra_features import EXTRA_NAMES, extra_features


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = load_data(find_data_file(args.data))
    print("Samples:", len(rows))

    features = [extract_features(r["src"]) for r in rows]
    extras = [extra_features(r["src"]) for r in rows]
    y = np.array([CLASS_ORDER.index(r["complexity"]) for r in rows])
    groups = np.array([r["problem"] for r in rows])
    rule = np.array([CLASS_ORDER.index(rule_based_estimate(f)) for f in features])
    X = np.hstack([
        np.array([[f[n] for n in FEATURE_NAMES] for f in features], dtype=float),
        rule.reshape(-1, 1),
        np.array([[e[n] for n in EXTRA_NAMES] for e in extras], dtype=float),
    ])

    ys, rules, probas = [], [], []
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
            ys.append(y[test])
            rules.append(rule[test])
            probas.append(proba)

    y = np.concatenate(ys)
    rule = np.concatenate(rules)
    proba = np.vstack(probas)
    ml = proba.argmax(axis=1)

    def run(accept_agreement):
        decisions = [decide(r, p, accept_agreement=accept_agreement) for r, p in zip(rule, proba)]
        final = np.array([-1 if d[0] is None else d[0] for d in decisions])
        status = np.array([d[1] for d in decisions])
        return final, status

    print()
    print(f"Predictions analysed: {len(y)} ({args.seeds} seeds pooled)")
    print(f"ML alone, answering everything: {pct((ml == y).mean())}   "
          f"Rule alone: {pct((rule == y).mean())}")

    print()
    print("=" * 66)
    print(f"COVERAGE AND ACCURACY (threshold {ML_MIN_CONFIDENCE})")
    print("=" * 66)
    print(f"{'policy':<44}{'coverage':>10}{'accuracy':>10}")
    for label, flag in [
        ("confidence only (nested-validated policy)", False),
        ("confidence or agreement (agent default)" if ACCEPT_AGREEMENT
         else "confidence or agreement", True),
    ]:
        final, status = run(flag)
        accepted = final >= 0
        print(f"{label:<44}{pct(accepted.mean()):>10}"
              f"{pct((final[accepted] == y[accepted]).mean()):>10}")

    final, status = run(True)
    print()
    print("=" * 66)
    print("BY STATUS (policy with agreement)")
    print("=" * 66)
    print(f"{'status':<14}{'count':>8}{'share':>8}{'ML label':>10}{'rule label':>12}")
    for name in ("confident", "agreed", "abstained"):
        mask = status == name
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<14}{0:>8}")
            continue
        print(f"{name:<14}{n:>8}{pct(n / len(y)):>8}"
              f"{pct((ml[mask] == y[mask]).mean()):>10}"
              f"{pct((rule[mask] == y[mask]).mean()):>12}")

    print()
    print("'ML label' / 'rule label' = how often that estimate is right in the group.")
    print("If 'agreed' is much less accurate than 'confident', set ACCEPT_AGREEMENT = False")
    print("in agents/complexity_agent.py. If 'abstained' is mostly wrong for both, the")
    print("abstentions are justified.")


if __name__ == "__main__":
    main()