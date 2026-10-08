

import argparse

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

# Keep in sync with agents/complexity_agent.py
try:
    from agents.complexity_agent import ML_MIN_CONFIDENCE, MAX_CLASS_GAP
except Exception:
    ML_MIN_CONFIDENCE, MAX_CLASS_GAP = 0.50, 1

MIN_BUCKET = 100  # buckets smaller than this are flagged as unreliable


def collect(rows, folds, seeds):
    """Return dict of arrays with one entry per (seed, test program)."""
    features = [extract_features(r["src"]) for r in rows]
    extras = [extra_features(r["src"]) for r in rows]
    y = np.array([CLASS_ORDER.index(r["complexity"]) for r in rows])
    groups = np.array([r["problem"] for r in rows])
    rule = np.array([CLASS_ORDER.index(rule_based_estimate(f)) for f in features])
    lines = np.array([
        sum(1 for line in r["src"].splitlines() if line.strip()) for r in rows
    ])
    X = np.hstack([
        np.array([[f[n] for n in FEATURE_NAMES] for f in features], dtype=float),
        rule.reshape(-1, 1),
        np.array([[e[n] for n in EXTRA_NAMES] for e in extras], dtype=float),
    ])

    out = {k: [] for k in ("y", "ml", "conf", "rule", "lines", "seed")}
    for seed in range(seeds):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
            print(f"seed {seed} fold {fold}/{folds}", flush=True)
            model = ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            model.fit(X[train], y[train])
            raw = model.predict_proba(X[test])
            proba = np.zeros((len(test), len(CLASS_ORDER)))
            for column, cls in enumerate(model.classes_):
                proba[:, int(cls)] = raw[:, column]

            out["y"].append(y[test])
            out["ml"].append(proba.argmax(axis=1))
            out["conf"].append(proba.max(axis=1))
            out["rule"].append(rule[test])
            out["lines"].append(lines[test])
            out["seed"].append(np.full(len(test), seed))

    return {k: np.concatenate(v) for k, v in out.items()}


def pct(x):
    return f"{100 * x:5.1f}%"


def acc_row(label, mask, d, width=24):
    n = int(mask.sum())
    if n == 0:
        print(f"{label:<{width}}{0:>9}")
        return
    ml_acc = (d["ml"][mask] == d["y"][mask]).mean()
    rule_acc = (d["rule"][mask] == d["y"][mask]).mean()
    flag = "  (small)" if n < MIN_BUCKET else ""
    print(f"{label:<{width}}{n:>9}{pct(ml_acc):>11}{pct(rule_acc):>11}{flag}")


def table(title, first_col="", width=24):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"{first_col:<{width}}{'Samples':>9}{'ML acc':>11}{'Rule acc':>11}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = load_data(find_data_file(args.data))
    print("Samples:", len(rows))
    d = collect(rows, args.folds, args.seeds)

    y, ml, rule, conf, lines = d["y"], d["ml"], d["rule"], d["conf"], d["lines"]
    gap = np.abs(ml - rule)
    ml_ok, rule_ok = ml == y, rule == y
    agree = ml == rule
    dis = ~agree
    everything = np.ones(len(y), dtype=bool)

    print()
    print(f"Predictions analysed: {len(y)} ({args.seeds} seeds pooled; each program "
          f"appears once per seed)")

    table("OVERALL")
    acc_row("All predictions", everything, d)
    print(f"{'Oracle (either right)':<24}{len(y):>9}{pct((ml_ok | rule_ok).mean()):>11}")

    table("AGREEMENT VS DISAGREEMENT")
    acc_row("Agreement", agree, d)
    acc_row("Disagreement", dis, d)

    print()
    print("=" * 60)
    print("ML CONFIDENCE CALIBRATION (all predictions)")
    print("=" * 60)
    print(f"{'Confidence':<14}{'Samples':>9}{'Mean conf':>11}{'ML acc':>10}")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        label = f"{lo:.1f}-{min(hi, 1.0):.1f}"
        if m.sum():
            print(f"{label:<14}{int(m.sum()):>9}{conf[m].mean():>11.2f}{pct(ml_ok[m].mean()):>10}")
        else:
            print(f"{label:<14}{0:>9}")

    table("DISAGREEMENT BY ML CONFIDENCE", "Confidence")
    for label, lo, hi in [("< 0.40", 0, 0.40), ("0.40-0.60", 0.40, 0.60),
                          ("0.60-0.80", 0.60, 0.80), (">= 0.80", 0.80, 1.0001)]:
        acc_row(label, dis & (conf >= lo) & (conf < hi), d)

    table("DISAGREEMENT BY CLASS GAP", "Gap")
    acc_row("1", dis & (gap == 1), d)
    acc_row("2", dis & (gap == 2), d)
    acc_row("3+", dis & (gap >= 3), d)

    table("DISAGREEMENT BY PROGRAM LENGTH", "Length (non-empty)")
    for label, lo, hi in [("<= 8", 0, 8), ("9-15", 9, 15), ("16-30", 16, 30), ("31+", 31, 10 ** 9)]:
        acc_row(label, dis & (lines >= lo) & (lines <= hi), d)

    print()
    print("=" * 60)
    print("WHO IS RIGHT WHEN THEY DISAGREE")
    print("=" * 60)
    n_dis = int(dis.sum())
    for label, m in [("ML right", dis & ml_ok), ("Rule right", dis & rule_ok),
                     ("Both wrong", dis & ~ml_ok & ~rule_ok)]:
        print(f"{label:<14}{int(m.sum()):>9}{pct(m.sum() / max(n_dis, 1)):>10}")

    # ---- exploratory policies (same predictions, NOT tuned/validated) ----
    def policy_acc(final):
        return (final == y).mean()

    print()
    print("=" * 60)
    print("POLICIES (exploratory - thresholds are not validated)")
    print("=" * 60)
    print(f"{'Policy':<50}{'Acc':>8}")
    print(f"{'ML always':<50}{pct(policy_acc(ml)):>8}")
    print(f"{'Rule always':<50}{pct(policy_acc(rule)):>8}")
    agent = np.where(agree, ml,
                     np.where((conf < ML_MIN_CONFIDENCE) | (gap > MAX_CLASS_GAP), rule, ml))
    print(f"{f'Current agent (conf<{ML_MIN_CONFIDENCE} or gap>{MAX_CLASS_GAP})':<50}"
          f"{pct(policy_acc(agent)):>8}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        final = np.where(conf < t, rule, ml)
        print(f"{f'Rule if ML conf < {t:.2f}':<50}{pct(policy_acc(final)):>8}")
    for g in [2, 3, 4]:
        final = np.where(gap >= g, rule, ml)
        print(f"{f'Rule if gap >= {g}':<50}{pct(policy_acc(final)):>8}")

    print()
    print("Buckets marked '(small)' have fewer than", MIN_BUCKET, "predictions.")
    print("If a pattern looks useful, validate it with a nested split before")
    print("changing the agent; do not pick the best threshold from this table.")


if __name__ == "__main__":
    main()