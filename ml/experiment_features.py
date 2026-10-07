"""
Can the 7-class complexity model be made more accurate?

Run from the project root:
    python -m ml.experiment_features
    python -m ml.experiment_features --data path/to/python_data.jsonl

Ideas tested, all with grouped cross-validation (split by problem, no leakage)
and identical folds for every model:

  A. current model: ExtraTrees (balanced) on your 29 features + rule estimate
  B. A + "extra" features: counts of AST node types, operators and common
     function calls (a cheap bag-of-patterns that lets the trees find signals
     your hand-made features miss)
  C. B with a different max_features setting
  D. ordinal model: six yes/no models "is the class above k?" combined into
     class probabilities (the classes are ordered, so this uses that order)
  E. average of ExtraTrees and Random Forest on B's features
"""

import argparse
import ast
import json
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

# ---------------------------------------------------------------
# Extra features: node types, operators, common calls
# ---------------------------------------------------------------
NODE_TYPES = [
    "For", "While", "If", "IfExp", "Call", "Subscript", "Attribute", "BinOp",
    "UnaryOp", "Compare", "BoolOp", "ListComp", "SetComp", "DictComp",
    "GeneratorExp", "Lambda", "FunctionDef", "ClassDef", "Return", "Assign",
    "AugAssign", "AnnAssign", "Break", "Continue", "Try", "With", "Import",
    "ImportFrom", "Dict", "List", "Set", "Tuple", "Slice", "Starred", "Yield",
    "Global", "Assert", "Raise", "Pass", "Delete", "Name", "Constant",
    "JoinedStr", "NamedExpr", "Expr",
]
OPS = [
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitAnd", "BitOr", "BitXor",
]
CALLS = [
    "range", "len", "input", "int", "map", "list", "sorted", "sort", "sum",
    "max", "min", "abs", "print", "append", "pop", "popleft", "appendleft",
    "heappush", "heappop", "heapify", "bisect_left", "bisect_right", "bisect",
    "insort", "deque", "Counter", "defaultdict", "set", "dict", "join",
    "split", "strip", "count", "index", "find", "replace", "sqrt", "pow",
    "lru_cache", "cache", "enumerate", "zip", "reversed", "reverse", "extend",
    "insert", "remove", "add", "discard", "get", "keys", "values", "items",
    "upper", "lower", "isdigit", "format", "str", "float", "bin", "gcd",
    "factorial", "permutations", "combinations", "product", "accumulate",
    "copy", "deepcopy", "setdefault", "most_common", "exit", "any", "all",
]
EXTRA_NAMES = (
    ["n_" + t for t in NODE_TYPES]
    + ["op_" + o for o in OPS]
    + ["call_" + c for c in CALLS]
    + ["in_ops", "in_ops_in_loop", "distinct_names"]
)


def extra_features(code):
    vec = dict.fromkeys(EXTRA_NAMES, 0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except Exception:
        return vec

    def has_in(node):
        return isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        )

    names = set()
    for node in ast.walk(tree):
        key = "n_" + type(node).__name__
        if key in vec:
            vec[key] += 1

        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            op_key = "op_" + type(node.op).__name__
            if op_key in vec:
                vec[op_key] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name and "call_" + name in vec:
                vec["call_" + name] += 1
        elif isinstance(node, ast.Name):
            names.add(node.id)

        if has_in(node):
            vec["in_ops"] += 1

    for loop in ast.walk(tree):
        if isinstance(loop, (ast.For, ast.While)):
            vec["in_ops_in_loop"] += sum(1 for n in ast.walk(loop) if has_in(n))

    vec["distinct_names"] = len(names)
    return vec



# Model wrappers: every model returns a (n, 7) probability matrix

class Plain:
    def __init__(self, estimator):
        self.est = estimator

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_full(self, X):
        raw = self.est.predict_proba(X)
        out = np.zeros((len(X), len(CLASS_ORDER)))
        for column, cls in enumerate(self.est.classes_):
            out[:, int(cls)] = raw[:, column]
        return out


class Ordinal:
    """Six binary models 'is the class > k?' combined into class probabilities."""

    def __init__(self, make_estimator):
        self.make = make_estimator

    def fit(self, X, y):
        self.models = []
        for k in range(len(CLASS_ORDER) - 1):
            target = (y > k).astype(int)
            if target.min() == target.max():
                self.models.append(float(target[0]))  # constant answer
            else:
                self.models.append(self.make().fit(X, target))
        return self

    def predict_full(self, X):
        above = []
        for model in self.models:
            if isinstance(model, float):
                above.append(np.full(len(X), model))
            else:
                above.append(model.predict_proba(X)[:, list(model.classes_).index(1)])
        above = np.minimum.accumulate(np.column_stack(above), axis=1)

        probs = np.zeros((len(X), len(CLASS_ORDER)))
        probs[:, 0] = 1.0 - above[:, 0]
        for k in range(1, len(CLASS_ORDER) - 1):
            probs[:, k] = above[:, k - 1] - above[:, k]
        probs[:, -1] = above[:, -1]

        probs = np.clip(probs, 0, None)
        totals = probs.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return probs / totals


class Average:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_full(self, X):
        return np.mean([m.predict_full(X) for m in self.models], axis=0)


def extra_trees(n=500, max_features="sqrt", leaf=2):
    return ExtraTreesClassifier(
        n_estimators=n, min_samples_leaf=leaf, max_features=max_features,
        class_weight="balanced", random_state=42, n_jobs=-1)


def random_forest(n=400):
    return RandomForestClassifier(
        n_estimators=n, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    rows = load_data(find_data_file(args.data))
    print("Samples:", len(rows))

    features = [extract_features(row["src"]) for row in rows]
    extras = [extra_features(row["src"]) for row in rows]
    y = np.array([CLASS_ORDER.index(row["complexity"]) for row in rows])
    groups = np.array([row["problem"] for row in rows])
    rule = np.array([CLASS_ORDER.index(rule_based_estimate(f)) for f in features])

    X_base = np.hstack([
        np.array([[f[n] for n in FEATURE_NAMES] for f in features], dtype=float),
        rule.reshape(-1, 1),
    ])
    X_extra = np.array([[e[n] for n in EXTRA_NAMES] for e in extras], dtype=float)
    X_all = np.hstack([X_base, X_extra])
    print("Features: base", X_base.shape[1], "| base + extra", X_all.shape[1])

    experiments = [
        ("A  ExtraTrees, base features (current)", X_base, lambda: Plain(extra_trees())),
        ("B  ExtraTrees, base + extra features", X_all, lambda: Plain(extra_trees())),
        ("C  ExtraTrees, base + extra, max_features 0.3", X_all, lambda: Plain(extra_trees(max_features=0.3))),
        ("D  Ordinal ExtraTrees, base + extra", X_all, lambda: Ordinal(lambda: extra_trees(n=300))),
        ("E  ExtraTrees + RandomForest average, base + extra", X_all,
         lambda: Average([Plain(extra_trees()), Plain(random_forest())])),
    ]

    results = {"Rule engine alone": []}
    for label, _, _ in experiments:
        results[(label, "argmax")] = []
        results[(label, "median")] = []

    for seed in range(args.seeds):
        splitter = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X_base, y, groups), 1):
            print(f"seed {seed} fold {fold}/{args.folds}", flush=True)
            results["Rule engine alone"].append(metrics(y[test], rule[test]))

            for label, X, factory in experiments:
                model = factory().fit(X[train], y[train])
                proba = model.predict_full(X[test])
                results[(label, "argmax")].append(metrics(y[test], decode_argmax(proba)))
                results[(label, "median")].append(metrics(y[test], decode_median(proba)))

    def accs(entries):
        return np.array([e["acc"] for e in entries])

    def mean(entries, key):
        return float(np.mean([e[key] for e in entries]))

    rule_acc = accs(results["Rule engine alone"])
    current_acc = accs(results[(experiments[0][0], "argmax")])

    print()
    header = (f"{'model | decoding':<62}{'acc':>7}{'macroF1':>9}{'within1':>9}"
              f"{'MAE':>7}{'vs rule':>9}{'vs A':>8}")
    print(header)
    print("-" * len(header))

    entries = results["Rule engine alone"]
    print(f"{'Rule engine alone':<62}{mean(entries, 'acc'):>7.3f}{mean(entries, 'f1'):>9.3f}"
          f"{mean(entries, 'within1'):>9.3f}{mean(entries, 'mae'):>7.3f}")

    ranked = [(accs(v).mean(), k, v) for k, v in results.items() if k != "Rule engine alone"]
    ranked.sort(key=lambda item: -item[0])
    for _, (label, decode), entries in ranked:
        acc = accs(entries)
        print(f"{label + ' | ' + decode:<62}{acc.mean():>7.3f}{mean(entries, 'f1'):>9.3f}"
              f"{mean(entries, 'within1'):>9.3f}{mean(entries, 'mae'):>7.3f}"
              f"{(acc - rule_acc).mean():>+9.3f}{(acc - current_acc).mean():>+8.3f}")

    print()
    print("'vs rule' / 'vs A' = mean accuracy difference on the same folds.")
    print("Only a clear gain over A (about +0.01 or more) is worth switching for.")


if __name__ == "__main__":
    main()