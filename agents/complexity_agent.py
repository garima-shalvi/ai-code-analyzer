import os

import joblib
import numpy as np

from representation.finding import create_finding
from ml.complexity_features import extract_features, rule_based_estimate
from ml.extra_features import extra_features

MODEL_PATH = "models/complexity_model.joblib"


ML_MIN_CONFIDENCE = 0.50
MAX_CLASS_GAP = 1

_complexity_model = None


def load_complexity_model():
    """Load the trained model once. Returns None if it has not been trained yet."""
    global _complexity_model

    if _complexity_model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _complexity_model = joblib.load(MODEL_PATH)

    return _complexity_model


# Rule-only detectors

def detect_deep_nesting(representation):
    findings = []
    depth = representation["features"]["max_nesting_depth"]

    if depth >= 4:
        findings.append(create_finding(
            rule_id="DEEP_NESTING",
            source="complexity_rule",
            category="deep-nesting",
            message="Code has deeply nested control flow and may be difficult to understand or maintain.",
            line=1,
            column=0,
            severity="medium",
            confidence=0.95,
            evidence=f"Maximum nesting depth: {depth}",
        ))

    return findings


def detect_recursive_functions(representation):
    findings = []

    for function_name in representation["features"]["recursive_functions"]:
        findings.append(create_finding(
            rule_id="RECURSIVE_FUNCTION",
            source="complexity_rule",
            category="recursion",
            message="Recursive function may increase execution cost because it repeatedly calls itself.",
            line=1,
            column=0,
            severity="medium",
            confidence=0.95,
            evidence=f"Recursive function: {function_name}",
        ))

    return findings


def estimate_space_complexity(representation):
    space_level = representation["features"]["space_complexity"]

    if space_level == 0:
        complexity = "O(1)"
    elif space_level == 1:
        complexity = "O(n)"
    else:
        complexity = f"O(n^{space_level})"

    return [create_finding(
        rule_id="SPACE_COMPLEXITY",
        source="complexity_analysis",
        category="space-complexity",
        message=f"Estimated space complexity: {complexity}.",
        line=1,
        column=0,
        severity="info",
        confidence=0.85,
        evidence=f"Input-dependent storage level: {space_level}",
    )]



# Time complexity: rule, ML, and the hybrid that combines them

def estimate_time_complexity(representation):
    
    features = extract_features(representation["code"])
    complexity = rule_based_estimate(features)

    return [create_finding(
        rule_id="TIME_COMPLEXITY",
        source="complexity_rule",
        category="time-complexity",
        message=f"Estimated time complexity: {complexity}.",
        line=1,
        column=0,
        severity="info",
        confidence=0.90,
        evidence=f"Rule-based complexity estimate: {complexity}",
    )]


def _ml_predict(features, rule_idx, data, code):
    """Return (ml_index, ml_confidence, top3_text) for one program."""
    order = data["class_order"]

    values = [features[name] for name in data["feature_names"]] + [rule_idx]

    # models saved with extra features also need those, in the saved order
    extra_names = data.get("extra_names")
    if extra_names:
        extras = extra_features(code)
        values += [extras[name] for name in extra_names]

    X = np.array(values, dtype=float).reshape(1, -1)

    probs = data["model"].predict_proba(X)[0]
    classes = [int(c) for c in data["model"].classes_]
    ranked = sorted(zip(classes, probs), key=lambda p: -p[1])

    ml_idx, ml_conf = ranked[0]
    top3 = ", ".join(f"{order[i]}={p:.2f}" for i, p in ranked[:3])
    return ml_idx, float(ml_conf), top3


def estimate_time_complexity_hybrid(representation):
    """
    Final time-complexity answer. The trained model decides the class (the
    rule engine's estimate is one of its input features); the rule estimate
    is reported as a second opinion:
      - they agree                              -> that class, high confidence
      - they disagree, ML sure and close        -> ML class, ML probability
      - ML unsure (< 0.50) or 2+ classes away   -> rule class (safety net)
    If the model file is missing, the rule-based estimate is used instead.
    """
    features = extract_features(representation["code"])
    rule_label = rule_based_estimate(features)

    data = load_complexity_model()

    # No trained model available: fall back to rules only.
    if data is None:
        return [create_finding(
            rule_id="TIME_COMPLEXITY",
            source="complexity_rule",
            category="time-complexity",
            message=f"Time complexity: {rule_label} (rule-based; ML model not found).",
            line=1,
            column=0,
            severity="info",
            confidence=0.70,
            evidence=f"Rule estimate: {rule_label} | ML: unavailable",
        )]

    order = data["class_order"]
    rule_idx = order.index(rule_label)
    ml_idx, ml_conf, top3 = _ml_predict(
        features, rule_idx, data, representation["code"]
    )
    ml_label = order[ml_idx]

    # The trained model decides the class (it beats the rule engine in grouped
    # cross-validation). A safety net falls back to the rule estimate when the
    # model is unsure or contradicts the rules by a wide margin.
    gap = abs(ml_idx - rule_idx)

    if ml_label == rule_label:
        final, conf = ml_label, min(0.99, 0.6 + ml_conf / 2)
        note = "ML and rule engine agree"
    elif ml_conf < ML_MIN_CONFIDENCE or gap > MAX_CLASS_GAP:
        final, conf = rule_label, 0.65
        note = f"rule engine used; ML suggested {ml_label} ({ml_conf:.2f}), too unsure or too far off"
    else:
        final, conf = ml_label, ml_conf
        note = f"ML prediction; rule engine estimated {rule_label}"

    agreement = "yes" if ml_label == rule_label else "no"

    return [create_finding(
        rule_id="HYBRID_TIME_COMPLEXITY",
        source="hybrid_rule_ml",
        category="time-complexity",
        message=f"Time complexity: {final} ({note}).",
        line=1,
        column=0,
        severity="info",
        confidence=round(conf, 3),
        evidence=(
            f"Rule estimate: {rule_label} | ML estimate: {ml_label} ({ml_conf:.2f}) "
            f"| Agreement: {agreement} | ML top3: {top3}"
        ),
    )]


# Backwards-compatible name used by ml/test_complexity_cases.py
estimate_time_complexity_ml = estimate_time_complexity_hybrid


def detect_complexity(representation):
    findings = []
    findings.extend(detect_deep_nesting(representation))
    findings.extend(detect_recursive_functions(representation))
    findings.extend(estimate_time_complexity_hybrid(representation))
    findings.extend(estimate_space_complexity(representation))
    return findings


if __name__ == "__main__":
    from agents.code_understanding import analyze_code

    code = """
def test():
    for i in range(n):
        arr.sort()
"""

    representation = analyze_code(code)

    for finding in detect_complexity(representation):
        print(finding)