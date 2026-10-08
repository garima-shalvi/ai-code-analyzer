
import os

import joblib
import numpy as np

from representation.finding import create_finding
from ml.complexity_features import extract_features, rule_based_estimate
from ml.extra_features import extra_features

MODEL_PATH = "models/complexity_model.joblib"

ML_MIN_CONFIDENCE = 0.50
ACCEPT_AGREEMENT = True

# Measured accuracy of the rule engine alone in grouped cross-validation.
RULE_ONLY_CONFIDENCE = 0.50

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


# Time complexity

def estimate_time_complexity(representation):
    """Rule-based estimate only (kept for tests and comparison)."""
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
        confidence=RULE_ONLY_CONFIDENCE,
        evidence=f"Rule-based complexity estimate: {complexity}",
    )]


def decide(rule_idx, proba, threshold=None, accept_agreement=None):
    """Decide what to report. Returns (class_index or None, status).

    A pure function of the rule estimate and the model's probabilities, so
    ml/validate_policy.py evaluates exactly the logic the agent runs.
    """
    threshold = ML_MIN_CONFIDENCE if threshold is None else threshold
    accept_agreement = ACCEPT_AGREEMENT if accept_agreement is None else accept_agreement

    ml_idx = int(np.argmax(proba))
    ml_conf = float(proba[ml_idx])

    if ml_conf >= threshold:
        return ml_idx, "confident"
    if accept_agreement and ml_idx == rule_idx:
        return ml_idx, "agreed"
    return None, "abstained"


def _ml_probabilities(features, rule_idx, data, code):
    """Model probabilities as a vector in class_order order."""
    values = [features[name] for name in data["feature_names"]] + [rule_idx]

    # models saved with extra features also need those, in the saved order
    extra_names = data.get("extra_names")
    if extra_names:
        extras = extra_features(code)
        values += [extras[name] for name in extra_names]

    X = np.array(values, dtype=float).reshape(1, -1)

    model = data["model"]
    expected = getattr(model, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        raise ValueError(
            f"model expects {expected} features but {X.shape[1]} were built; "
            f"retrain with: python -m ml.save_complexity_model"
        )

    raw = model.predict_proba(X)[0]
    proba = np.zeros(len(data["class_order"]))
    for column, cls in enumerate(model.classes_):
        proba[int(cls)] = raw[column]
    return proba


def _rule_only_finding(rule_label, reason):
    return create_finding(
        rule_id="TIME_COMPLEXITY",
        source="complexity_rule",
        category="time-complexity",
        message=f"Time complexity: {rule_label} (rule-based; {reason}).",
        line=1,
        column=0,
        severity="info",
        confidence=RULE_ONLY_CONFIDENCE,
        evidence=f"Rule estimate: {rule_label} | ML: unavailable ({reason})",
    )


def estimate_time_complexity_hybrid(representation):
    """Final time-complexity finding: a class, or UNKNOWN when evidence is too weak."""
    code = representation["code"]
    features = extract_features(code)
    rule_label = rule_based_estimate(features)

    data = load_complexity_model()
    if data is None:
        return [_rule_only_finding(rule_label, "ML model not found")]

    order = data["class_order"]
    rule_idx = order.index(rule_label)

    try:
        proba = _ml_probabilities(features, rule_idx, data, code)
    except Exception as exc:
        return [_rule_only_finding(rule_label, f"ML unavailable: {exc}")]

    final_idx, status = decide(rule_idx, proba)

    ml_idx = int(np.argmax(proba))
    ml_label = order[ml_idx]
    ml_conf = float(proba[ml_idx])
    agree = ml_label == rule_label

    if status == "confident":
        final = ml_label
        note = "ML and rule engine agree" if agree else (
            f"ML prediction; rule engine estimated {rule_label}"
        )
    elif status == "agreed":
        final = ml_label
        note = "ML and rule engine agree, but ML confidence is low"
    else:
        final = "UNKNOWN"
        note = (
            f"insufficient ML confidence; "
            f"ML suggested {ml_label}, rule engine estimated {rule_label}"
        )

    ranked = sorted(range(len(proba)), key=lambda i: -proba[i])[:3]
    top3 = ", ".join(f"{order[i]}={proba[i]:.2f}" for i in ranked)

    return [create_finding(
        rule_id="HYBRID_TIME_COMPLEXITY",
        source="hybrid_rule_ml",
        category="time-complexity",
        message=f"Time complexity: {final} ({note}).",
        line=1,
        column=0,
        severity="info",
        confidence=round(ml_conf, 3),
        evidence=(
            f"Status: {status} | Rule estimate: {rule_label} | "
            f"ML estimate: {ml_label} ({ml_conf:.2f}) | "
            f"Agreement: {'yes' if agree else 'no'} | ML top3: {top3}"
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