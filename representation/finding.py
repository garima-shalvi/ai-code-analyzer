def create_finding(
    rule_id,
    source,
    category,
    message,
    line,
    column,
    severity,
    confidence,
    evidence
):
    return {
        "rule_id": rule_id,
        "source": source,
        "category": category,
        "message": message,
        "line": line,
        "column": column,
        "severity": severity,
        "confidence": confidence,
        "evidence": evidence
    }