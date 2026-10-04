from representation.finding import create_finding

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
            evidence=f"Maximum nesting depth: {depth}"
        ))

    return findings



def detect_recursive_functions(representation):
    findings = []

    recursive_functions = representation["features"]["recursive_functions"]

    for function_name in recursive_functions:
        findings.append(create_finding(
            rule_id="RECURSIVE_FUNCTION",
            source="complexity_rule",
            category="recursion",
            message="Recursive function may increase execution cost because it repeatedly calls itself.",
            line=1,
            column=0,
            severity="medium",
            confidence=0.95,
            evidence=f"Recursive function: {function_name}"
        ))

    return findings
def detect_complexity(representation):
    findings = []

    findings.extend(detect_deep_nesting(representation))
    findings.extend(detect_recursive_functions(representation))
    findings.extend(estimate_time_complexity(representation))
    findings.extend(estimate_space_complexity(representation))

    return findings

def estimate_time_complexity(representation):
    findings = []

    loops = representation["features"]["loop_details"]

    if not loops:
        complexity = "O(1)"
    else:
        max_degree = 0

        for loop in loops:
            if loop["bound_type"] == "input-dependent":
                degree = loop["nesting_depth"]
            else:
                degree = loop["nesting_depth"] - 1

            max_degree = max(max_degree, degree)

        if max_degree == 0:
            complexity = "O(1)"
        elif max_degree == 1:
            complexity = "O(n)"
        else:
            complexity = f"O(n^{max_degree})"

    findings.append(create_finding(
        rule_id="TIME_COMPLEXITY",
        source="complexity_analysis",
        category="time-complexity",
        message=f"Estimated time complexity: {complexity}.",
        line=1,
        column=0,
        severity="info",
        confidence=0.90,
        evidence=f"Loop details: {loops}"
    ))

    return findings
def estimate_space_complexity(representation):
    findings = []

    space_level = representation["features"]["space_complexity"]

    if space_level == 0:
        complexity = "O(1)"
    elif space_level == 1:
        complexity = "O(n)"
    else:
        complexity = f"O(n^{space_level})"

    findings.append(create_finding(
        rule_id="SPACE_COMPLEXITY",
        source="complexity_analysis",
        category="space-complexity",
        message=f"Estimated space complexity: {complexity}.",
        line=1,
        column=0,
        severity="info",
        confidence=0.85,
        evidence=f"Input-dependent storage level: {space_level}"
    ))

    return findings

if __name__ == "__main__":
    from agents.code_understanding import analyze_code

    code = """
def test(n):
    arr = [1, 2, 3, 4]
    return arr
"""

    representation = analyze_code(code)

    findings = estimate_space_complexity(representation)

    for finding in findings:
        print(finding)