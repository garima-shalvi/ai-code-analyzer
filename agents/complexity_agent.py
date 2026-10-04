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

    return findings
    
if __name__ == "__main__":
    from agents.code_understanding import analyze_code

    code = """
def test(n):
    if n > 0:
        for i in range(n):
            while i < n:
                if i > 2:
                    return test(n - 1)
    return 0
"""

    representation = analyze_code(code)

    findings = detect_complexity(representation)

    for finding in findings:
        print(finding)