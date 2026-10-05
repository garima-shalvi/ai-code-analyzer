def generate_report(result):
    findings = result["findings"]

    bugs = []
    complexity = []

    for finding in findings:
      if finding["source"] == "static_rule":
        bugs.append(finding)
      elif finding["source"] in ("complexity_rule", "complexity_analysis"):
        complexity.append(finding)

    print("AI Code Analysis Report")
    print("=======================")
    print(f"Total Findings: {len(findings)}")
    print(f"Bug Findings: {len(bugs)}")
    print(f"Complexity Findings: {len(complexity)}")

    print("\nFindings:")

    for finding in findings:
        print(f"\n[{finding['severity'].upper()}] {finding['rule_id']}")
        print(f"Category: {finding['category']}")
        print(f"Line: {finding['line']}")
        print(f"Message: {finding['message']}")
        print(f"Confidence: {finding['confidence'] * 100:.0f}%")
        print(f"Evidence: {finding['evidence']}")