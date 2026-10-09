
from agents.reporting_agent import generate_report, format_report


def check_summary(report):
    summary = report["summary"]
    expected_total = (
        len(report["bugs"])
        + len(report["complexity"])
        + len(report["optimizations"])
    )
    assert summary["total_findings"] == expected_total
    assert summary["bugs"] == len(report["bugs"])
    assert summary["complexity_findings"] == len(report["complexity"])
    assert summary["optimization_recommendations"] == len(report["optimizations"])


def test_valid_code():
    report = generate_report("def add(a, b):\n    return a + b\n")
    assert report["status"] == "success"
    check_summary(report)
    assert "AI CODE ANALYSIS REPORT" in format_report(report)
    print("PASS: valid code")


def test_invalid_syntax():
    report = generate_report("def broken(:\n    pass\n")
    assert report["status"] == "error"
    assert "ANALYSIS ERROR" in format_report(report)
    print("PASS: invalid syntax")


def test_empty_code():
    report = generate_report("")
    assert report["status"] == "success"
    check_summary(report)
    assert "AI CODE ANALYSIS REPORT" in format_report(report)
    print("PASS: empty code")


if __name__ == "__main__":
    test_valid_code()
    test_invalid_syntax()
    test_empty_code()
    print("All reporting agent tests passed!")