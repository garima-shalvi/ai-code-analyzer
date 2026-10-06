from agents.bug_detection import detect_bugs
from agents.complexity_agent import detect_complexity
from agents.optimization_agent import detect_optimizations
from agents.code_understanding import analyze_code


def generate_report(code):
    representation = analyze_code(code)

    if "error" in representation:
        return {
            "status": "error",
            "error": representation
        }

    bugs = detect_bugs(representation)
    complexity = detect_complexity(representation)
    optimizations = detect_optimizations(representation)

    return {
        "status": "success",
        "summary": {
            "total_findings": len(bugs) + len(complexity) + len(optimizations),
            "bugs": len(bugs),
            "complexity_findings": len(complexity),
            "optimization_recommendations": len(optimizations)
        },
        "bugs": bugs,
        "complexity": complexity,
        "optimizations": optimizations
    }


def format_report(report):
    if report["status"] == "error":
        return (
            "==================================================\n"
            "           AI CODE ANALYSIS REPORT\n"
            "==================================================\n\n"
            "ANALYSIS ERROR\n"
            "--------------\n"
            f"{report['error']}\n"
        )

    summary = report["summary"]

    lines = []

    lines.append("==================================================")
    lines.append("           AI CODE ANALYSIS REPORT")
    lines.append("==================================================")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-------")
    lines.append(f"Bugs detected:             {summary['bugs']}")
    lines.append(f"Complexity findings:       {summary['complexity_findings']}")
    lines.append(
        f"Optimization suggestions:  {summary['optimization_recommendations']}"
    )
    lines.append(f"Total findings:            {summary['total_findings']}")
    lines.append("")

    lines.append("COMPLEXITY")
    lines.append("----------")

    if report["complexity"]:
        for finding in report["complexity"]:
            lines.append(finding["message"])
            lines.append(
                f"Confidence: {finding['confidence'] * 100:.1f}%"
            )
            lines.append(f"Evidence: {finding['evidence']}")
            lines.append("")
    else:
        lines.append("No complexity findings.")
        lines.append("")

    lines.append("BUGS")
    lines.append("----")

    if report["bugs"]:
        for finding in report["bugs"]:
            lines.append(
                f"[{finding['severity'].upper()}] {finding['message']}"
            )
            lines.append(f"Line: {finding['line']}")
            lines.append(
                f"Confidence: {finding['confidence'] * 100:.1f}%"
            )
            lines.append("")
    else:
        lines.append("No bugs detected.")
        lines.append("")

    lines.append("OPTIMIZATION RECOMMENDATIONS")
    lines.append("----------------------------")

    if report["optimizations"]:
        for finding in report["optimizations"]:
            lines.append(finding["message"])
            lines.append(f"Line: {finding['line']}")
            lines.append(
                f"Confidence: {finding['confidence'] * 100:.1f}%"
            )
            lines.append(f"Evidence: {finding['evidence']}")
            lines.append("")
    else:
        lines.append("No optimization recommendations.")
        lines.append("")

    lines.append("==================================================")

    return "\n".join(lines)


if __name__ == "__main__":
    code = """
def find_common(a, b):
    result = []
    for x in a:
        if x in b:
            result.append(x)
    return result
"""

    report = generate_report(code)
    print(format_report(report))