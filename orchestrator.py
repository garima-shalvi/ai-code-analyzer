from agents.code_understanding import analyze_code
from agents.bug_detection import detect_bugs
from agents.complexity_agent import detect_complexity

def analyze_source(code):
    representation = analyze_code(code)

    if "error" in representation:
        return {
            "representation": representation,
            "findings": []
        }

    findings = []

    findings.extend(detect_bugs(representation))
    findings.extend(detect_complexity(representation))

    return {
        "representation": representation,
        "findings": findings
    }
if __name__ == "__main__":
    code = """
def process(items=[]):
    try:
        if len(items) > 0:
            for item in items:
                while item > 0:
                    if item > 10:
                        return process(items)
        else:
            return None
    except:
        pass

    return items
    print("unreachable")
"""

    result = analyze_source(code)

    for finding in result["findings"]:
        print(finding)