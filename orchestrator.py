
import sys

from agents.reporting_agent import format_report, generate_report


def analyze_source(code):
    """Run every agent on the code and return the structured report (a dict).

    If the code has a syntax error, the dict has status "error" and says why;
    otherwise it contains the ranked findings.
    """
    return generate_report(code)


def analyze_file(path):
    with open(path, encoding="utf-8") as file:
        return analyze_source(file.read())


DEMO = {
    "program with bugs": '''
def average(scores, history=[]):
    total = 0
    count = 0
    for s in scores:
        total += s
    return total / count
''',
    "program with a syntax error": '''
def broken:
    print("hello")
''',
}


def main(argv):
    if len(argv) > 1:
        try:
            print(format_report(analyze_file(argv[1])))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Could not read {argv[1]}: {exc}")
        return

    for title, code in DEMO.items():
        print(f"##### {title} #####")
        print(format_report(analyze_source(code)))
        print()


if __name__ == "__main__":
    main(sys.argv)