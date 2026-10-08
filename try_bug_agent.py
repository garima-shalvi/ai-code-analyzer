

import sys

from agents.bug_detection import detect_bugs
from agents.code_understanding import analyze_code

PROGRAMS = {
    "grades.py": ("""
def average(scores, history=[]):
    total = 0
    count = 0
    for s in scores:
        total += s
    bonus = 5
    return total / count

def read_scores(path):
    try:
        f = open(path)
        return [int(x) for x in f]
    except:
        return []
""", {
        "MUTABLE_DEFAULT_ARGUMENT": "history=[] is shared between calls",
        "UNUSED_VARIABLE": "bonus is never used",
        "DIVISION_BY_ZERO": "count is always 0 when dividing",
        "BARE_EXCEPT": "except: hides every error",
    }),

    "parser.py": ("""
import re

def parse_age(line):
    m = re.match(r"age=(\\d+)", line)
    age = int(m.group(1))
    if age > 18:
        status = "adult"
    return status

def next_year():
    age = input("age: ")
    return age + 1
""", {
        "NONE_DEREFERENCE": "re.match can return None, then m.group fails",
        "POSSIBLY_UNDEFINED_VARIABLE": "status is only set when age > 18",
        "TYPE_MISMATCH": "input() is a string, so age + 1 fails",
    }),

    "lists.py": ("""
def clean(items):
    for x in items:
        if x is 0:
            items.remove(x)
    return items

def first_or_none(values):
    if values == None:
        return None
    list = sorted(values)
    return list[0]
""", {
        "IS_WITH_LITERAL": "x is 0 compares identity, not value",
        "MODIFY_WHILE_ITERATING": "items.remove inside a loop over items",
        "COMPARE_TO_NONE": "values == None should use 'is None'",
        "BUILTIN_SHADOWING": "list = ... hides the built-in list",
    }),

    "loops.py": ("""
def wait_forever():
    while True:
        pass

def countdown(n):
    return countdown(n - 1)
    print("done")
""", {
        "INFINITE_LOOP": "while True with no way out",
        "RECURSION_NO_BASE_CASE": "countdown never stops calling itself",
        "UNREACHABLE_CODE": "print after return never runs",
    }),

    "returns.py": ("""
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1

def normalise():
    text = input("name: ")
    text.strip()
    return text

def sorted_copy():
    values = [3, 1, 2]
    result = values.sort()
    return result
""", {
        "INCONSISTENT_RETURN": "sign(0) returns None",
        "IGNORED_RESULT": "text.strip() result is thrown away",
        "LIST_METHOD_RETURNS_NONE": "list.sort() returns None",
    }),

    "clean.py (no bugs)": ("""
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out

def total(values):
    s = 0
    for v in values:
        s += v
    return s
""", {}),
}


def show(findings):
    for f in sorted(findings, key=lambda f: (f["line"], f["column"])):
        print(
            f"  line {f['line']:>3}  [{f['severity'].upper():<6} {f['confidence']:.2f}] "
            f"{f['rule_id']:<26} {f['message'][:70]}"
        )


def run_file(path):
    with open(path, encoding="utf-8") as file:
        code = file.read()

    representation = analyze_code(code)
    if "error" in representation:
        print("Could not analyse the file:", representation["error"])
        return

    findings = detect_bugs(representation)
    print(f"{path}: {len(findings)} findings")
    show(findings)


def run_builtin():
    planted_total = found_total = extra_total = 0

    for name, (code, planted) in PROGRAMS.items():
        representation = analyze_code(code)
        findings = detect_bugs(representation)
        found_ids = {f["rule_id"] for f in findings}

        print("=" * 72)
        print(name)
        print("=" * 72)
        print("Planted bugs:")
        if not planted:
            print("  (none - the agent should report nothing)")
        for rule_id, why in planted.items():
            mark = "FOUND " if rule_id in found_ids else "MISSED"
            print(f"  {mark} {rule_id:<28} {why}")

        extra = [f for f in findings if f["rule_id"] not in planted]
        print("Extra findings (not planted):" if extra else "No extra findings.")
        show(extra)
        print()

        planted_total += len(planted)
        found_total += len([r for r in planted if r in found_ids])
        extra_total += len(extra)

    print("=" * 72)
    print(f"Planted bugs found: {found_total} of {planted_total}")
    print(f"Extra findings:     {extra_total} (check whether each one is a real issue)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        run_builtin()