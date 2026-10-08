from agents.bug_detection import detect_bugs
from agents.code_understanding import analyze_code
TESTS = [
    {
        "name": "average_empty_list",
        "code": """
def average(values):
    total = sum(values)
    return total / len(values)
""",
        "expected": [],
    },
    {
        "name": "missing_dict_value",
        "code": """
def increase_age(data):
    age = data.get("age")
    return age + 1
""",
        "expected": ["TYPE_MISMATCH"],
    },
    {
        "name": "list_modification",
        "code": """
def remove_invalid(items):
    for item in items:
        if item < 0:
            items.remove(item)
    return items
""",
        "expected": ["MODIFY_WHILE_ITERATING"],
    },
    {
        "name": "intentional_server_loop",
        "code": """
def run_server():
    while True:
        request = get_request()
        if request == "shutdown":
            break
        handle(request)
""",
        "expected": [],
    },
    {
        "name": "undefined_branch",
        "code": """
def get_message(flag):
    if flag:
        message = "hello"
    return message
""",
        "expected": ["POSSIBLY_UNDEFINED_VARIABLE"],
    },
    {
        "name": "safe_branch_assignment",
        "code": """
def get_message(flag):
    if flag:
        message = "hello"
    else:
        message = "goodbye"
    return message
""",
        "expected": [],
    },
    {
        "name": "zero_division",
        "code": """
def calculate(x):
    denominator = 0
    return x / denominator
""",
        "expected": ["DIVISION_BY_ZERO"],
    },
    {
        "name": "safe_division",
        "code": """
def calculate(x, denominator):
    if denominator == 0:
        return 0
    return x / denominator
""",
        "expected": [],
    },
    {
        "name": "none_dereference",
        "code": """
import re

def extract(text):
    match = re.match(r"ID:(.*)", text)
    return match.group(1)
""",
        "expected": ["NONE_DEREFERENCE"],
    },
    {
        "name": "checked_match",
        "code": """
import re

def extract(text):
    match = re.match(r"ID:(.*)", text)
    if match is None:
        return None
    return match.group(1)
""",
        "expected": [],
    },
    {
        "name": "mutable_default",
        "code": """
def add_item(item, items=[]):
    items.append(item)
    return items
""",
        "expected": ["MUTABLE_DEFAULT_ARGUMENT"],
    },
    {
        "name": "safe_default",
        "code": """
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
""",
        "expected": [],
    },
    {
        "name": "bare_except",
        "code": """
def load_data():
    try:
        return read_file()
    except:
        return None
""",
        "expected": ["BARE_EXCEPT"],
    },
    {
        "name": "specific_exception",
        "code": """
def load_data():
    try:
        return read_file()
    except FileNotFoundError:
        return None
""",
        "expected": [],
    },
    {
        "name": "ignored_strip",
        "code": """
def clean_name(name):
    name.strip()
    return name
""",
        "expected": ["IGNORED_RESULT"],
    },
    {
        "name": "intentional_ignored_append",
        "code": """
def add_name(names, name):
    names.append(name)
    return names
""",
        "expected": [],
    },
    {
        "name": "list_sort_return",
        "code": """
def smallest(values):
    result = values.sort()
    return result[0]
""",
        "expected": ["LIST_METHOD_RETURNS_NONE"],
    },
    {
        "name": "correct_sort",
        "code": """
def smallest(values):
    values.sort()
    return values[0]
""",
        "expected": [],
    },
    {
        "name": "none_comparison",
        "code": """
def check(value):
    if value == None:
        return True
    return False
""",
        "expected": ["COMPARE_TO_NONE"],
    },
    {
        "name": "correct_none_comparison",
        "code": """
def check(value):
    if value is None:
        return True
    return False
""",
        "expected": [],
    },
    {
        "name": "builtin_shadow",
        "code": """
def process(values):
    list = values
    return list
""",
        "expected": ["BUILTIN_SHADOWING"],
    },
    {
        "name": "local_name_not_builtin",
        "code": """
def process(values):
    result = values
    return result
""",
        "expected": [],
    },
    {
        "name": "unreachable_after_return",
        "code": """
def calculate(x):
    if x > 0:
        return x
        print("never runs")
    return 0
""",
        "expected": ["UNREACHABLE_CODE"],
    },
    {
        "name": "reachable_code",
        "code": """
def calculate(x):
    if x > 0:
        print("positive")
        return x
    return 0
""",
        "expected": [],
    },
    {
        "name": "recursion_without_base",
        "code": """
def countdown(n):
    print(n)
    countdown(n - 1)
""",
        "expected": ["RECURSION_NO_BASE_CASE"],
    },
    {
        "name": "valid_recursion",
        "code": """
def countdown(n):
    if n <= 0:
        return
    countdown(n - 1)
""",
        "expected": [],
    },
    {
        "name": "inconsistent_return",
        "code": """
def absolute(x):
    if x >= 0:
        return x
""",
        "expected": ["INCONSISTENT_RETURN"],
    },
    {
        "name": "consistent_return",
        "code": """
def absolute(x):
    if x >= 0:
        return x
    return -x
""",
        "expected": [],
    },
    {
        "name": "used_before_assignment",
        "code": """
def calculate():
    total = total + 10
    return total
""",
        "expected": ["USED_BEFORE_ASSIGNMENT"],
    },
    {
        "name": "proper_assignment",
        "code": """
def calculate():
    total = 0
    total = total + 10
    return total
""",
        "expected": [],
    },
    {
        "name": "input_type_error",
        "code": """
def next_age():
    age = input("Age: ")
    return age + 1
""",
        "expected": ["TYPE_MISMATCH"],
    },
    {
        "name": "input_converted",
        "code": """
def next_age():
    age = int(input("Age: "))
    return age + 1
""",
        "expected": [],
    },
    {
        "name": "tuple_assertion",
        "code": """
def check():
    assert (1, 2)
    return True
""",
        "expected": ["ASSERT_ON_TUPLE"],
    },
    {
        "name": "normal_assertion",
        "code": """
def check(value):
    assert value > 0
    return True
""",
        "expected": [],
    },
    {
        "name": "unused_variable",
        "code": """
def calculate(x):
    temporary = x * 10
    result = x + 5
    return result
""",
        "expected": ["UNUSED_VARIABLE"],
    },
    {
        "name": "used_variable",
        "code": """
def calculate(x):
    temporary = x * 10
    return temporary + 5
""",
        "expected": [],
    },
]


def run_tests():
    total = len(TESTS)
    passed = 0
    failed = 0
    total_expected = 0
    total_found = 0

    print("=" * 80)
    print("BLIND BUG DETECTION EVALUATION")
    print("=" * 80)

    for test in TESTS:
        representation = analyze_code(test["code"])
        findings = detect_bugs(representation)
        found = {finding["rule_id"] for finding in findings}
        expected = set(test["expected"])

        missing = expected - found
        extra = found - expected

        total_expected += len(expected)
        total_found += len(found)

        if not missing and not extra:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"\n{test['name']}")
        print(f"  {status}")
        print(f"  Expected: {sorted(expected) if expected else 'None'}")
        print(f"  Found:    {sorted(found) if found else 'None'}")

        if missing:
            print(f"  MISSED:   {sorted(missing)}")

        if extra:
            print(f"  EXTRA:    {sorted(extra)}")

    print("\n" + "=" * 80)
    print(f"Tests passed:     {passed}/{total}")
    print(f"Tests failed:     {failed}/{total}")
    print(f"Expected bugs:    {total_expected}")
    print(f"Detected bugs:    {total_found}")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()