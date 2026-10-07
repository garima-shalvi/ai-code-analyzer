from agents.code_understanding import analyze_code
from agents.complexity_agent import estimate_time_complexity_ml, estimate_time_complexity

TEST_CASES = {
    "O(1)": """
def test():
    x = 10
    y = x + 5
    return y
""",

    "O(log n)": """
def test(n):
    while n > 1:
        n //= 2
""",

    "O(n)": """
def test(n):
    total = 0
    for i in range(n):
        total += i
    return total
""",

    "O(n log n)": """
def test(n):
    arr = []
    for i in range(n):
        arr.append(i)
    arr.sort()
    return arr
""",

    "O(n^2)": """
def test(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i + j
    return total
""",

    "O(2^n)": """
def test(n):
    if n <= 1:
        return 1
    return test(n - 1) + test(n - 1)
"""
}


for expected, code in TEST_CASES.items():

    representation = analyze_code(code)

    rule = estimate_time_complexity(representation)[0]
    ml = estimate_time_complexity_ml(representation)[0]

    print("=" * 60)
    print("EXPECTED:", expected)
    print("RULE:   ", rule["message"])
    print("ML:     ", ml["message"])
    print("CONF:   ", ml["confidence"])