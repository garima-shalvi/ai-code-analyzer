
import sys

from agents.code_understanding import analyze_code
from agents.optimization_agent import detect_optimizations

SORT, MEMBER, NESTED = "OPT_SORT_IN_LOOP", "OPT_LIST_MEMBERSHIP", "OPT_NESTED_LOOPS"

CASES = [
    # ---------------- DETECT ----------------
    ("DETECT", "sort of unchanged list inside loop", """
def process(arr, n):
    for _ in range(n):
        arr.sort()
""", {SORT}),

    ("DETECT", "sorted() of unchanged data inside while loop", """
def f(data, queries):
    i = 0
    while i < len(queries):
        ordered = sorted(data)
        print(ordered[0] + queries[i])
        i += 1
""", {SORT}),

    ("DETECT", "list membership against a list built in the function", """
def find_common(a, b):
    b = list(b)
    result = []
    for x in a:
        if x in b:
            result.append(x)
    return result
""", {MEMBER}),

    ("DETECT", "classic order-preserving dedupe with a list", """
def dedupe(items):
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen
""", {MEMBER}),

    ("DETECT", "membership inside a comprehension", """
def common(a, b):
    b = list(b)
    return [x for x in a if x in b]
""", {MEMBER}),

    ("DETECT", "nested loops over two inputs", """
def pairs(a, b):
    total = 0
    for x in a:
        for y in b:
            total += x * y
    return total
""", {NESTED}),

    # ---------------- SAFE ----------------
    ("SAFE", "already uses a set", """
def find_common(a, b):
    lookup = set(b)
    result = []
    for x in a:
        if x in lookup:
            result.append(x)
    return result
""", set()),

    ("SAFE", "sort already outside the loop", """
def f(arr, n):
    arr.sort()
    for i in range(n):
        print(arr[i])
""", set()),

    ("SAFE", "dict membership", """
def f(keys, xs):
    table = {k: 1 for k in keys}
    return [x for x in xs if x in table]
""", set()),

    ("SAFE", "membership on a constant-size loop", """
def f(b):
    b = list(b)
    for i in range(3):
        if i in b:
            print(i)
""", set()),

    # ---------------- UNSAFE ----------------
    ("UNSAFE", "data appended then sorted each iteration", """
def f(arr, n):
    for i in range(n):
        arr.append(i)
        arr.sort()
""", set()),

    ("UNSAFE", "data replaced inside the loop", """
def f(rows):
    for row in rows:
        row = sorted(row)
        print(row)
""", set()),

    ("UNSAFE", "list shrinks while sorted (pop)", """
def f(arr):
    while arr:
        arr.sort()
        arr.pop()
""", set()),

    ("UNSAFE", "list reshuffled by a helper, then sorted", """
import random
def f(arr, n):
    for _ in range(n):
        random.shuffle(arr)
        arr.sort(key=lambda v: -v)
""", set()),

    ("UNSAFE", "attribute list appended, then sorted", """
class Board:
    def run(self, n):
        for i in range(n):
            self.items.append(i)
            self.items.sort()
""", set()),

    # ---------------- CLEAN ----------------
    ("CLEAN", "fizzbuzz + running total", """
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        else:
            out.append(str(i))
    return out

def total(values):
    s = 0
    for v in values:
        s += v
    return s
""", set()),

    ("CLEAN", "single loop, no sort, no membership", """
def count_pos(nums):
    c = 0
    for x in nums:
        if x > 0:
            c += 1
    return c
""", set()),

    # ---------------- TRICKY ----------------
    ("TRICKY", "grid traversal (nested loops, nothing to optimise)", """
def fill(n):
    grid = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = i + j
    return grid
""", set()),

    ("TRICKY", "nested loops with range(0, 3) constant bounds", """
def f():
    t = 0
    for i in range(0, 3):
        for j in range(0, 3):
            t += i + j
    return t
""", set()),

    ("TRICKY", "tiny constant vowel list in a loop", """
def count_vowels(s):
    vowels = ['a', 'e', 'i', 'o', 'u']
    n = 0
    for ch in s:
        if ch in vowels:
            n += 1
    return n
""", set()),

    ("TRICKY", "parameter is really a dict", """
def lookup_all(table, keys):
    out = []
    for k in keys:
        if k in table:
            out.append(table[k])
    return out
""", set()),

    ("TRICKY", "same variable name is a set in another function", """
def build(items):
    seen = set(items)
    return seen

def dedupe(items, seen):
    out = []
    for x in items:
        if x in seen:
            out.append(x)
    return out
""", {MEMBER}),

    ("TRICKY", "small list that grows in the loop (still a real list)", """
def f(items):
    seen = [0]
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen
""", {MEMBER}),

    ("TRICKY", "parameter used like a dict through .get()", """
def f(d, xs):
    for x in xs:
        if x in d:
            print(d.get(x))
""", set()),

    ("TRICKY", "list parameter indexed by position, not by the tested key", """
def f(a, b):
    for i in range(len(a)):
        if a[i] in b:
            print(b[0])
""", {MEMBER}),
]


def show(findings):
    for f in sorted(findings, key=lambda f: (f["line"], f["column"])):
        print(f"      line {f['line']:>3}  [{f['severity'].upper():<6} conf {f['confidence']:.2f}] "
              f"{f['rule_id']}")


def run_file(path):
    with open(path, encoding="utf-8") as file:
        code = file.read()
    representation = analyze_code(code)
    if "error" in representation:
        print("Could not analyse the file:", representation["error"])
        return
    findings = detect_optimizations(representation)
    print(f"{path}: {len(findings)} recommendations")
    show(findings)


def run_suite():
    true_rec = missed = false_rec = 0
    by_group = {}
    problems = []

    for group, name, code, expected in CASES:
        found_findings = detect_optimizations(analyze_code(code))
        found = {f["rule_id"] for f in found_findings}

        hit = found & expected
        miss = expected - found
        extra = found - expected

        true_rec += len(hit)
        missed += len(miss)
        false_rec += len(extra)

        stats = by_group.setdefault(group, [0, 0, 0, 0, 0])
        stats[0] += 1
        stats[1] += 0 if (miss or extra) else 1
        stats[2] += len(hit)
        stats[3] += len(miss)
        stats[4] += len(extra)

        status = "ok  " if not (miss or extra) else "FAIL"
        print(f"{status} [{group:<6}] {name}")
        if miss or extra:
            if miss:
                print(f"       MISSED opportunity : {sorted(miss)}")
            if extra:
                print(f"       FALSE recommendation: {sorted(extra)}")
                show([f for f in found_findings if f["rule_id"] in extra])
            problems.append((group, name))

    print()
    print("=" * 64)
    print(f"{'group':<8}{'cases':>7}{'passed':>8}{'true':>6}{'missed':>8}{'false':>7}")
    for group in ("DETECT", "SAFE", "UNSAFE", "CLEAN", "TRICKY"):
        c, p, t, m, e = by_group[group]
        print(f"{group:<8}{c:>7}{p:>8}{t:>6}{m:>8}{e:>7}")
    print("=" * 64)
    print(f"True recommendations : {true_rec}")
    print(f"Missed opportunities : {missed}")
    print(f"False recommendations: {false_rec}")
    precision = true_rec / (true_rec + false_rec) if (true_rec + false_rec) else 1.0
    recall = true_rec / (true_rec + missed) if (true_rec + missed) else 1.0
    print(f"Precision {precision:.2f} | Recall {recall:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        run_suite()
