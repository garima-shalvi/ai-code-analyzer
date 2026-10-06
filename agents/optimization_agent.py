"""
Optimization Agent (goal / utility based).

Goal:   reduce the computational cost of the analyzed program while
        preserving its behaviour.
State:  the parsed code + loop information from the Code Understanding module.
Actions (candidate recommendations):
        HOIST_SORT          - move a repeated sort out of a loop
        USE_SET_MEMBERSHIP  - replace list membership tests with a set
        REVIEW_NESTED_LOOPS - advisory: look for lookup / precomputation
Utility: 0.6 * gain - 0.15 * effort - 0.25 * risk   (each term in [0, 1])
         Candidates are ranked by utility, then reported as findings.

Recommendations are suggestions, never automatic rewrites: the agent cannot
prove that a change preserves behaviour.
"""

import ast

from representation.finding import create_finding
from agents.code_understanding import get_loop_bound_type

# Utility weights: performance matters most, correctness risk is penalised
# more heavily than implementation effort.
W_GAIN, W_EFFORT, W_RISK = 0.6, 0.15, 0.25
HIGH_PRIORITY, MEDIUM_PRIORITY = 0.40, 0.20

MUTATORS = {
    "append", "extend", "insert", "remove", "pop", "clear",
    "add", "discard", "update", "reverse", "sort",
}
LOOP_NODES = (ast.For, ast.While)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# ---------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------
def compute_utility(gain, effort, risk):
    return round(W_GAIN * gain - W_EFFORT * effort - W_RISK * risk, 3)


def priority_from_utility(utility):
    if utility >= HIGH_PRIORITY:
        return "high"
    if utility >= MEDIUM_PRIORITY:
        return "medium"
    return "low"


def _gain(base, loops):
    """Gain grows with loop depth and shrinks if every enclosing loop is
    constant-bounded (e.g. range(3)), where the saving is negligible."""
    gain = min(1.0, base + 0.05 * (len(loops) - 1))
    if all(get_loop_bound_type(loop) == "constant" for loop in loops):
        gain *= 0.3
    return gain


# ---------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------
def _collect(node, loops, out):
    """Record every node with the list of loops it executes inside.
    Comprehensions count as loops. A function body resets the loop stack."""
    out.append((node, loops))

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for child in ast.iter_child_nodes(node):
            _collect(child, [], out)
    elif isinstance(node, ast.For):
        # target and iterable are evaluated outside the repeated body
        _collect(node.target, loops, out)
        _collect(node.iter, loops, out)
        for stmt in node.body + node.orelse:
            _collect(stmt, loops + [node], out)
    elif isinstance(node, (ast.While,) + COMPREHENSIONS):
        for child in ast.iter_child_nodes(node):
            _collect(child, loops + [node], out)
    else:
        for child in ast.iter_child_nodes(node):
            _collect(child, loops, out)


def _modified_names(loop, ignore=None):
    """Names that are assigned or mutated anywhere inside the loop."""
    names = set()

    for n in ast.walk(loop):
        if n is ignore:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in MUTATORS
            and isinstance(n.func.value, ast.Name)
        ):
            names.add(n.func.value.id)
        elif (
            isinstance(n, ast.Subscript)
            and isinstance(n.ctx, (ast.Store, ast.Del))
            and isinstance(n.value, ast.Name)
        ):
            names.add(n.value.id)

    return names


def _value_kind(value):
    if isinstance(value, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(value, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)):
        return "hashed"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in ("list", "sorted"):
            return "list"
        if value.func.id in ("set", "dict", "frozenset"):
            return "hashed"
    return "other"


def _infer_kinds(tree):
    """Best-effort guess of what each name holds: list / hashed / param.
    Python has no static types, so this is only a heuristic."""
    kinds = {}

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in n.args.args + n.args.kwonlyargs:
                kinds.setdefault(arg.arg, "param")
        elif isinstance(n, ast.Assign):
            kind = _value_kind(n.value)
            for target in n.targets:
                if isinstance(target, ast.Name):
                    kinds[target.id] = kind

    return kinds


def _is_sort_call(call):
    f = call.func
    return (
        (isinstance(f, ast.Attribute) and f.attr == "sort")
        or (isinstance(f, ast.Name) and f.id == "sorted")
    )


# ---------------------------------------------------------------
# Pattern detectors: each returns candidate recommendations
# ---------------------------------------------------------------
def _candidate(rule_id, node, message, gain, effort, risk, confidence,
               current, potential, why):
    utility = compute_utility(gain, effort, risk)
    return {
        "rule_id": rule_id,
        "line": getattr(node, "lineno", 1),
        "column": getattr(node, "col_offset", 0),
        "message": message,
        "utility": utility,
        "priority": priority_from_utility(utility),
        "confidence": confidence,
        "evidence": (
            f"{why} | current: {current} | potential: {potential} "
            f"| utility={utility:.2f} (gain={gain:.2f}, effort={effort:.2f}, risk={risk:.2f})"
        ),
    }


def detect_sort_in_loop(nodes):
    candidates = []

    for node, loops in nodes:
        if not loops or not isinstance(node, ast.Call) or not _is_sort_call(node):
            continue

        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - {"sorted"}
        modified = _modified_names(loops[-1], ignore=node)

        # Only recommend hoisting when the sorted data does not change
        # inside the loop (loop-invariant). Otherwise stay silent.
        if used & modified:
            continue

        candidates.append(_candidate(
            "OPT_SORT_IN_LOOP", node,
            "Sorting is repeated inside a loop on data that does not change in the loop. "
            "Consider moving the sort outside the loop if the program's logic permits.",
            gain=_gain(0.8, loops), effort=0.2, risk=0.2, confidence=0.80,
            current="k iterations x O(n log n)", potential="O(n log n) once",
            why=f"sort of {sorted(used) or 'value'} inside loop at depth {len(loops)}",
        ))

    return candidates


def detect_list_membership_in_loop(tree, nodes):
    candidates = []
    kinds = _infer_kinds(tree)

    for node, loops in nodes:
        if not loops or not isinstance(node, ast.Compare):
            continue

        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if not isinstance(comparator, ast.Name):
                continue

            kind = kinds.get(comparator.id)
            if kind not in ("list", "param"):
                continue

            name = comparator.id
            changes = name in _modified_names(loops[-1])
            risk = 0.5 if changes else 0.15
            confidence = 0.80 if kind == "list" else 0.50
            hedge = "" if kind == "list" else f" (if '{name}' is a list)"
            extra = (
                f" '{name}' changes inside the loop, so keep a set updated alongside it."
                if changes else ""
            )

            candidates.append(_candidate(
                "OPT_LIST_MEMBERSHIP", node,
                f"Membership test on '{name}' inside a loop{hedge}. "
                f"Consider using a set for average O(1) lookups.{extra}",
                gain=_gain(0.9, loops), effort=0.2, risk=risk, confidence=confidence,
                current="O(n x m)", potential="O(n + m)",
                why=f"'{name}' inferred as {kind}; test at loop depth {len(loops)}",
            ))

    return candidates


def detect_nested_loops(representation):
    candidates = []

    for loop in representation["features"].get("loop_details", []):
        if loop["nesting_depth"] < 2 or loop["bound_type"] == "constant":
            continue

        class _At:  # minimal stand-in carrying the line number
            lineno = loop["line"]
            col_offset = 0

        candidates.append(_candidate(
            "OPT_NESTED_LOOPS", _At,
            "Nested loops over input-dependent bounds. Check whether the inner work "
            "can be replaced by a lookup structure or precomputation.",
            gain=0.4, effort=0.7, risk=0.5, confidence=0.50,
            current="O(n^2) or worse", potential="may be reducible",
            why=f"loop at depth {loop['nesting_depth']} with {loop['bound_type']} bound",
        ))
        break  # one advisory per program is enough

    return candidates


# ---------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------
def detect_optimizations(representation):
    try:
        tree = ast.parse(representation["code"])
    except (SyntaxError, KeyError):
        return []

    nodes = []
    _collect(tree, [], nodes)

    candidates = []
    candidates.extend(detect_sort_in_loop(nodes))
    candidates.extend(detect_list_membership_in_loop(tree, nodes))
    candidates.extend(detect_nested_loops(representation))

    # De-duplicate, then rank by utility (the agent's decision step)
    unique = {}
    for c in candidates:
        unique.setdefault((c["rule_id"], c["line"]), c)

    ranked = sorted(unique.values(), key=lambda c: -c["utility"])

    return [
        create_finding(
            rule_id=c["rule_id"],
            source="optimization_agent",
            category="optimization",
            message=f"[{c['priority'].upper()} priority] {c['message']}",
            line=c["line"],
            column=c["column"],
            severity=c["priority"],
            confidence=c["confidence"],
            evidence=c["evidence"],
        )
        for c in ranked
    ]


# ---------------------------------------------------------------
# Self-check: one positive and one negative case per pattern
# ---------------------------------------------------------------
EXAMPLES = {
    "sort in loop (positive)": ("""
def f(arr, n):
    for i in range(n):
        arr.sort()
""", {"OPT_SORT_IN_LOOP"}),
    "sort outside loop (negative)": ("""
def f(arr, n):
    arr.sort()
    for i in range(n):
        print(arr[i])
""", set()),
    "sort of changing data (negative)": ("""
def f(rows):
    for row in rows:
        sorted(row)
""", set()),
    "list membership (positive)": ("""
def find_common(a, b):
    result = []
    for x in a:
        if x in b:
            result.append(x)
    return result
""", {"OPT_LIST_MEMBERSHIP"}),
    "set membership (negative)": ("""
def find_common(a, b):
    lookup = set(b)
    result = []
    for x in a:
        if x in lookup:
            result.append(x)
    return result
""", set()),
    "nested loops (positive)": ("""
def f(n):
    t = 0
    for i in range(n):
        for j in range(n):
            t += i + j
    return t
""", {"OPT_NESTED_LOOPS"}),
    "constant nested loops (negative)": ("""
def f():
    t = 0
    for i in range(3):
        for j in range(3):
            t += i + j
    return t
""", set()),
}


if __name__ == "__main__":
    from agents.code_understanding import analyze_code

    failures = 0
    for name, (code, expected) in EXAMPLES.items():
        found = {f["rule_id"] for f in detect_optimizations(analyze_code(code))}
        status = "ok  " if found == expected else "FAIL"
        failures += found != expected
        print(f"{status} {name}: found={sorted(found)} expected={sorted(expected)}")

    print()
    demo = analyze_code(EXAMPLES["list membership (positive)"][0])
    for finding in detect_optimizations(demo):
        print(finding)

    print(f"\n{failures} failing example(s)")