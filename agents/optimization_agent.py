
import ast

from representation.finding import create_finding
from agents.code_understanding import get_loop_bound_type

# Utility weights: performance matters most, correctness risk is penalised
# more heavily than implementation effort.
W_GAIN, W_EFFORT, W_RISK = 0.6, 0.15, 0.25
HIGH_PRIORITY, MEDIUM_PRIORITY = 0.40, 0.20

# "Do nothing" has utility 0, so a recommendation must beat that. Raise this
# (e.g. to 0.05) to also silence weak advisories such as OPT_NESTED_LOOPS.
MIN_UTILITY = 0.0

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

MUTATORS = {
    "append", "extend", "insert", "remove", "pop", "clear",
    "add", "discard", "update", "reverse", "sort",
}
# Calls known not to modify their arguments / receiver. Any OTHER call that
# receives a name (e.g. random.shuffle(arr), heappush(h, x), process(arr)) is
# assumed to possibly modify it.
PURE_CALLS = {
    "len", "sorted", "print", "range", "min", "max", "sum", "abs", "enumerate",
    "zip", "str", "int", "float", "bool", "list", "set", "tuple", "dict",
    "frozenset", "isinstance", "reversed", "any", "all", "round", "repr",
    "format", "sqrt", "floor", "ceil", "pow", "log", "hash", "ord", "chr",
    "count", "index", "get", "keys", "values", "items", "copy", "join",
    "startswith", "endswith", "lower", "upper", "strip", "split",
}
# A membership test against a list of at most this many constant elements is
# already cheap: a set would not help.
SMALL_LIST_MAX = 8
# Methods that only dicts / sets have: calling one on a parameter is strong
# evidence that it is NOT a list.
HASHED_METHODS = {"get", "items", "keys", "values", "setdefault", "add", "discard"}
LOOP_NODES = (ast.For, ast.While)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)



# Utility function

def compute_utility(gain, effort, risk):
    return round(W_GAIN * gain - W_EFFORT * effort - W_RISK * risk, 3)


def priority_from_utility(utility):
    if utility >= HIGH_PRIORITY:
        return "high"
    if utility >= MEDIUM_PRIORITY:
        return "medium"
    return "low"


def _gain(base, loops):
    """Gain grows with loop depth. If every enclosing loop has a constant
    bound (e.g. range(3)) the repeated work is O(1) in total, so the
    asymptotic gain is zero."""
    if loops and all(get_loop_bound_type(loop) == "constant" for loop in loops):
        return 0.0
    return min(1.0, base + 0.05 * (len(loops) - 1))



# AST helpers

def _collect(node, loops, out, scope):
    """Record every node with the loops it executes inside and the function
    (or module) scope it belongs to. Comprehensions count as loops. A
    function body resets the loop stack and starts a new scope."""
    out.append((node, loops, scope))

    if isinstance(node, FUNCTIONS):
        for child in ast.iter_child_nodes(node):
            _collect(child, [], out, node)
    elif isinstance(node, ast.For):
        # target and iterable are evaluated outside the repeated body
        _collect(node.target, loops, out, scope)
        _collect(node.iter, loops, out, scope)
        for stmt in node.body + node.orelse:
            _collect(stmt, loops + [node], out, scope)
    elif isinstance(node, (ast.While,) + COMPREHENSIONS):
        for child in ast.iter_child_nodes(node):
            _collect(child, loops + [node], out, scope)
    else:
        for child in ast.iter_child_nodes(node):
            _collect(child, loops, out, scope)


def _root_name(node):
    """self.items.append -> 'self', arr[i].x -> 'arr', arr -> 'arr'."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _call_name(call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _modified_names(loop, ignore=None):
    """Names that are assigned or possibly mutated anywhere inside the loop.
    Conservative: when unsure, a name counts as modified."""
    names = set()

    for n in ast.walk(loop):
        if n is ignore:
            continue

        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)

        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
            root = _root_name(n.value)
            if root:
                names.add(root)

        elif isinstance(n, ast.Call):
            name = _call_name(n)

            # receiver of a mutating method: arr.append(..), self.items.append(..)
            if isinstance(n.func, ast.Attribute):
                root = _root_name(n.func.value)
                if root and (name in MUTATORS or name not in PURE_CALLS):
                    names.add(root)

            # arguments of a call we cannot vouch for: random.shuffle(arr)
            if name not in PURE_CALLS and name not in MUTATORS:
                for arg in list(n.args) + [k.value for k in n.keywords]:
                    if isinstance(arg, ast.Starred):
                        arg = arg.value
                    root = _root_name(arg)
                    if root:
                        names.add(root)

    return names


def _value_kind(value):
    if isinstance(value, ast.List):
        if (
            value.elts
            and len(value.elts) <= SMALL_LIST_MAX
            and all(isinstance(e, ast.Constant) for e in value.elts)
        ):
            return "small"
        return "list"
    if isinstance(value, ast.ListComp):
        return "list"
    if isinstance(value, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)):
        return "hashed"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in ("list", "sorted"):
            return "list"
        if value.func.id in ("set", "dict", "frozenset"):
            return "hashed"
    return "other"


def _walk_own(scope):
    """Walk a scope's nodes without entering nested function definitions."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, FUNCTIONS):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _infer_kinds(tree):
    """Best-effort guess of what each name holds (list / hashed / param),
    kept PER SCOPE so a variable in one function cannot affect another.
    Returns {scope_node: {name: kind}}. Python has no static types, so this
    is only a heuristic."""
    scopes = {}

    def build(scope, inherited):
        kinds = dict(inherited)

        if isinstance(scope, FUNCTIONS):
            args = scope.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                kinds[arg.arg] = "param"

        for n in _walk_own(scope):
            if isinstance(n, ast.Assign):
                kind = _value_kind(n.value)
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        kinds[target.id] = kind

        # usage evidence: a parameter with .get() / .items() / .add() ... called
        # on it is a dict or set, not a list
        for n in _walk_own(scope):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in HASHED_METHODS
                and isinstance(n.func.value, ast.Name)
                and kinds.get(n.func.value.id) == "param"
            ):
                kinds[n.func.value.id] = "hashed"

        scopes[scope] = kinds

        for n in _walk_own(scope):
            if isinstance(n, FUNCTIONS):
                build(n, kinds)

    build(tree, {})
    return scopes


def _indexed_by_tested_key(name, key, loop):
    """True for the dict idiom `if k in d: ... d[k]`. A list indexed by the very
    value that was just tested for membership would make no sense."""
    key_dump = ast.dump(key)
    for n in ast.walk(loop):
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == name
            and ast.dump(n.slice) == key_dump
        ):
            return True
    return False


def _is_sort_call(call):
    f = call.func
    return (
        (isinstance(f, ast.Attribute) and f.attr == "sort")
        or (isinstance(f, ast.Name) and f.id == "sorted")
    )



# Pattern detectors: each returns candidate recommendations

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

    for node, loops, _scope in nodes:
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
    scope_kinds = _infer_kinds(tree)

    for node, loops, scope in nodes:
        if not loops or not isinstance(node, ast.Compare):
            continue

        kinds = scope_kinds.get(scope, {})

        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if not isinstance(comparator, ast.Name):
                continue

            kind = kinds.get(comparator.id)
            if kind not in ("list", "param", "small"):
                continue

            name = comparator.id
            changes = name in _modified_names(loops[-1])

            # tiny constant list that never changes: a set gains nothing
            if kind == "small":
                if not changes:
                    continue
                kind = "list"  # it grows in the loop, so it is a real list

            if _indexed_by_tested_key(name, node.left, loops[-1]):
                continue
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

def _writes_nested_subscript(loop):
    for node in ast.walk(loop):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Subscript)
                ):
                    return True
    return False

def detect_nested_loops(representation):
    candidates = []

    try:
        tree = ast.parse(representation["code"])
    except (SyntaxError, KeyError):
        return candidates

    loops_by_line = {
        node.lineno: node
        for node in ast.walk(tree)
        if isinstance(node, LOOP_NODES)
    }

    for loop in representation["features"].get("loop_details", []):
        if loop["nesting_depth"] < 2 or loop["bound_type"] == "constant":
            continue

        node = loops_by_line.get(loop["line"])
        if node is not None and _writes_nested_subscript(node):
            continue

        class _At:
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
        break

    return candidates



# Agent entry point

def detect_optimizations(representation):
    try:
        tree = ast.parse(representation["code"])
    except (SyntaxError, KeyError):
        return []

    nodes = []
    _collect(tree, [], nodes, tree)

    candidates = []
    candidates.extend(detect_sort_in_loop(nodes))
    candidates.extend(detect_list_membership_in_loop(tree, nodes))
    candidates.extend(detect_nested_loops(representation))

    # Decision step: drop anything that does not beat "do nothing",
    # de-duplicate, then rank by utility.
    unique = {}
    for c in candidates:
        if c["utility"] <= MIN_UTILITY:
            continue
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



# Self-check: one positive and one negative case per pattern

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