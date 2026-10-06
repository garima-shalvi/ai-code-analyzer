"""
Complexity-oriented features for Python programs.

Design notes
------------
* Uses only the standard library (ast), so it can be tested on its own.
* Loops include comprehensions / generator expressions (Python code in
  competitive programming uses them heavily).
* Loops are split into constant-bound loops (range(10), a literal list),
  logarithmic loops (while-loops that halve / double / take a modulus
  step, e.g. binary search, digit loops, Euclid) and ordinary
  input-dependent loops.
* The usual "for _ in range(int(input()))" test-case wrapper is not
  counted as a nesting level.
* Implicit linear work (sum(x), max(x), x.count(...)) inside loops is
  counted, because it hides a nested loop.
"""

import ast
import warnings

FEATURE_NAMES = [
    # basic structure
    "loop_count",
    "for_count",
    "while_count",
    "condition_count",
    "call_count",
    "subscript_access_count",
    "max_nesting_depth",
    # loop structure
    "max_loop_nesting",
    "max_dependent_loop_nesting",
    "max_poly_loop_nesting",
    "max_log_loop_nesting",
    "constant_loop_count",
    "log_loop_count",
    "has_testcase_loop",
    # recursion
    "is_recursive",
    "max_self_calls",
    "uses_memo",
    # library / operation patterns
    "sort_calls",
    "sort_in_loop",
    "heap_ops",
    "bisect_ops",
    "implicit_linear_total",
    "implicit_linear_in_loop",
    "list_alloc_in_loop",
    "exp_patterns",
    "sqrt_patterns",
    # size / parsing
    "function_length",
    "parse_ok",
]

TESTCASE_NAMES = {
    "t", "T", "tc", "tt", "tests", "test", "cases", "testcases",
    "test_cases", "nt", "num_tests", "tcs", "ntc",
}

IMPLICIT_LINEAR_FUNCS = {"sum", "max", "min", "any", "all", "sorted", "reversed"}
IMPLICIT_LINEAR_METHODS = {"count", "index", "copy", "extend", "join", "remove", "insert"}
EXP_FUNCS = {"permutations", "combinations", "product", "combinations_with_replacement"}


# ---------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------

def _call_name(node):
    """Return the plain name of a call target: f(...) -> 'f', a.b(...) -> 'b'."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_const_expr(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_const_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_const_expr(node.left) and _is_const_expr(node.right)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_const_expr(e) for e in node.elts)
    return False


def _is_range_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    )


def _is_constant_iter(it):
    if _is_range_call(it):
        return all(_is_const_expr(a) for a in it.args)
    if isinstance(it, (ast.List, ast.Tuple, ast.Set, ast.Constant)):
        return _is_const_expr(it)
    if isinstance(it, ast.Dict):
        return all(
            (k is None or _is_const_expr(k)) and _is_const_expr(v)
            for k, v in zip(it.keys, it.values)
        )
    if (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id in {"enumerate", "reversed", "sorted", "list", "tuple", "set", "zip"}
    ):
        return all(_is_constant_iter(a) for a in it.args)
    return False


def _is_input_call(node):
    return (
        isinstance(node, ast.Call)
        and _call_name(node) in {"input", "raw_input", "readline"}
    )


def _is_testcase_loop(node):
    """for _ in range(int(input())) / for _ in range(t)."""
    if not isinstance(node, ast.For) or not _is_range_call(node.iter):
        return False
    if len(node.iter.args) != 1:
        return False

    arg = node.iter.args[0]

    if (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Name)
        and arg.func.id == "int"
        and arg.args
        and _is_input_call(arg.args[0])
    ):
        return True

    if isinstance(arg, ast.Name) and arg.id in TESTCASE_NAMES:
        return True

    return False


def _is_step_stmt(node):
    """Statements that shrink or grow a value geometrically (log-style loops)."""
    if isinstance(node, ast.AugAssign):
        if (
            isinstance(node.op, (ast.FloorDiv, ast.Div, ast.RShift, ast.Mult, ast.LShift))
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and node.value.value >= 2
        ):
            return True

    if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
        value = node.value
        if (
            isinstance(value.op, (ast.FloorDiv, ast.Div, ast.RShift))
            and isinstance(value.right, ast.Constant)
            and isinstance(value.right.value, (int, float))
            and value.right.value >= 2
        ):
            return True

        # (lo + hi) // 2 wrapped in parentheses is covered above;
        # gcd style: a, b = b, a % b
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
        for element in node.value.elts:
            if isinstance(element, ast.BinOp) and isinstance(element.op, ast.Mod):
                return True

    return False


def _is_log_loop(node):
    if not isinstance(node, ast.While):
        return False
    for child in ast.walk(node):
        if _is_step_stmt(child):
            return True
    return False


# ---------------------------------------------------------------
# main visitor
# ---------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.max_loop_nesting = 0
        self.max_dependent = 0
        self.max_poly = 0
        self.max_log = 0
        self.constant_loops = 0
        self.log_loops = 0
        self.has_testcase_loop = 0
        self.sort_calls = 0
        self.sort_in_loop = 0
        self.heap_ops = 0
        self.bisect_ops = 0
        self.implicit_total = 0
        self.implicit_in_loop = 0
        self.list_alloc_in_loop = 0
        self.exp_patterns = 0
        self.sqrt_patterns = 0
        self.loop_count = 0
        self.for_count = 0
        self.while_count = 0


def _record_loop(stats, depth, dep, poly, log):
    stats.max_loop_nesting = max(stats.max_loop_nesting, depth)
    stats.max_dependent = max(stats.max_dependent, dep)
    stats.max_poly = max(stats.max_poly, poly)
    stats.max_log = max(stats.max_log, log)


def _visit(node, depth, dep, poly, log, stats):
    if isinstance(node, (ast.For, ast.AsyncFor)):
        if _is_testcase_loop(node):
            stats.has_testcase_loop = 1
            for child in ast.iter_child_nodes(node):
                _visit(child, depth, dep, poly, log, stats)
            return

        stats.loop_count += 1
        stats.for_count += 1

        constant = _is_constant_iter(node.iter)
        if constant:
            stats.constant_loops += 1
            n_depth, n_dep, n_poly, n_log = depth + 1, dep, poly, log
        else:
            n_depth, n_dep, n_poly, n_log = depth + 1, dep + 1, poly + 1, log

        _record_loop(stats, n_depth, n_dep, n_poly, n_log)

        _visit(node.iter, depth, dep, poly, log, stats)
        for child in node.body + node.orelse:
            _visit(child, n_depth, n_dep, n_poly, n_log, stats)
        return

    if isinstance(node, ast.While):
        stats.loop_count += 1
        stats.while_count += 1

        if _is_log_loop(node):
            stats.log_loops += 1
            n_depth, n_dep, n_poly, n_log = depth + 1, dep + 1, poly, log + 1
        else:
            n_depth, n_dep, n_poly, n_log = depth + 1, dep + 1, poly + 1, log

        _record_loop(stats, n_depth, n_dep, n_poly, n_log)

        _visit(node.test, n_depth, n_dep, n_poly, n_log, stats)
        for child in node.body + node.orelse:
            _visit(child, n_depth, n_dep, n_poly, n_log, stats)
        return

    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        c_depth, c_dep, c_poly, c_log = depth, dep, poly, log

        for generator in node.generators:
            stats.loop_count += 1
            stats.for_count += 1
            _visit(generator.iter, c_depth, c_dep, c_poly, c_log, stats)

            if _is_constant_iter(generator.iter):
                stats.constant_loops += 1
                c_depth += 1
            else:
                c_depth += 1
                c_dep += 1
                c_poly += 1

            _record_loop(stats, c_depth, c_dep, c_poly, c_log)

            for condition in generator.ifs:
                _visit(condition, c_depth, c_dep, c_poly, c_log, stats)

        if isinstance(node, ast.DictComp):
            _visit(node.key, c_depth, c_dep, c_poly, c_log, stats)
            _visit(node.value, c_depth, c_dep, c_poly, c_log, stats)
        else:
            _visit(node.elt, c_depth, c_dep, c_poly, c_log, stats)
        return

    # ----- non-loop nodes: record patterns, then recurse -----
    in_loop = depth > 0

    if isinstance(node, ast.Call):
        name = _call_name(node)

        if name in {"sorted", "sort"}:
            stats.sort_calls += 1
            if in_loop:
                stats.sort_in_loop += 1

        if name is not None and name.startswith("heap"):
            stats.heap_ops += 1
        if name is not None and name.startswith("bisect"):
            stats.bisect_ops += 1
        if name in EXP_FUNCS:
            stats.exp_patterns += 1
        if name in {"sqrt", "isqrt"}:
            stats.sqrt_patterns += 1

        implicit = False
        if isinstance(node.func, ast.Name) and name in IMPLICIT_LINEAR_FUNCS:
            implicit = len(node.args) == 1 and not isinstance(node.args[0], ast.Constant)
        elif isinstance(node.func, ast.Attribute) and name in IMPLICIT_LINEAR_METHODS:
            implicit = True
        if implicit:
            stats.implicit_total += 1
            if in_loop:
                stats.implicit_in_loop += 1

    elif isinstance(node, ast.BinOp):
        # 1 << n, 2 ** n with a non-constant exponent
        if (
            isinstance(node.op, (ast.LShift, ast.Pow))
            and isinstance(node.left, ast.Constant)
            and node.left.value in (1, 2)
            and not _is_const_expr(node.right)
        ):
            stats.exp_patterns += 1

        # [0] * n created inside a loop
        if isinstance(node.op, ast.Mult) and in_loop:
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if isinstance(side, ast.List) and not _is_const_expr(other):
                    stats.list_alloc_in_loop += 1
                    break

        # n ** 0.5
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 0.5
        ):
            stats.sqrt_patterns += 1

    elif isinstance(node, ast.Compare):
        # i * i <= n
        operands = [node.left] + list(node.comparators)
        for operand in operands:
            if (
                isinstance(operand, ast.BinOp)
                and isinstance(operand.op, ast.Mult)
                and isinstance(operand.left, ast.Name)
                and isinstance(operand.right, ast.Name)
                and operand.left.id == operand.right.id
            ):
                stats.sqrt_patterns += 1
                break

    for child in ast.iter_child_nodes(node):
        _visit(child, depth, dep, poly, log, stats)


def _function_features(tree):
    is_recursive = 0
    max_self_calls = 0
    uses_memo = 0
    longest_function = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            longest_function = max(
                longest_function, (node.end_lineno or node.lineno) - node.lineno + 1
            )

            self_calls = 0
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == node.name:
                        self_calls += 1
                    elif isinstance(child.func, ast.Attribute) and child.func.attr == node.name:
                        self_calls += 1

            if self_calls > 0:
                is_recursive = 1
            max_self_calls = max(max_self_calls, self_calls)

            for decorator in node.decorator_list:
                text = ast.dump(decorator)
                if "lru_cache" in text or "'cache'" in text:
                    uses_memo = 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and ("memo" in node.id.lower() or "cache" in node.id.lower()):
            uses_memo = 1

    return is_recursive, max_self_calls, uses_memo, longest_function


def _max_branch_nesting(node, depth=0):
    maximum = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.If)):
            maximum = max(maximum, _max_branch_nesting(child, depth + 1))
        else:
            maximum = max(maximum, _max_branch_nesting(child, depth))
    return maximum


def empty_features():
    return {name: 0 for name in FEATURE_NAMES}


def extract_features(code):
    features = empty_features()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        features["function_length"] = len(code.splitlines())
        return features

    stats = _Stats()
    _visit(tree, 0, 0, 0, 0, stats)

    condition_count = 0
    call_count = 0
    subscript_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition_count += 1
        elif isinstance(node, ast.Call):
            call_count += 1
        elif isinstance(node, ast.Subscript):
            subscript_count += 1

    is_recursive, max_self_calls, uses_memo, longest_function = _function_features(tree)

    features.update({
        "loop_count": stats.loop_count,
        "for_count": stats.for_count,
        "while_count": stats.while_count,
        "condition_count": condition_count,
        "call_count": call_count,
        "subscript_access_count": subscript_count,
        "max_nesting_depth": _max_branch_nesting(tree),
        "max_loop_nesting": stats.max_loop_nesting,
        "max_dependent_loop_nesting": stats.max_dependent,
        "max_poly_loop_nesting": stats.max_poly,
        "max_log_loop_nesting": stats.max_log,
        "constant_loop_count": stats.constant_loops,
        "log_loop_count": stats.log_loops,
        "has_testcase_loop": stats.has_testcase_loop,
        "is_recursive": is_recursive,
        "max_self_calls": max_self_calls,
        "uses_memo": uses_memo,
        "sort_calls": stats.sort_calls,
        "sort_in_loop": stats.sort_in_loop,
        "heap_ops": stats.heap_ops,
        "bisect_ops": stats.bisect_ops,
        "implicit_linear_total": stats.implicit_total,
        "implicit_linear_in_loop": stats.implicit_in_loop,
        "list_alloc_in_loop": stats.list_alloc_in_loop,
        "exp_patterns": stats.exp_patterns,
        "sqrt_patterns": stats.sqrt_patterns,
        "function_length": max(len(code.splitlines()), longest_function),
        "parse_ok": 1,
    })

    return features


# ---------------------------------------------------------------
# rule-based estimator (the human-written baseline)
# ---------------------------------------------------------------

CLASS_ORDER = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "np"]
def rule_based_estimate(features):
    if features["exp_patterns"] > 0:
        return "np"
    if features["max_self_calls"] >= 2 and not features["uses_memo"]:
        return "np"

    degree = features["max_poly_loop_nesting"]

    has_log = (
        features["max_log_loop_nesting"] > 0
        or features["sort_calls"] > 0
        or features["heap_ops"] > 0
        or features["bisect_ops"] > 0
    )

    if features["implicit_linear_in_loop"] > 0 and degree >= 1:
        degree += 1

    if (features["sort_in_loop"] > 0 or features["heap_ops"] > 0 or features["bisect_ops"] > 0) and degree >= 1:
        degree += 1

    if degree >= 3:
        return "cubic"
    if degree == 2:
        return "quadratic"
    if degree == 1:
        return "nlogn" if has_log else "linear"
    if has_log:
        return "logn" if features["sort_calls"] == 0 else "nlogn"
    return "constant"

