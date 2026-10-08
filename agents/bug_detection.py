

import ast
import builtins
import operator

from representation.finding import create_finding

BUILTIN_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__path__", "__class__",
}

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_TYPES = FUNC_TYPES + (ast.Lambda, ast.ClassDef)
COMP_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
LOOP_TYPES = (ast.For, ast.AsyncFor, ast.While)
TRY_TYPES = tuple(
    t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t
)

SHADOW_NAMES = {
    "list", "dict", "set", "str", "int", "float", "bool", "tuple", "sum",
    "min", "max", "len", "sorted", "input", "range", "type", "map", "filter",
    "all", "any", "next", "iter", "zip", "open",
}
MUTATING_METHODS = {"append", "extend", "insert", "remove", "pop", "clear"}
LIST_NONE_METHODS = {"sort", "append", "extend", "insert", "remove", "reverse", "clear"}
STR_METHODS_IGNORED = {
    "strip", "lstrip", "rstrip", "replace", "lower", "upper", "title",
    "capitalize", "swapcase", "casefold", "center", "ljust", "rjust",
    "zfill", "format", "join", "split", "splitlines",
}
STR_RETURNS_STR = {
    "lower", "upper", "strip", "lstrip", "rstrip", "replace", "title",
    "capitalize", "swapcase", "casefold", "center", "ljust", "rjust",
    "zfill", "format", "join",
}
BUILTIN_RESULT = {
    "str": "str", "int": "int", "float": "float", "bool": "bool", "len": "int",
    "input": "str", "list": "list", "dict": "dict", "set": "set",
    "tuple": "tuple", "repr": "str", "ord": "int", "chr": "str",
    "sorted": "list", "isinstance": "bool",
}
DISCARDED_BUILTINS = {"sorted", "reversed", "len", "sum", "min", "max", "abs", "round"}

ZERO_CATCH = {"ZeroDivisionError", "ArithmeticError", "Exception", "BaseException"}
ATTR_CATCH = {"AttributeError", "TypeError", "Exception", "BaseException"}
TYPE_CATCH = {"TypeError", "Exception", "BaseException"}

OP_SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
}
CMP_FUNCS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}

NUM = {"int", "float", "bool"}
SEQ = {"str", "list", "tuple", "bytes"}
UNKNOWN = object()  # "value not known" marker for constants


# ===============================================================
# Generic helpers
# ===============================================================
def _src(node, limit=80):
    try:
        text = ast.unparse(node)
    except Exception:
        text = type(node).__name__
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _make(rule_id, category, message, node, severity, confidence,
          evidence=None, source="static_rule"):
    return create_finding(
        rule_id=rule_id,
        source=source,
        category=category,
        message=message,
        line=getattr(node, "lineno", 1),
        column=getattr(node, "col_offset", 0),
        severity=severity,
        confidence=confidence,
        evidence=evidence if evidence is not None else _src(node),
    )


def _parent_map(tree):
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _own_nodes(scope):
    """Nodes of a scope without descending into nested functions/classes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, SCOPE_TYPES):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_none_const(node):
    return isinstance(node, ast.Constant) and node.value is None


def _is_const_true(node):
    return isinstance(node, ast.Constant) and bool(node.value)


def _is_exit_call(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in ("exit", "quit")
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in ("sys", "os")
        and func.attr in ("exit", "_exit", "abort")
    )


def _target_names(target):
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _has_loop_exit(loop):
    """True if a loop body contains a break for this loop, a return, a raise
    or a call that ends the program."""
    def visit(node, in_inner_loop):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, SCOPE_TYPES):
                continue
            if isinstance(child, (ast.Return, ast.Raise)):
                return True
            if isinstance(child, ast.Break) and not in_inner_loop:
                return True
            if isinstance(child, ast.Call) and _is_exit_call(child):
                return True
            if isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await)):
                return True  # generators / async loops are stopped by their consumer
            inner = in_inner_loop or isinstance(child, LOOP_TYPES)
            if visit(child, inner):
                return True
        return False

    return visit(loop, False)


def _inside_try_body(node, parents):
    """True if node sits in the body of a try that has except handlers
    (an exception can then end a loop that has no explicit exit)."""
    child = node
    parent = parents.get(child)
    while parent is not None and not isinstance(parent, FUNC_TYPES):
        if isinstance(parent, TRY_TYPES) and parent.handlers and child in parent.body:
            return True
        child, parent = parent, parents.get(parent)
    return False


def _following_exit(node, parents):
    """True if a break/return/raise follows the statement containing node in
    the same block."""
    while node in parents and not isinstance(node, ast.stmt):
        node = parents[node]
    parent = parents.get(node)

    for field in ("body", "orelse", "finalbody"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and node in block:
            index = block.index(node)
            return any(
                isinstance(s, (ast.Break, ast.Return, ast.Raise))
                for s in block[index + 1:]
            )
    return False


# ===============================================================
# Layer 1: simple reflex rules (stateless AST patterns)
# ===============================================================
def detect_mutable_default_arguments(tree):
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, FUNC_TYPES):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]

            for default in defaults:
                is_literal = isinstance(default, (ast.List, ast.Dict, ast.Set))
                is_call = (
                    isinstance(default, ast.Call)
                    and isinstance(default.func, ast.Name)
                    and default.func.id in ("list", "dict", "set")
                )
                if is_literal or is_call:
                    findings.append(_make(
                        "MUTABLE_DEFAULT_ARGUMENT", "mutable-default-argument",
                        "Mutable objects should not be used as default function "
                        "arguments; the same object is shared between calls.",
                        default, "medium", 0.99,
                    ))

    return findings


def detect_bare_except(tree):
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(_make(
                "BARE_EXCEPT", "bare-except",
                "Bare except catches all exceptions and can hide unexpected errors.",
                node, "medium", 0.99, evidence="except:",
            ))

    return findings


def detect_swallowed_exceptions(tree):
    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue

        only_pass = all(
            isinstance(s, ast.Pass)
            or (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and s.value.value is Ellipsis
            )
            for s in node.body
        )
        if only_pass:
            findings.append(_make(
                "SWALLOWED_EXCEPTION", "swallowed-exception",
                "The exception is caught and silently ignored, which hides failures.",
                node, "medium", 0.85, evidence=f"except {_src(node.type)}: pass",
            ))

    return findings


def detect_unreachable_code(tree, parents):
    findings = []

    def check_nested(statement):
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            check_block(statement.body)
            check_block(statement.orelse)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            check_block(statement.body)
        elif isinstance(statement, TRY_TYPES):
            check_block(statement.body)
            for handler in statement.handlers:
                check_block(handler.body)
            check_block(statement.orelse)
            check_block(statement.finalbody)

    def check_block(statements):
        terminated = False

        for statement in statements:
            if terminated:
                findings.append(_make(
                    "UNREACHABLE_CODE", "unreachable-code",
                    "This statement is unreachable because the previous statement "
                    "always terminates execution.",
                    statement, "medium", 0.99,
                ))
                return  # one finding per block is enough

            check_nested(statement)

            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                terminated = True
            elif (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and _is_exit_call(statement.value)
            ):
                terminated = True
            elif (
                isinstance(statement, ast.While)
                and _is_const_true(statement.test)
                and not _has_loop_exit(statement)
                and not _inside_try_body(statement, parents)
            ):
                terminated = True

    check_block(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, FUNC_TYPES):
            check_block(node.body)

    return findings


def detect_comparison_mistakes(tree):
    findings = []

    def is_singleton(node):
        return node.value is None or node.value is True or node.value is False or node.value is Ellipsis

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue

        operands = [node.left] + list(node.comparators)
        for index, op in enumerate(node.ops):
            left, right = operands[index], operands[index + 1]

            if isinstance(op, (ast.Is, ast.IsNot)):
                for side in (left, right):
                    literal = (
                        isinstance(side, ast.Constant) and not is_singleton(side)
                    ) or isinstance(side, (ast.List, ast.Dict, ast.Set))
                    if literal:
                        findings.append(_make(
                            "IS_WITH_LITERAL", "identity-comparison",
                            "'is' compares object identity, not value; comparing "
                            "with a literal gives unreliable results. Use == instead.",
                            node, "medium", 0.95,
                        ))
                        break

            elif isinstance(op, (ast.Eq, ast.NotEq)):
                if _is_none_const(left) or _is_none_const(right):
                    findings.append(_make(
                        "COMPARE_TO_NONE", "none-comparison",
                        "Compare with None using 'is' / 'is not', not == / !=.",
                        node, "low", 0.90,
                    ))

    return findings


def detect_infinite_loops(tree, parents):
    findings = []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.While)
            and _is_const_true(node.test)
            and not _has_loop_exit(node)
            and not _inside_try_body(node, parents)
        ):
            findings.append(_make(
                "INFINITE_LOOP", "infinite-loop",
                "'while True' loop has no break, return, raise or exit call, so it "
                "never terminates (unless this is an intentional server/event loop).",
                node, "medium", 0.70, evidence="while True: (no exit found)",
            ))

    return findings


def detect_recursion_without_base_case(tree, parents):
    findings = []

    for fn in ast.walk(tree):
        if not isinstance(fn, FUNC_TYPES):
            continue

        own = list(_own_nodes(fn))
        is_method = isinstance(parents.get(fn), ast.ClassDef)

        # the name is rebound locally (import / assignment / self.name = ...),
        # so a call to it is not a call to this function
        rebound = fn.name in _bound_names(fn.body) or fn.name in _param_names(fn) or any(
            isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and n.attr == fn.name
            for n in own
        )
        if rebound:
            continue

        def is_self_call(n):
            if not isinstance(n, ast.Call):
                return False
            f = n.func
            if isinstance(f, ast.Name):
                return f.id == fn.name and not is_method  # a bare name never reaches a method
            return (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in ("self", "cls")
                and f.attr == fn.name
            )

        self_calls = [n for n in own if is_self_call(n)]
        if not self_calls:
            continue

        def guarded(call):
            node = call
            while node in parents and node is not fn:
                node = parents[node]
                if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor,
                                     ast.While, ast.ExceptHandler)):
                    return True
            return False

        has_guarded_call = any(guarded(c) for c in self_calls)
        has_base_return = any(
            isinstance(n, (ast.Return, ast.Raise))
            and not any(is_self_call(m) for m in ast.walk(n))
            for n in own
        )

        if not has_guarded_call and not has_base_return:
            findings.append(_make(
                "RECURSION_NO_BASE_CASE", "recursion-no-base-case",
                f"Function '{fn.name}' calls itself but has no base case, so the "
                f"recursion never stops (RecursionError).",
                fn, "high", 0.90, evidence=f"def {fn.name}(...) calls {fn.name}(...)",
            ))

    return findings


def detect_builtin_shadowing(tree):
    findings = []
    seen = set()

    def flag(name, node):
        key = (name, node.lineno)
        if name in SHADOW_NAMES and key not in seen:
            seen.add(key)
            findings.append(_make(
                "BUILTIN_SHADOWING", "builtin-shadowing",
                f"The name '{name}' shadows a Python built-in; later uses of the "
                f"built-in in this scope will break.",
                node, "low", 0.85, evidence=f"'{name}' reassigned",
            ))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            flag(node.id, node)
        elif isinstance(node, ast.arg):
            flag(node.arg, node)
        elif isinstance(node, FUNC_TYPES + (ast.ClassDef,)):
            flag(node.name, node)

    return findings


def detect_assert_on_tuple(tree):
    findings = []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Tuple)
            and node.test.elts
        ):
            findings.append(_make(
                "ASSERT_ON_TUPLE", "assert-on-tuple",
                "Asserting a non-empty tuple is always true, so this assertion "
                "can never fail. Remove the parentheses around the condition and message.",
                node, "high", 0.99,
            ))

    return findings


def detect_unused_variables(tree):
    findings = []

    for fn in ast.walk(tree):
        if not isinstance(fn, FUNC_TYPES):
            continue

        loads = {
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        declared = set()
        dynamic = False
        for n in ast.walk(fn):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                declared.update(n.names)
            elif (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("locals", "vars", "eval", "exec")
            ):
                dynamic = True
        if dynamic:
            continue

        reported = set()
        for stmt in _own_nodes(fn):
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
            else:
                continue

            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if (
                    name in loads or name in declared or name in reported
                    or name.startswith("_")
                ):
                    continue
                reported.add(name)
                findings.append(_make(
                    "UNUSED_VARIABLE", "unused-variable",
                    f"Variable '{name}' is assigned but never used.",
                    target, "low", 0.70, evidence=f"{name} = ...",
                ))

    return findings


def detect_modification_during_iteration(tree, parents):
    findings = []

    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        if not isinstance(loop.iter, ast.Name):
            continue

        name = loop.iter.id
        reported = False
        rebound = any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
            for stmt in loop.body for n in ast.walk(stmt)
        )
        if name in ("self", "cls") or rebound:
            continue

        for stmt in loop.body:
            for node in ast.walk(stmt):
                if isinstance(node, SCOPE_TYPES):
                    continue

                mutates = (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in MUTATING_METHODS
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == name
                    and not (node.func.attr == "clear" and (node.args or node.keywords))
                ) or (
                    isinstance(node, ast.Delete)
                    and any(
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == name
                        for t in node.targets
                    )
                )

                if mutates and not _following_exit(node, parents) and not reported:
                    reported = True
                    findings.append(_make(
                        "MODIFY_WHILE_ITERATING", "modify-while-iterating",
                        f"'{name}' is modified while it is being iterated over, which "
                        f"skips elements or behaves unpredictably. Iterate over a copy "
                        f"(e.g. {name}[:]) instead.",
                        node, "high", 0.80,
                    ))

    return findings


# ===============================================================
# Layer 2: model-based analysis (internal model of program state)
# ===============================================================
class Val:
    """Abstract value of an expression: probable type, known constant,
    and whether it is None ("yes" / "maybe" / "no")."""

    __slots__ = ("kind", "const", "none")

    def __init__(self, kind="unknown", const=UNKNOWN, none="no"):
        self.kind = kind
        self.const = const
        self.none = none


def _info(val=None, defined="yes"):
    val = val or Val()
    return {"def": defined, "kind": val.kind, "const": val.const, "none": val.none}


def _val_of(info):
    return Val(info["kind"], info["const"], info["none"])


def _copy(state):
    return {name: dict(info) for name, info in state.items()}


def _same_const(a, b):
    return a is not UNKNOWN and b is not UNKNOWN and type(a) is type(b) and a == b


def _merge_vals(vals):
    kinds = {v.kind for v in vals}
    kind = next(iter(kinds)) if len(kinds) == 1 else "unknown"
    first = vals[0].const
    const = first if all(_same_const(v.const, first) for v in vals) else UNKNOWN
    nones = {v.none for v in vals}
    none = next(iter(nones)) if len(nones) == 1 else "maybe"
    return Val(kind, const, none)


def _merge_states(states):
    """Join the states of several control-flow paths. None = unreachable."""
    states = [s for s in states if s is not None]
    if not states:
        return None
    if len(states) == 1:
        return states[0]

    merged = {}
    for name in set().union(*states):
        infos = [s.get(name) for s in states]
        present = [i for i in infos if i is not None]
        everywhere = len(present) == len(infos) and all(i["def"] == "yes" for i in present)
        value = _merge_vals([_val_of(i) for i in present])
        merged[name] = _info(value, "yes" if everywhere else "maybe")
    return merged


def _bound_names(statements):
    """Names bound (assigned, imported, defined) directly in a scope."""
    names = set()
    stack = list(statements)

    while stack:
        node = stack.pop()

        if isinstance(node, FUNC_TYPES + (ast.ClassDef,)):
            names.add(node.name)
            continue
        if isinstance(node, (ast.Lambda,) + COMP_TYPES):
            continue

        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif type(node).__name__ in ("MatchAs", "MatchStar") and getattr(node, "name", None):
            names.add(node.name)
        elif type(node).__name__ == "MatchMapping" and getattr(node, "rest", None):
            names.add(node.rest)

        stack.extend(ast.iter_child_nodes(node))

    return names


def _declared_outer(fn):
    names = set()
    for node in _own_nodes(fn):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _param_names(fn):
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _compute(op, a, b):
    number = (int, float)
    try:
        if isinstance(op, ast.Add):
            if isinstance(a, str) and isinstance(b, str):
                return a + b if len(a) + len(b) <= 200 else UNKNOWN
            if isinstance(a, number) and isinstance(b, number):
                return a + b
        elif isinstance(a, number) and isinstance(b, number):
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.FloorDiv):
                return a // b
            if isinstance(op, ast.Mod):
                return a % b
            if isinstance(op, ast.Pow) and abs(b) <= 64 and abs(a) <= 10 ** 6:
                result = a ** b
                return result if isinstance(result, number) else UNKNOWN
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return UNKNOWN
    return UNKNOWN


def _type_error(op, lk, rk):
    """True if lk <op> rk always raises TypeError for these probable types."""
    if isinstance(op, ast.Mod) and lk == "str":
        return False  # string formatting
    if not isinstance(op, tuple(OP_SYMBOL)):
        return False

    if "none" in (lk, rk):
        return True
    if lk == "unknown" or rk == "unknown":
        return False

    both_num = lk in NUM and rk in NUM
    if isinstance(op, ast.Add):
        return not (both_num or (lk == rk and lk in SEQ))
    if isinstance(op, ast.Sub):
        return not (both_num or (lk == "set" and rk == "set"))
    if isinstance(op, ast.Mult):
        seq_times_int = (lk in SEQ and rk in ("int", "bool")) or (rk in SEQ and lk in ("int", "bool"))
        return not (both_num or seq_times_int)
    if isinstance(op, ast.Mod):
        return not (both_num or lk == "bytes")
    return not both_num  # Div, FloorDiv, Pow


class _FlowAnalyzer:
    """Walks one scope statement by statement, keeping a model of every
    variable (definedness, probable type, constant value, None-ness)."""

    def __init__(self, findings, local_names, outer_names, has_star, is_module=False):
        self.is_module = is_module
        self.findings = findings
        self.local_names = local_names
        self.outer_names = outer_names
        self.has_star = has_star
        self.seen = set()
        self.loops = []
        self.try_stack = []
        self.lambda_depth = 0
        self.value_returns = False
        self.bare_returns = False

    # ---------------- reporting ----------------
    def report(self, rule_id, category, message, node, severity, confidence,
               evidence=None, key=None):
        dedupe = (rule_id, getattr(node, "lineno", 1), key)
        if dedupe in self.seen:
            return
        self.seen.add(dedupe)
        self.findings.append(_make(
            rule_id, category, message, node, severity, confidence,
            evidence=evidence, source="flow_analysis",
        ))

    def _guarded(self, names):
        return any("*" in caught or (caught & names) for caught in self.try_stack)

    # ---------------- pure constant evaluation ----------------
    def _pconst(self, node, state):
        if isinstance(node, ast.Constant):
            v = node.value
            return v if isinstance(v, (bool, int, float, str)) else UNKNOWN
        if isinstance(node, ast.Name):
            info = state.get(node.id)
            return info["const"] if info else UNKNOWN
        if isinstance(node, ast.UnaryOp):
            c = self._pconst(node.operand, state)
            if c is UNKNOWN:
                return UNKNOWN
            if isinstance(node.op, ast.Not):
                return not c
            if isinstance(c, (int, float)):
                if isinstance(node.op, ast.USub):
                    return -c
                if isinstance(node.op, ast.UAdd):
                    return +c
            return UNKNOWN
        if isinstance(node, ast.BinOp):
            a = self._pconst(node.left, state)
            b = self._pconst(node.right, state)
            if a is UNKNOWN or b is UNKNOWN:
                return UNKNOWN
            return _compute(node.op, a, b)
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            consts = [self._pconst(o, state) for o in operands]
            if any(c is UNKNOWN for c in consts):
                return UNKNOWN
            try:
                result = True
                for i, op in enumerate(node.ops):
                    fn = CMP_FUNCS.get(type(op))
                    if fn is None:
                        return UNKNOWN
                    result = result and fn(consts[i], consts[i + 1])
                return result
            except TypeError:
                return UNKNOWN
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            last = UNKNOWN
            for operand in node.values:
                c = self._pconst(operand, state)
                if c is UNKNOWN:
                    return UNKNOWN
                last = c
                if is_and and not c:
                    return c
                if not is_and and c:
                    return c
            return last
        return UNKNOWN

    def _truth(self, test, state):
        """True / False if the condition's value is known, else None."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, left, right = test.ops[0], test.left, test.comparators[0]
            if isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                for a, b in ((left, right), (right, left)):
                    if _is_none_const(b) and isinstance(a, ast.Name):
                        info = state.get(a.id)
                        if info and info["none"] == "yes":
                            return isinstance(op, (ast.Is, ast.Eq))
                        if info and info["none"] == "no" and info["kind"] not in ("unknown", "none"):
                            return isinstance(op, (ast.IsNot, ast.NotEq))
        elif isinstance(test, ast.Name):
            info = state.get(test.id)
            if info and info["none"] == "yes":
                return False
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._truth(test.operand, state)
            return None if inner is None else (not inner)

        c = self._pconst(test, state)
        return None if c is UNKNOWN else bool(c)

    # ---------------- narrowing at branches ----------------
    def _narrow(self, test, state, truth):
        """State assuming `test` evaluated to `truth`; None if that path is dead."""
        known = self._truth(test, state)
        if known is not None and known != truth:
            return None
        new = _copy(state)
        self._apply_narrowing(test, new, truth)
        return new

    def _apply_narrowing(self, test, state, truth):
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            self._apply_narrowing(test.operand, state, not truth)

        elif isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And) and truth:
                for v in test.values:
                    self._apply_narrowing(v, state, True)
            elif isinstance(test.op, ast.Or) and not truth:
                for v in test.values:
                    self._apply_narrowing(v, state, False)

        elif isinstance(test, ast.Name):
            info = state.get(test.id)
            if info and truth:
                info["none"] = "no"
                if info["kind"] == "none":
                    info["kind"] = "unknown"

        elif isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, left, right = test.ops[0], test.left, test.comparators[0]

            for a, b in ((left, right), (right, left)):
                if isinstance(a, ast.Name) and _is_none_const(b) and isinstance(
                        op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                    info = state.get(a.id)
                    if info is None:
                        return
                    positive = isinstance(op, (ast.Is, ast.Eq))
                    is_none = positive == truth
                    if is_none:
                        info["none"], info["kind"] = "yes", "none"
                    else:
                        info["none"] = "no"
                        if info["kind"] == "none":
                            info["kind"] = "unknown"
                    return

            equal_branch = (isinstance(op, ast.Eq) and truth) or (isinstance(op, ast.NotEq) and not truth)
            if equal_branch:
                for a, b in ((left, right), (right, left)):
                    if isinstance(a, ast.Name) and a.id in state:
                        c = self._pconst(b, state)
                        if isinstance(c, (int, float)) and not isinstance(c, bool):
                            state[a.id]["const"] = c
                            state[a.id]["none"] = "no"
                            return

        elif (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "isinstance"
            and test.args
            and isinstance(test.args[0], ast.Name)
            and truth
        ):
            info = state.get(test.args[0].id)
            if info:
                info["none"] = "no"

    # ---------------- expressions ----------------
    def eval(self, node, state, bound=frozenset()):
        method = getattr(self, "_e_" + type(node).__name__, None)
        if method is not None:
            return method(node, state, bound)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.eval(child, state, bound)
        return Val()

    def _load(self, node, state):
        name = node.id
        info = state.get(name)

        if info is not None:
            if info["def"] == "maybe" and not self.lambda_depth and not self.is_module:
                self.report(
                    "POSSIBLY_UNDEFINED_VARIABLE", "possibly-undefined-variable",
                    f"Variable '{name}' may be undefined here because it is only "
                    f"assigned on some paths.",
                    node, "low", 0.50, evidence=name, key=name,
                )
            return _val_of(info)

        if self.lambda_depth:
            return Val()

        if self._guarded({"NameError", "UnboundLocalError", "Exception", "BaseException"}):
            return Val()  # intentional probe: try: name / except NameError

        if name in self.local_names and not (self.is_module and name in BUILTIN_NAMES):
            self.report(
                "USED_BEFORE_ASSIGNMENT", "undefined-variable",
                f"Variable '{name}' is used before it is assigned (or after it was "
                f"deleted), which raises an error.",
                node, "high", 0.85, evidence=name, key=name,
            )
        elif name in self.outer_names or name in BUILTIN_NAMES or self.has_star:
            pass
        else:
            self.report(
                "UNDEFINED_NAME", "undefined-variable",
                f"Name '{name}' is not defined anywhere, so this raises NameError.",
                node, "high", 0.90, evidence=name, key=name,
            )
        return Val()

    def _e_Name(self, node, state, bound):
        if node.id in bound or not isinstance(node.ctx, ast.Load):
            return Val()
        return self._load(node, state)

    def _e_Constant(self, node, state, bound):
        v = node.value
        if v is None:
            return Val("none", UNKNOWN, "yes")
        if isinstance(v, bool):
            return Val("bool", v)
        if isinstance(v, int):
            return Val("int", v)
        if isinstance(v, float):
            return Val("float", v)
        if isinstance(v, str):
            return Val("str", v)
        if isinstance(v, bytes):
            return Val("bytes")
        return Val()

    def _e_JoinedStr(self, node, state, bound):
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                self.eval(part.value, state, bound)
        return Val("str")

    def _e_BinOp(self, node, state, bound):
        left = self.eval(node.left, state, bound)
        right = self.eval(node.right, state, bound)
        return self._binop(node, node.op, left, right, node.right)

    def _e_UnaryOp(self, node, state, bound):
        v = self.eval(node.operand, state, bound)
        if isinstance(node.op, ast.Not):
            const = UNKNOWN if v.const is UNKNOWN else (not v.const)
            return Val("bool", const)
        if isinstance(node.op, (ast.USub, ast.UAdd)) and v.kind in ("int", "float"):
            if v.const is UNKNOWN:
                return Val(v.kind)
            return Val(v.kind, -v.const if isinstance(node.op, ast.USub) else +v.const)
        return Val("int") if v.kind == "int" else Val()

    def _e_BoolOp(self, node, state, bound):
        is_and = isinstance(node.op, ast.And)
        current = state
        vals = []

        for index, operand in enumerate(node.values):
            if index > 0:
                narrowed = self._narrow(node.values[index - 1], current, is_and)
                if narrowed is None:
                    break
                current = narrowed
            vals.append(self.eval(operand, current, bound))
            if current is not state:
                # walrus targets assigned in a narrowed copy must stay visible
                for sub_node in ast.walk(operand):
                    if (
                        isinstance(sub_node, ast.NamedExpr)
                        and isinstance(sub_node.target, ast.Name)
                        and sub_node.target.id in current
                    ):
                        state[sub_node.target.id] = dict(current[sub_node.target.id])

        if not vals:
            return Val()

        kinds = {v.kind for v in vals}
        kind = next(iter(kinds)) if len(kinds) == 1 else "unknown"
        if is_and:
            none = "maybe" if any(v.none != "no" for v in vals) else "no"
        else:
            none = vals[-1].none
        return Val(kind, UNKNOWN, none)

    def _e_Compare(self, node, state, bound):
        self.eval(node.left, state, bound)
        for comparator in node.comparators:
            self.eval(comparator, state, bound)
        return Val("bool", self._pconst(node, state))

    def _e_IfExp(self, node, state, bound):
        self.eval(node.test, state, bound)
        vals = []
        true_state = self._narrow(node.test, state, True)
        false_state = self._narrow(node.test, state, False)
        if true_state is not None:
            vals.append(self.eval(node.body, true_state, bound))
        if false_state is not None:
            vals.append(self.eval(node.orelse, false_state, bound))
        return _merge_vals(vals) if vals else Val()

    def _check_deref(self, val, expr_node, use_node):
        # "maybe" (None on some merged path) is deliberately not reported: it is
        # dominated by false positives from conditions the analysis cannot correlate.
        if val.none in ("no", "maybe") or self._guarded(ATTR_CATCH):
            return
        text = _src(expr_node)
        if val.none == "yes":
            self.report(
                "NONE_DEREFERENCE", "none-dereference",
                f"'{text}' is None at this point, so using it like an object "
                f"raises an error.",
                use_node, "high", 0.90, evidence=text, key=text,
            )
        else:  # "optional": result of a call that can return None
            self.report(
                "NONE_DEREFERENCE", "none-dereference",
                f"'{text}' comes from a call that can return None (e.g. a failed "
                f"match or lookup); check it before using it.",
                use_node, "medium", 0.70, evidence=text, key=text,
            )

    def _e_Attribute(self, node, state, bound):
        recv = self.eval(node.value, state, bound)
        self._check_deref(recv, node.value, node)
        return Val()

    def _e_Subscript(self, node, state, bound):
        recv = self.eval(node.value, state, bound)
        self._check_deref(recv, node.value, node)
        self.eval(node.slice, state, bound)
        return Val("str") if recv.kind == "str" else Val()

    def _e_List(self, node, state, bound):
        for e in node.elts:
            self.eval(e, state, bound)
        return Val("list")

    def _e_Tuple(self, node, state, bound):
        for e in node.elts:
            self.eval(e, state, bound)
        return Val("tuple")

    def _e_Set(self, node, state, bound):
        for e in node.elts:
            self.eval(e, state, bound)
        return Val("set")

    def _e_Dict(self, node, state, bound):
        for k in node.keys:
            if k is not None:
                self.eval(k, state, bound)
        for v in node.values:
            self.eval(v, state, bound)
        return Val("dict")

    def _comprehension(self, node, state, bound, kind):
        current_bound = set(bound)
        current = state

        for gen in node.generators:
            self.eval(gen.iter, current, frozenset(current_bound))
            current_bound |= _target_names(gen.target)
            for cond in gen.ifs:
                self.eval(cond, current, frozenset(current_bound))
                narrowed = self._narrow(cond, current, True)
                if narrowed is None:
                    return Val(kind)
                current = narrowed

        inner = frozenset(current_bound)
        if isinstance(node, ast.DictComp):
            self.eval(node.key, current, inner)
            self.eval(node.value, current, inner)
        else:
            self.eval(node.elt, current, inner)
        return Val(kind)

    def _e_ListComp(self, node, state, bound):
        return self._comprehension(node, state, bound, "list")

    def _e_SetComp(self, node, state, bound):
        return self._comprehension(node, state, bound, "set")

    def _e_DictComp(self, node, state, bound):
        return self._comprehension(node, state, bound, "dict")

    def _e_GeneratorExp(self, node, state, bound):
        return self._comprehension(node, state, bound, "unknown")

    def _e_Lambda(self, node, state, bound):
        args = node.args
        for d in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            self.eval(d, state, bound)
        names = _param_names(node)
        self.lambda_depth += 1
        self.eval(node.body, state, frozenset(set(bound) | names))
        self.lambda_depth -= 1
        return Val()

    def _e_NamedExpr(self, node, state, bound):
        value = self.eval(node.value, state, bound)
        if isinstance(node.target, ast.Name):
            state[node.target.id] = _info(value)
        return value

    def _is_builtin(self, name, state):
        return name not in state and name not in self.local_names and name not in self.outer_names

    def _e_Call(self, node, state, bound):
        func = node.func
        recv = None

        if isinstance(func, ast.Attribute):
            recv = self.eval(func.value, state, bound)
            self._check_deref(recv, func.value, node)
        else:
            callee = self.eval(func, state, bound)
            if isinstance(func, ast.Name):
                self._check_deref(callee, func, node)

        for arg in node.args:
            self.eval(arg.value if isinstance(arg, ast.Starred) else arg, state, bound)
        for kw in node.keywords:
            self.eval(kw.value, state, bound)

        return self._call_result(node, recv, state)

    def _call_result(self, node, recv, state):
        func = node.func

        if isinstance(func, ast.Name):
            if func.id in BUILTIN_RESULT and self._is_builtin(func.id, state):
                return Val(BUILTIN_RESULT[func.id])
            return Val()

        if not isinstance(func, ast.Attribute):
            return Val()

        attr = func.attr
        one_arg = len(node.args) == 1 and not node.keywords

        if recv is not None:
            if recv.kind == "str":
                if attr in STR_RETURNS_STR:
                    return Val("str")
                if attr in ("split", "splitlines"):
                    return Val("list")
                if attr in ("find", "count", "index", "rfind"):
                    return Val("int")
            elif recv.kind == "list":
                if attr in LIST_NONE_METHODS:
                    return Val("none", UNKNOWN, "yes")
                if attr == "copy":
                    return Val("list")
            elif recv.kind == "dict":
                if attr == "get" and one_arg:
                    return Val("unknown", UNKNOWN, "optional")
                if attr == "copy":
                    return Val("dict")

        base = func.value
        if isinstance(base, ast.Name) and base.id == "re" and attr in ("match", "search", "fullmatch"):
            return Val("unknown", UNKNOWN, "optional")
        if isinstance(base, ast.Name) and base.id == "os" and attr == "getenv" and one_arg:
            return Val("unknown", UNKNOWN, "optional")
        if (
            attr == "get" and one_arg
            and isinstance(base, ast.Attribute) and base.attr == "environ"
        ):
            return Val("unknown", UNKNOWN, "optional")

        return Val()

    # ---------------- arithmetic checks ----------------
    def _known_zero(self, v):
        return (
            isinstance(v.const, (int, float))
            and not isinstance(v.const, bool)
            and v.const == 0
        )

    def _binop(self, node, op, left, right, right_node):
        if (
            isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod))
            and self._known_zero(right)
            and not (isinstance(op, ast.Mod) and left.kind == "str")
            and not self._guarded(ZERO_CATCH)
        ):
            literal = isinstance(right_node, ast.Constant)
            what = "a literal 0" if literal else f"'{_src(right_node)}', which is always 0 here"
            self.report(
                "DIVISION_BY_ZERO", "division-by-zero",
                f"Division by zero: the divisor is {what}.",
                node, "high", 0.95 if literal else 0.90,
            )

        if _type_error(op, left.kind, right.kind) and not self._guarded(TYPE_CATCH):
            self.report(
                "TYPE_MISMATCH", "type-mismatch",
                f"Unsupported operand types for '{OP_SYMBOL[type(op)]}': "
                f"{left.kind} and {right.kind}; this raises TypeError.",
                node, "high", 0.90,
            )

        lk, rk = left.kind, right.kind
        kind = "unknown"
        if lk in NUM and rk in NUM:
            if isinstance(op, ast.Div):
                kind = "float"
            elif isinstance(op, ast.Pow):
                kind = "float" if "float" in (lk, rk) else "unknown"
            elif "float" in (lk, rk):
                kind = "float"
            else:
                kind = "int"
        elif isinstance(op, ast.Add) and lk == rk and lk in SEQ:
            kind = lk
        elif isinstance(op, ast.Mult):
            seq = [k for k in (lk, rk) if k in SEQ]
            if seq and ({lk, rk} & {"int", "bool"}):
                kind = seq[0]
        elif isinstance(op, ast.Mod) and lk == "str":
            kind = "str"

        const = UNKNOWN
        if left.const is not UNKNOWN and right.const is not UNKNOWN:
            const = _compute(op, left.const, right.const)
        return Val(kind, const)

    # ---------------- assignment ----------------
    def assign(self, target, val, state):
        if isinstance(target, ast.Name):
            state[target.id] = _info(val)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.assign(element, Val(), state)
        elif isinstance(target, ast.Starred):
            self.assign(target.value, Val("list"), state)
        elif isinstance(target, ast.Attribute):
            recv = self.eval(target.value, state)
            self._check_deref(recv, target.value, target)
        elif isinstance(target, ast.Subscript):
            recv = self.eval(target.value, state)
            self._check_deref(recv, target.value, target)
            self.eval(target.slice, state)

    def _check_none_returning(self, value, state):
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in LIST_NONE_METHODS
            and isinstance(value.func.value, ast.Name)
        ):
            info = state.get(value.func.value.id)
            if info and info["kind"] == "list":
                self.report(
                    "LIST_METHOD_RETURNS_NONE", "none-return-value",
                    f"'{value.func.attr}' changes the list in place and returns None, "
                    f"so using its result is a bug.",
                    value, "high", 0.90,
                )

    def _check_ignored_result(self, call, state):
        func = call.func

        if isinstance(func, ast.Attribute) and func.attr in STR_METHODS_IGNORED:
            base = func.value
            kind = "unknown"
            if isinstance(base, ast.Name) and base.id in state:
                kind = state[base.id]["kind"]
            elif isinstance(base, (ast.Constant, ast.JoinedStr)):
                kind = "str" if isinstance(getattr(base, "value", ""), str) else "unknown"
            if kind == "str":
                self.report(
                    "IGNORED_RESULT", "ignored-result",
                    f"Strings are immutable: '{func.attr}' returns a new string, but "
                    f"the result is discarded so nothing changes.",
                    call, "medium", 0.90,
                )
        elif (
            isinstance(func, ast.Name)
            and func.id in DISCARDED_BUILTINS
            and self._is_builtin(func.id, state)
            and not self.try_stack  # e.g. try: round(x) / except TypeError (validation probe)
        ):
            self.report(
                "IGNORED_RESULT", "ignored-result",
                f"The result of '{func.id}(...)' is discarded, so this statement has no effect.",
                call, "medium", 0.85,
            )

    # ---------------- statements ----------------
    def exec_block(self, statements, state):
        for statement in statements:
            if state is None:
                return None
            handler = getattr(self, "_s_" + type(statement).__name__, None)
            if handler is not None:
                state = handler(statement, state)
        return state

    def _s_Expr(self, node, state):
        self.eval(node.value, state)
        if isinstance(node.value, ast.Call):
            if _is_exit_call(node.value):
                return None
            self._check_ignored_result(node.value, state)
        return state

    def _s_Assign(self, node, state):
        value = node.value
        self._check_none_returning(value, state)

        first = node.targets[0]
        if (
            len(node.targets) == 1
            and isinstance(first, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(first.elts) == len(value.elts)
            and not any(isinstance(e, ast.Starred) for e in first.elts + value.elts)
        ):
            vals = [self.eval(e, state) for e in value.elts]
            for target, val in zip(first.elts, vals):
                self.assign(target, val, state)
            return state

        val = self.eval(value, state)
        for target in node.targets:
            self.assign(target, val, state)
        return state

    def _s_AugAssign(self, node, state):
        target = node.target
        current = self._load(target, state) if isinstance(target, ast.Name) else self.eval(target, state)
        value = self.eval(node.value, state)

        if isinstance(node.op, ast.Add) and current.kind == "list":
            result = Val("list")  # list += any iterable is valid
        else:
            result = self._binop(node, node.op, current, value, node.value)

        if isinstance(target, ast.Name):
            state[target.id] = _info(result)
        return state

    def _s_AnnAssign(self, node, state):
        if node.value is not None:
            self.assign(node.target, self.eval(node.value, state), state)
        return state

    def _s_Delete(self, node, state):
        for target in node.targets:
            if isinstance(target, ast.Name):
                state.pop(target.id, None)
            else:
                self.eval(target, state)
        return state

    def _s_Return(self, node, state):
        if node.value is None:
            self.bare_returns = True
        else:
            self.eval(node.value, state)
            self._check_none_returning(node.value, state)
            if not _is_none_const(node.value):
                self.value_returns = True
        return None

    def _s_Raise(self, node, state):
        if node.exc is not None:
            self.eval(node.exc, state)
        if node.cause is not None:
            self.eval(node.cause, state)
        return None

    def _s_Assert(self, node, state):
        self.eval(node.test, state)
        if node.msg is not None:
            failing = self._narrow(node.test, state, False)
            if failing is not None:
                self.eval(node.msg, failing)
        return self._narrow(node.test, state, True)

    def _s_Break(self, node, state):
        if self.loops:
            self.loops[-1]["breaks"].append(_copy(state))
        return None

    def _s_Continue(self, node, state):
        if self.loops:
            self.loops[-1]["continues"].append(_copy(state))
        return None

    def _s_Import(self, node, state):
        for alias in node.names:
            state[alias.asname or alias.name.split(".")[0]] = _info()
        return state

    def _s_ImportFrom(self, node, state):
        for alias in node.names:
            if alias.name != "*":
                state[alias.asname or alias.name] = _info()
        return state

    def _s_FunctionDef(self, node, state):
        for decorator in node.decorator_list:
            self.eval(decorator, state)
        args = node.args
        for default in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            self.eval(default, state)
        state[node.name] = _info()
        return state

    _s_AsyncFunctionDef = _s_FunctionDef

    def _s_ClassDef(self, node, state):
        for expr in list(node.bases) + [k.value for k in node.keywords] + list(node.decorator_list):
            self.eval(expr, state)
        state[node.name] = _info()
        return state

    def _s_If(self, node, state):
        self.eval(node.test, state)
        true_state = self._narrow(node.test, state, True)
        false_state = self._narrow(node.test, state, False)

        after_true = self.exec_block(node.body, true_state) if true_state is not None else None
        if false_state is None:
            after_false = None
        elif node.orelse:
            after_false = self.exec_block(node.orelse, false_state)
        else:
            after_false = false_state
        return _merge_states([after_true, after_false])

    def _widen(self, state, names):
        """Forget what we knew about variables that the loop body reassigns."""
        new = _copy(state)
        for name in names:
            info = new.get(name)
            if info is None:
                new[name] = {"def": "maybe", "kind": "unknown", "const": UNKNOWN, "none": "no"}
            else:
                info["kind"], info["const"] = "unknown", UNKNOWN
                if info["none"] == "yes":
                    info["none"] = "maybe"
        return new

    def _s_While(self, node, state):
        entry = self._widen(state, _bound_names(node.body))
        self.eval(node.test, entry)
        body_state = self._narrow(node.test, entry, True)

        self.loops.append({"breaks": [], "continues": []})
        if body_state is not None:
            self.exec_block(node.body, body_state)
        context = self.loops.pop()

        always_true = self._truth(node.test, entry) is True
        normal = None if always_true else self._narrow(node.test, entry, False)
        if normal is not None and node.orelse:
            normal = self.exec_block(node.orelse, normal)
        return _merge_states([normal] + context["breaks"])

    def _element_val(self, iterable, state):
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Name)
            and iterable.func.id == "range"
            and self._is_builtin("range", state)
        ):
            return Val("int")
        if isinstance(iterable, ast.Name) and iterable.id in state and state[iterable.id]["kind"] == "str":
            return Val("str")
        if isinstance(iterable, ast.Constant) and isinstance(iterable.value, str):
            return Val("str")
        return Val()

    def _s_For(self, node, state):
        self.eval(node.iter, state)
        element = self._element_val(node.iter, state)

        names = _bound_names(node.body) | _target_names(node.target)
        entry = self._widen(state, names)
        body_state = _copy(entry)
        self.assign(node.target, element, body_state)

        self.loops.append({"breaks": [], "continues": []})
        end = self.exec_block(node.body, body_state)
        context = self.loops.pop()

        normal = _merge_states([entry, end] + context["continues"])
        if normal is not None and node.orelse:
            normal = self.exec_block(node.orelse, normal)
        return _merge_states([normal] + context["breaks"])

    _s_AsyncFor = _s_For

    @staticmethod
    def _caught_names(node):
        caught = set()
        for handler in node.handlers:
            if handler.type is None:
                caught.add("*")
                continue
            types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            for t in types:
                if isinstance(t, ast.Name):
                    caught.add(t.id)
                elif isinstance(t, ast.Attribute):
                    caught.add(t.attr)
        return caught

    def _s_Try(self, node, state):
        before = _copy(state)

        self.try_stack.append(self._caught_names(node))
        body_end = self.exec_block(node.body, state)
        self.try_stack.pop()

        handler_start = _copy(before)
        for name in _bound_names(node.body):
            info = handler_start.get(name)
            if info is None:
                handler_start[name] = {"def": "maybe", "kind": "unknown", "const": UNKNOWN, "none": "no"}
            else:
                info["kind"], info["const"] = "unknown", UNKNOWN

        if node.orelse and body_end is not None:
            body_end = self.exec_block(node.orelse, body_end)

        outcomes = [body_end]
        for handler in node.handlers:
            hs = _copy(handler_start)
            if handler.type is not None:
                self.eval(handler.type, hs)
            if handler.name:
                hs[handler.name] = _info()
            outcomes.append(self.exec_block(handler.body, hs))

        merged = _merge_states(outcomes)

        if node.finalbody:
            start = merged if merged is not None else _copy(handler_start)
            final_end = self.exec_block(node.finalbody, start)
            return final_end if merged is not None else None
        return merged

    _s_TryStar = _s_Try

    def _s_With(self, node, state):
        for item in node.items:
            self.eval(item.context_expr, state)
            if item.optional_vars is not None:
                self.assign(item.optional_vars, Val(), state)
        return self.exec_block(node.body, state)

    _s_AsyncWith = _s_With

    def _s_Match(self, node, state):
        self.eval(node.subject, state)
        outcomes = []
        irrefutable = False

        for case in node.cases:
            case_state = _copy(state)
            for sub in ast.walk(case.pattern):
                name = getattr(sub, "name", None) or getattr(sub, "rest", None)
                if isinstance(name, str):
                    case_state[name] = _info()
            if case.guard is not None:
                self.eval(case.guard, case_state)
            outcomes.append(self.exec_block(case.body, case_state))
            if (
                case.guard is None
                and type(case.pattern).__name__ == "MatchAs"
                and case.pattern.pattern is None
            ):
                irrefutable = True

        if not irrefutable:
            outcomes.append(state)
        return _merge_states(outcomes)

    # ---------------- entry points ----------------
    def run_module(self, tree):
        state = {
            name: _info()
            for name in ("__name__", "__doc__", "__file__", "__package__",
                         "__spec__", "__loader__", "__builtins__")
        }
        self.exec_block(tree.body, state)

    def run_function(self, fn, declared=()):
        state = {name: _info() for name in declared}
        args = fn.args
        positional = args.posonlyargs + args.args
        for arg in positional + args.kwonlyargs:
            state[arg.arg] = _info(Val())
        if args.vararg:
            state[args.vararg.arg] = _info(Val("tuple"))
        if args.kwarg:
            state[args.kwarg.arg] = _info(Val("dict"))

        end = self.exec_block(fn.body, state)

        is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _own_nodes(fn))
        if self.value_returns and not is_generator and (end is not None or self.bare_returns):
            reason = "a bare 'return'" if self.bare_returns and end is None else "falling off the end"
            self.report(
                "INCONSISTENT_RETURN", "inconsistent-return",
                f"Function '{fn.name}' returns a value on some paths but returns "
                f"None on others ({reason}).",
                fn, "low", 0.70, evidence=f"def {fn.name}(...)", key=fn.name,
            )


def _enclosing_names(fn, parents):
    names = set()
    node = parents.get(fn)
    while node is not None:
        if isinstance(node, FUNC_TYPES):
            names |= _bound_names(node.body) | _param_names(node)
        node = parents.get(node)
    return names


def detect_flow_bugs(tree, parents):
    findings = []

    module_names = set(_bound_names(tree.body))
    has_star = False  # names may be created in ways we cannot see
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            module_names.update(node.names)
        elif isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            has_star = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("globals", "exec", "eval")
        ):
            has_star = True

    try:
        _FlowAnalyzer(findings, module_names, set(), has_star, is_module=True).run_module(tree)
    except Exception:
        pass  # a failure in the model must never crash the whole agent

    for fn in ast.walk(tree):
        if not isinstance(fn, FUNC_TYPES):
            continue
        try:
            locals_ = (_bound_names(fn.body) - _declared_outer(fn)) | _param_names(fn)
            outer = module_names | _enclosing_names(fn, parents)
            _FlowAnalyzer(findings, locals_, outer, has_star).run_function(
                fn, declared=_declared_outer(fn)
            )
        except Exception:
            continue

    return findings


# ===============================================================
# Agent entry point
# ===============================================================
def detect_bugs(representation):
    tree = representation["tree"]
    parents = _parent_map(tree)

    findings = []

    # Layer 1: simple reflex rules
    findings.extend(detect_mutable_default_arguments(tree))
    findings.extend(detect_bare_except(tree))
    findings.extend(detect_swallowed_exceptions(tree))
    findings.extend(detect_unreachable_code(tree, parents))
    findings.extend(detect_comparison_mistakes(tree))
    findings.extend(detect_infinite_loops(tree, parents))
    findings.extend(detect_recursion_without_base_case(tree, parents))
    findings.extend(detect_builtin_shadowing(tree))
    findings.extend(detect_assert_on_tuple(tree))
    findings.extend(detect_unused_variables(tree))
    findings.extend(detect_modification_during_iteration(tree, parents))

    # Layer 2: model-based analysis
    findings.extend(detect_flow_bugs(tree, parents))

    findings.sort(key=lambda f: (f["line"], f["column"]))
    return findings


# ===============================================================
# Self-check: positive and negative examples per rule
# ===============================================================
EXAMPLES = {
    # ---- layer 1 ----
    "mutable default (+)": ("def f(x=[]):\n    return x\n", {"MUTABLE_DEFAULT_ARGUMENT"}),
    "mutable default (-)": ("def f(x=None):\n    return x\n", set()),
    "bare except (+)": ("try:\n    pass\nexcept:\n    print('x')\n", {"BARE_EXCEPT"}),
    "swallowed (+)": ("try:\n    int('a')\nexcept ValueError:\n    pass\n", {"SWALLOWED_EXCEPTION"}),
    "handled except (-)": ("try:\n    int('a')\nexcept ValueError:\n    print('bad')\n", set()),
    "unreachable (+)": ("def f():\n    return 1\n    print('x')\n", {"UNREACHABLE_CODE"}),
    "is literal (+)": ("def f(x):\n    return x is 5\n", {"IS_WITH_LITERAL"}),
    "is None (-)": ("def f(x):\n    return x is None\n", set()),
    "== None (+)": ("def f(x):\n    return x == None\n", {"COMPARE_TO_NONE"}),
    "infinite loop (+)": ("def f():\n    while True:\n        pass\n", {"INFINITE_LOOP"}),
    "loop with break (-)": ("def f():\n    while True:\n        if input():\n            break\n", set()),
    "no base case (+)": ("def f(n):\n    return f(n - 1)\n", {"RECURSION_NO_BASE_CASE"}),
    "with base case (-)": ("def f(n):\n    if n == 0:\n        return 1\n    return n * f(n - 1)\n", set()),
    "shadow builtin (+)": ("def f(items):\n    list = [1]\n    return list\n", {"BUILTIN_SHADOWING"}),
    "assert tuple (+)": ("def f(x):\n    assert (x > 0, 'positive')\n", {"ASSERT_ON_TUPLE"}),
    "unused var (+)": ("def f():\n    x = 5\n    return 1\n", {"UNUSED_VARIABLE"}),
    "used var (-)": ("def f():\n    x = 5\n    return x\n", set()),
    "modify while iterating (+)": ("def f(items):\n    for x in items:\n        if x:\n            items.remove(x)\n", {"MODIFY_WHILE_ITERATING"}),
    "remove then break (-)": ("def f(items):\n    for x in items:\n        if x:\n            items.remove(x)\n            break\n", set()),
    # ---- layer 2 ----
    "undefined name (+)": ("def f():\n    return missing + 1\n", {"UNDEFINED_NAME"}),
    "used before assignment (+)": ("def f():\n    print(x)\n    x = 1\n", {"USED_BEFORE_ASSIGNMENT"}),
    "possibly undefined (+)": ("def f(c):\n    if c:\n        x = 1\n    return x\n", {"POSSIBLY_UNDEFINED_VARIABLE"}),
    "defined on both paths (-)": ("def f(c):\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n", set()),
    "global name (-)": ("LIMIT = 3\n\ndef f():\n    return LIMIT\n", set()),
    "div by zero var (+)": ("def f(a):\n    y = 0\n    return a / y\n", {"DIVISION_BY_ZERO"}),
    "div guarded (-)": ("def f(a):\n    y = 0\n    if y != 0:\n        return a / y\n    return 0\n", set()),
    "div in try (-)": ("def f(a):\n    try:\n        return a / 0\n    except ZeroDivisionError:\n        return 0\n", set()),
    "div by zero branch (+)": ("def f(a, y):\n    if y == 0:\n        return a / y\n    return a / y\n", {"DIVISION_BY_ZERO"}),
    "str + int (+)": ("def f():\n    age = input()\n    return age + 1\n", {"TYPE_MISMATCH"}),
    "str(int) + str (-)": ("def f(n):\n    return 'n=' + str(n)\n", set()),
    "none deref (+)": ("def f():\n    x = None\n    return x.upper()\n", {"NONE_DEREFERENCE"}),
    "none checked (-)": ("def f(x=None):\n    if x is None:\n        x = []\n    return x.copy()\n", set()),
    "early return none (-)": ("def f(x=None):\n    if x is None:\n        return 0\n    return x.bit_length()\n", set()),
    "re.match deref (+)": ("import re\n\ndef f(s):\n    m = re.match('a', s)\n    return m.group(0)\n", {"NONE_DEREFERENCE"}),
    "inconsistent return (+)": ("def f(x):\n    if x:\n        return 1\n", {"INCONSISTENT_RETURN"}),
    "consistent return (-)": ("def f(x):\n    if x:\n        return 1\n    return 0\n", set()),
    "raise at end (-)": ("def f(x):\n    if x:\n        return 1\n    raise ValueError('no')\n", set()),
    "ignored str result (+)": ("def f():\n    s = ' a '\n    s.strip()\n    return s\n", {"IGNORED_RESULT"}),
    "sort returns None (+)": ("def f():\n    xs = [3, 1]\n    ys = xs.sort()\n    return ys\n", {"LIST_METHOD_RETURNS_NONE"}),
    "global declared in branch (-)": ("_cache = None\n\ndef f():\n    global _cache\n    if _cache is None:\n        _cache = [1]\n    return _cache\n", set()),
    "module builtin rebind (-)": ("Error = ValueError\nBlockingIOError = BlockingIOError\n", set()),
    "module dunder (-)": ("if __doc__ is not None:\n    pass\n__name__ = 'x'\n", set()),
    "none default param (-)": ("def f(x=None):\n    return x.upper()\n", set()),
    "walrus + comprehension (-)": ("def f(xs):\n    return [y for x in xs if (y := x) > 0]\n", set()),
    "name probe in try (-)": ("try:\n    has_key\nexcept NameError:\n    has_key = None\n", set()),
    "walrus in and (-)": ("def f(m):\n    if m and (n := m.find('x')) > 0:\n        return m[:n]\n    return m\n", set()),
    "loop in try (-)": ("def clear(self):\n    try:\n        while True:\n            self.pop()\n    except KeyError:\n        print('empty')\n", set()),
    "infinite generator (-)": ("def count():\n    i = 0\n    while True:\n        yield i\n        i += 1\n", set()),
    "method calls other global (-)": ("def Lock():\n    return 1\n\nclass C:\n    def Lock(self):\n        from x import Lock\n        return Lock()\n    def f(self):\n        return Lock()\n", set()),
    "self.method recursion (+)": ("class C:\n    def f(self, n):\n        return self.f(n - 1)\n", {"RECURSION_NO_BASE_CASE"}),
    "rebound iterated list (-)": ("def f(args):\n    for a in args:\n        args = []\n        args.append(a)\n    return args\n", set()),
    "clear with args on self (-)": ("class C:\n    def f(self):\n        for c in self:\n            self.clear(c)\n", set()),
    "round probe in try (-)": ("def f(n):\n    try:\n        round(n)\n    except TypeError:\n        return 0\n    return 1\n", set()),
    "param shadows function (-)": ("class C:\n    def _close(self, _close=print):\n        _close(1)\n", set()),
    "maybe None not reported (-)": ("def f(c):\n    x = None\n    if c:\n        x = [1]\n    return x.copy()\n", set()),
    "clean function (-)": ("def add(a, b):\n    return a + b\n", set()),
}


if __name__ == "__main__":
    from agents.code_understanding import analyze_code

    failures = 0
    for name, (code, expected) in EXAMPLES.items():
        found = {f["rule_id"] for f in detect_bugs(analyze_code(code))}
        status = "ok  " if found == expected else "FAIL"
        failures += found != expected
        print(f"{status} {name}: found={sorted(found)} expected={sorted(expected)}")

    print(f"\n{failures} failing example(s)")