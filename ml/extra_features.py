
import ast
import warnings

NODE_TYPES = [
    "For", "While", "If", "IfExp", "Call", "Subscript", "Attribute", "BinOp",
    "UnaryOp", "Compare", "BoolOp", "ListComp", "SetComp", "DictComp",
    "GeneratorExp", "Lambda", "FunctionDef", "ClassDef", "Return", "Assign",
    "AugAssign", "AnnAssign", "Break", "Continue", "Try", "With", "Import",
    "ImportFrom", "Dict", "List", "Set", "Tuple", "Slice", "Starred", "Yield",
    "Global", "Assert", "Raise", "Pass", "Delete", "Name", "Constant",
    "JoinedStr", "NamedExpr", "Expr",
]
OPS = [
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitAnd", "BitOr", "BitXor",
]
CALLS = [
    "range", "len", "input", "int", "map", "list", "sorted", "sort", "sum",
    "max", "min", "abs", "print", "append", "pop", "popleft", "appendleft",
    "heappush", "heappop", "heapify", "bisect_left", "bisect_right", "bisect",
    "insort", "deque", "Counter", "defaultdict", "set", "dict", "join",
    "split", "strip", "count", "index", "find", "replace", "sqrt", "pow",
    "lru_cache", "cache", "enumerate", "zip", "reversed", "reverse", "extend",
    "insert", "remove", "add", "discard", "get", "keys", "values", "items",
    "upper", "lower", "isdigit", "format", "str", "float", "bin", "gcd",
    "factorial", "permutations", "combinations", "product", "accumulate",
    "copy", "deepcopy", "setdefault", "most_common", "exit", "any", "all",
]
EXTRA_NAMES = (
    ["n_" + t for t in NODE_TYPES]
    + ["op_" + o for o in OPS]
    + ["call_" + c for c in CALLS]
    + ["in_ops", "in_ops_in_loop", "distinct_names"]
)


def extra_features(code):
    vec = dict.fromkeys(EXTRA_NAMES, 0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except Exception:
        return vec

    def has_in(node):
        return isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        )

    names = set()
    for node in ast.walk(tree):
        key = "n_" + type(node).__name__
        if key in vec:
            vec[key] += 1

        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            op_key = "op_" + type(node.op).__name__
            if op_key in vec:
                vec[op_key] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name and "call_" + name in vec:
                vec["call_" + name] += 1
        elif isinstance(node, ast.Name):
            names.add(node.id)

        if has_in(node):
            vec["in_ops"] += 1

    for loop in ast.walk(tree):
        if isinstance(loop, (ast.For, ast.While)):
            vec["in_ops_in_loop"] += sum(1 for n in ast.walk(loop) if has_in(n))

    vec["distinct_names"] = len(names)
    return vec