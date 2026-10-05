import ast
from representation.schema import create_representation, create_error_representation


def get_max_nesting(node, depth=0):
    max_depth = depth

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.If)):
            max_depth = max(max_depth, get_max_nesting(child, depth + 1))
        else:
            max_depth = max(max_depth, get_max_nesting(child, depth))

    return max_depth


def iter_own_body(node):
    
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            
            continue
        stack.extend(ast.iter_child_nodes(current))


def detect_recursive_functions(function_relationships):
    
    recursive = set()

    def has_cycle_back_to(start, current, visited):
        for callee in function_relationships.get(current, []):
            if callee == start:
                return True
            if callee in visited or callee not in function_relationships:
                continue
            visited.add(callee)
            if has_cycle_back_to(start, callee, visited):
                return True
        return False

    for func_name in function_relationships:
        if has_cycle_back_to(func_name, func_name, {func_name}):
            recursive.add(func_name)

    return recursive


def analyze_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return create_error_representation(
            "syntax_error",
            f"line {e.lineno}: {e.msg}"
        )

    function_count = 0
    loop_count = 0
    condition_count = 0
    call_count = 0
    subscript_access_count = 0
    function_relationships = {}
    variable_scope = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
        elif isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        elif isinstance(node, ast.If):
            condition_count += 1
        elif isinstance(node, ast.Call):
            call_count += 1
        elif isinstance(node, ast.Subscript):
            subscript_access_count += 1

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            calls = []
            defined = set()
            used = set()

            for child in iter_own_body(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    calls.append(child.func.id)

                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        defined.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        used.add(child.id)

            function_relationships[function_name] = calls
            variable_scope[function_name] = {
                "defined": sorted(defined),
                "used": sorted(used)
            }

    recursive_functions = detect_recursive_functions(function_relationships)

    features = {
        "function_count": function_count,
        "loop_count": loop_count,
        "max_loop_nesting": get_max_loop_nesting(tree),
        "condition_count": condition_count,
        "call_count": call_count,
        "subscript_access_count": subscript_access_count,
        "max_nesting_depth": get_max_nesting(tree),
        "loop_details": get_loop_details(tree),
        "space_complexity": get_space_complexity(tree),
        "recursive_functions": sorted(recursive_functions),
        "function_relationships": function_relationships,
        "variable_scope": variable_scope
    }
    return create_representation(features, tree)
def get_max_loop_nesting(node, depth=0):
    max_depth = depth

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While)):
            max_depth = max(
                max_depth,
                get_max_loop_nesting(child, depth + 1)
            )
        else:
            max_depth = max(
                max_depth,
                get_max_loop_nesting(child, depth)
            )

    return max_depth

def get_loop_bound_type(node):
    if not isinstance(node, ast.For):
        return "unknown"

    iterable = node.iter

    if isinstance(iterable, ast.Call):
        if isinstance(iterable.func, ast.Name) and iterable.func.id == "range":
            for arg in iterable.args:
                if isinstance(arg, ast.Constant):
                    continue
                return "input-dependent"
            return "constant"

    if isinstance(iterable, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Constant)):
        return "constant"

    if isinstance(iterable, ast.Name):
        return "input-dependent"

    return "unknown"

def get_space_complexity(tree):
    max_space = 0

    for node in ast.walk(tree):

        if isinstance(node, ast.ListComp):
            depth = 1

            if isinstance(node.elt, ast.ListComp):
                depth = 2

            elif isinstance(node.elt, ast.BinOp):
                if isinstance(node.elt.left, ast.List):
                    depth = 2

            max_space = max(max_space, depth)

        elif isinstance(node, ast.BinOp):
            if (
                isinstance(node.op, ast.Mult)
                and (
                    isinstance(node.left, ast.List)
                    or isinstance(node.right, ast.List)
                )
                and (
                    isinstance(node.left, ast.Name)
                    or isinstance(node.right, ast.Name)
                )
            ):
                max_space = max(max_space, 1)

        elif isinstance(node, ast.List):
            is_constant = True

            for element in node.elts:
                if isinstance(element, ast.Name):
                    is_constant = False

            if not is_constant:
                max_space = max(max_space, 1)

        elif isinstance(node, (ast.Dict, ast.Set)):
            max_space = max(max_space, 1)

    return max_space

def get_loop_details(node, depth=0):
    loops = []

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While)):
            loops.append({
                "line": child.lineno,
                "type": "for" if isinstance(child, ast.For) else "while",
                "bound_type": get_loop_bound_type(child),
                "nesting_depth": depth + 1
            })

            loops.extend(get_loop_details(child, depth + 1))
        else:
            loops.extend(get_loop_details(child, depth))

    return loops
def get_functions(tree):
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node)

    return functions

def get_function_features(function):
    loop_count = 0
    condition_count = 0
    call_count = 0
    subscript_access_count = 0

    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1

        elif isinstance(node, ast.If):
            condition_count += 1

        elif isinstance(node, ast.Call):
            call_count += 1

        elif isinstance(node, ast.Subscript):
            subscript_access_count += 1

    return {
        "function_name": function.name,
        "loop_count": loop_count,
        "condition_count": condition_count,
        "call_count": call_count,
        "subscript_access_count": subscript_access_count,
        "max_nesting_depth": get_max_nesting(function),
        "max_loop_nesting": get_max_loop_nesting(function),
        "is_recursive": int(is_function_recursive(function)),
        "function_length": function.end_lineno - function.lineno + 1
    }

def extract_function_features(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = get_functions(tree)

    features = []

    for function in functions:
        features.append(get_function_features(function))

    return features

def is_function_recursive(function):
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == function.name:
                    return True

    return False

def get_target_function(code, method):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    method = method.strip()

    if method.startswith("class "):
        return None

    if method.startswith("def "):
        method = method[4:]

    function_name = method.split("(")[0].strip()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node

    return None

if __name__ == "__main__":
    valid_code = """
def add(a, b):
    return a + b

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

    invalid_code = """
def broken(
"""

    print(extract_function_features(valid_code))
    print(extract_function_features(invalid_code))