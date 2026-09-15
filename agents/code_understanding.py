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
        "condition_count": condition_count,
        "call_count": call_count,
        "subscript_access_count": subscript_access_count,
        "max_nesting_depth": get_max_nesting(tree),
        "recursive_functions": sorted(recursive_functions),
        "function_relationships": function_relationships,
        "variable_scope": variable_scope
    }
    return create_representation(features)


if __name__ == "__main__":
    code = """
def find_max(arr):
    max_val = 0
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val
"""

import json
result = analyze_code(code)
print(json.dumps(result, indent=2))