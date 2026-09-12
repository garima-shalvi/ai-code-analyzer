import ast
from representation.schema import create_representation
def get_max_nesting(node, depth=0):
    max_depth = depth

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.If)):
            max_depth = max(max_depth, get_max_nesting(child, depth + 1))
        else:
            max_depth = max(max_depth, get_max_nesting(child, depth))

    return max_depth

def analyze_code(code):
    tree = ast.parse(code)

    function_count = 0
    loop_count = 0
    condition_count = 0
    call_count = 0
    array_access_count = 0
    recursion = False
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
            array_access_count += 1

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            calls = []
            defined = []
            used = []

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)

                        if child.func.id == function_name:
                            recursion = True

                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        defined.append(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        used.append(child.id)

            function_relationships[function_name] = calls
            variable_scope[function_name] = {
                "defined": defined,
                "used": used
            }

    features = {
    "function_count": function_count,
    "loop_count": loop_count,
    "condition_count": condition_count,
    "call_count": call_count,
    "array_access_count": array_access_count,
    "max_nesting_depth": get_max_nesting(tree),
    "recursion": recursion,
    "function_relationships": function_relationships,
    "variable_scope": variable_scope
    }
    return create_representation(features)
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def process(arr):
    total = 0
    for i in range(len(arr)):
        if arr[i] > 0:
            total += factorial(arr[i])
    return total
"""

result = analyze_code(code)

print(result)