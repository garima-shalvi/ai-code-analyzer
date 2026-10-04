import ast
from representation.finding import create_finding
from agents.code_understanding import analyze_code

def detect_mutable_default_arguments(tree):
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = node.args.defaults

            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    findings.append(create_finding(
                        rule_id="MUTABLE_DEFAULT_ARGUMENT",
                        source="static_rule",
                        category="mutable-default-argument",
                        message="Mutable objects should not be used as default function arguments.",
                        line=default.lineno,
                        column=default.col_offset,
                        severity="medium",
                        confidence=0.99,
                        evidence=ast.unparse(default)
                    ))

    return findings

def detect_bare_except(tree):
    findings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(create_finding(
                    rule_id="BARE_EXCEPT",
                    source="static_rule",
                    category="bare-except",
                    message="Bare except catches all exceptions and can hide unexpected errors.",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="medium",
                    confidence=0.99,
                    evidence="except:"
                ))

    return findings

def detect_unreachable_code(tree):
    findings = []

    def check_block(statements):
        terminated = False

        for statement in statements:
            if terminated:
                findings.append(create_finding(
                    rule_id="UNREACHABLE_CODE",
                    source="static_rule",
                    category="unreachable-code",
                    message="This statement is unreachable because the previous statement always terminates execution.",
                    line=statement.lineno,
                    column=statement.col_offset,
                    severity="medium",
                    confidence=0.99,
                    evidence=ast.unparse(statement)
                ))
                continue

            if isinstance(statement, (ast.Return, ast.Raise)):
                terminated = True

            elif isinstance(statement, (ast.Break, ast.Continue)):
                terminated = True

            if isinstance(statement, (ast.If, ast.For, ast.While)):
                check_block(statement.body)
                check_block(statement.orelse)

            elif isinstance(statement, ast.Try):
                check_block(statement.body)
                for handler in statement.handlers:
                    check_block(handler.body)
                check_block(statement.orelse)
                check_block(statement.finalbody)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_block(node.body)

    return findings

def detect_bugs(representation):
    tree = representation["tree"]

    findings = []

    findings.extend(detect_mutable_default_arguments(tree))
    findings.extend(detect_bare_except(tree))
    findings.extend(detect_unreachable_code(tree))

    return findings

if __name__ == "__main__":
    code = """
def process(items=[]):
    try:
        x = 10 / 0
    except:
        pass

    return items
    print("unreachable")
"""

    representation = analyze_code(code)

    findings = detect_bugs(representation)

    for finding in findings:
        print(finding)