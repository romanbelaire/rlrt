"""Shared AST + import gate for generated tool and grader code.

All generated Python files must pass this gate before being written to disk
or imported at runtime. Fail fast — never silently skip a failing file.
"""

import ast
import importlib.util
import inspect
import types

BANNED_NAMES = {"os", "sys", "subprocess", "socket", "requests", "httpx", "open",
                "eval", "exec", "__import__", "builtins", "globals", "locals"}
ALLOWED_IMPORTS = {"inspect_ai", "typing"}


def validate_tool_source(src: str) -> None:
    """Raise ValueError if src fails any static check for a generated @tool file."""
    tree = ast.parse(src)  # raises SyntaxError on bad code

    # Collect all Name references and import roots
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    import_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_roots.add(node.module.split(".")[0])

    forbidden_names = names & BANNED_NAMES
    if forbidden_names:
        raise ValueError(f"Generated tool contains banned names: {forbidden_names}")

    forbidden_imports = import_roots - ALLOWED_IMPORTS - {""}
    if forbidden_imports:
        raise ValueError(f"Generated tool contains disallowed imports: {forbidden_imports}")

    # Must contain exactly one @tool-decorated function definition
    tool_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(
            (isinstance(d, ast.Name) and d.id == "tool") or
            (isinstance(d, ast.Attribute) and d.attr == "tool")
            for d in n.decorator_list
        )
    ]
    if len(tool_defs) != 1:
        raise ValueError(
            f"Expected exactly 1 @tool function, found {len(tool_defs)}"
        )

    # The @tool function must contain exactly one async inner function
    outer = tool_defs[0]
    inner_async = [
        n for n in ast.walk(outer)
        if isinstance(n, ast.AsyncFunctionDef) and n is not outer
    ]
    if len(inner_async) != 1:
        raise ValueError(
            f"Expected exactly 1 async inner function, found {len(inner_async)}"
        )


def validate_grader_source(src: str) -> None:
    """Raise ValueError if src fails any static check for a generated grader file."""
    tree = ast.parse(src)

    # Must define an async function whose name starts with 'grade_'
    grade_fns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("grade_")
    ]
    if not grade_fns:
        raise ValueError("Generated grader has no async grade_* function")

    # No banned names
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden_names = names & BANNED_NAMES
    if forbidden_names:
        raise ValueError(f"Generated grader contains banned names: {forbidden_names}")


def load_tool_from_source(src: str, module_name: str) -> types.ModuleType:
    """Compile and exec validated tool source in an isolated module; return the module."""
    mod = types.ModuleType(module_name)
    exec(compile(src, module_name, "exec"), mod.__dict__)
    return mod


def load_grader_from_source(src: str, module_name: str) -> types.ModuleType:
    """Compile and exec validated grader source in an isolated module; return the module."""
    mod = types.ModuleType(module_name)
    exec(compile(src, module_name, "exec"), mod.__dict__)
    return mod
