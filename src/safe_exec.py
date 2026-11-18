"""Best-effort safe execution of LLM-generated pandas code.

This is an AST allow-list plus restricted builtins. It blocks the obvious dangers
(imports, file/network access, eval/exec, dunder tricks) but is NOT a full sandbox.
For production, run generated code in a separate container or process.
"""
from __future__ import annotations

import ast
import io
import math
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ALLOWED_IMPORTS = {"pandas", "numpy", "math"}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
    "memoryview", "os", "sys", "subprocess", "socket", "shutil", "pathlib",
}

BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_json", "to_sql", "to_parquet", "to_feather",
    "read_csv", "read_excel", "read_pickle", "read_json", "read_sql", "read_parquet",
    "eval", "query_exec",
}

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len",
        "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "isinstance", "True", "False", "None",
    )
}


class UnsafeCodeError(Exception):
    """Raised when generated code fails the safety check."""


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        raise UnsafeCodeError(f"Generated code has a syntax error: {err}") from err

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(f"Import of '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"Import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCodeError("Access to dunder attributes is not allowed.")
            if node.attr in BLOCKED_ATTRS:
                raise UnsafeCodeError(f"Use of '.{node.attr}' is not allowed.")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCodeError("global/nonlocal statements are not allowed.")


def run_code(code: str, df: pd.DataFrame) -> tuple[object, str]:
    """Run checked code against a copy of df.

    The code should assign its final answer to a variable named `result`.
    Returns (result, captured_stdout).
    """
    check_code(code)

    def _guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise UnsafeCodeError(f"Import of '{name}' is not allowed.")
        return __import__(name, *args, **kwargs)

    builtins = dict(SAFE_BUILTINS)
    builtins["__import__"] = _guarded_import

    env = {"__builtins__": builtins, "pd": pd, "np": np, "math": math, "df": df.copy()}
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "<generated>", "exec"), env)  # noqa: S102 - guarded above
    return env.get("result"), buffer.getvalue()
