#!/usr/bin/env python3
"""
Check that the router docstring critic only edited what it was allowed to.

Run in the fix-branch checkout after the critic step. ``HEAD`` is the commit holding
the text describe_router_params.py generated in this run and ``HEAD~1`` the text
before it; the working tree holds the critic's edits. The critic may only:

- edit files the generated commit touched,
- change docstrings the generated commit changed (never code, comments or any other
  docstring),
- keep every ``- **name**`` parameter bullet (it narrows descriptions, never drops a
  parameter).

Runs on git and AST only — no cognee environment needed.

Exit codes: 0 = edits allowed, 1 = a rule was broken.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import Counter

BULLET_RE = re.compile(r"^\s*- \*\*(\w+)\*\*", re.MULTILINE)
DocstringMap = dict[str, str]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def show(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def _has_docstring(node: ast.AST) -> bool:
    # Only these carry docstrings; a lambda's or an IfExp's body is an expression.
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = node.body
    return bool(
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def docstrings(source: str) -> DocstringMap:
    """Every docstring keyed by the dotted path of its module/class/function."""
    tree = ast.parse(source)
    found: DocstringMap = {}
    if _has_docstring(tree):
        found["<module>"] = tree.body[0].value.value

    # Handlers are nested inside router factories (and sometimes in if/try
    # blocks), so walk every node, naming each def by its enclosing defs.
    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                while name in found:  # same name defined twice: keep both
                    name += "'"
                if _has_docstring(child):
                    found[name] = child.body[0].value.value
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def without_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if _has_docstring(node):
            node.body[0].value.value = ""
    return ast.dump(tree)


def main() -> int:
    generated_files = set(git("diff", "--name-only", "HEAD~1", "HEAD").split())
    edited = git("diff", "--name-only", "HEAD").split()
    untracked = git("ls-files", "--others", "--exclude-standard", "cognee").split()

    problems: list[str] = [f"{path}: new file" for path in untracked]
    for path in edited:
        if path not in generated_files:
            problems.append(f"{path}: not touched by the generated commit")
            continue
        before, generated = show("HEAD~1", path), show("HEAD", path)
        with open(path, encoding="utf-8") as handle:
            criticised = handle.read()
        try:
            if without_docstrings(criticised) != without_docstrings(generated):
                problems.append(f"{path}: changed something other than a docstring")
                continue
        except SyntaxError as error:
            problems.append(f"{path}: no longer parses ({error})")
            continue
        if Counter(BULLET_RE.findall(criticised)) != Counter(BULLET_RE.findall(generated)):
            problems.append(f"{path}: added or removed a parameter bullet")
        old, new, final = docstrings(before), docstrings(generated), docstrings(criticised)
        for key, text in final.items():
            if new.get(key) != text and old.get(key) == new.get(key):
                problems.append(f"{path}: edited {key}, which this run did not generate")

    if problems:
        print("The critic made edits it is not allowed to make:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"Critic edits are within bounds ({len(edited)} file(s) changed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
