"""AST detector for FastAPI router⇄core type-contract regressions.

Grew out of a router/core signature audit that found ~91 type inconsistencies
between FastAPI endpoint signatures and the core functions they wrap. This
module statically detects the high-signal, unambiguous offenders so a ratcheted
test (``test_api_type_contract.py``) can block NEW ones from landing:

    R1 ``form_in_body_model``       — ``Form(...)`` used as a default inside a
                                       Pydantic body model (Form is for params,
                                       not JSON body fields).
    R2 ``tuple_default_on_field``   — a model field whose default is a tuple,
                                       almost always a stray trailing comma
                                       (e.g. ``name: str = (Form(...),)``).
    R3 ``weak_response_model``      — ``response_model`` declared as ``None`` /
                                       a bare ``dict``/``list``/``Dict``/``List``
                                       / ``List[Any]`` / ``Dict[..., Any]``,
                                       which erases the schema in OpenAPI.

Detection is intentionally high-precision (few rules, no guessing) so the gate
is trustworthy. It is a ratchet, not a one-shot cleanup: a baseline allowlist of
pre-existing violations is tolerated; anything new fails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

_ROUTER_METHODS = {"get", "post", "put", "patch", "delete"}
_WEAK_CONTAINER_NAMES = {"dict", "list", "Dict", "List"}
# Pydantic base + cognee's request/response DTO bases (which subclass BaseModel).
_MODEL_BASE_NAMES = {"BaseModel", "InDTO", "OutDTO"}


@dataclass(frozen=True)
class Violation:
    rule: str
    file: str  # repo-relative posix path
    symbol: str  # endpoint path / model.field — stable across line moves
    detail: str

    def key(self) -> str:
        return f"{self.rule}::{self.file}::{self.symbol}"


def _is_base_model(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id in _MODEL_BASE_NAMES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in _MODEL_BASE_NAMES:
            return True
    return False


def _calls(node: ast.AST, func_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    )


def _is_weak_response_model(value: ast.AST) -> bool:
    # None
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    # bare dict / list / Dict / List
    if isinstance(value, ast.Name) and value.id in _WEAK_CONTAINER_NAMES:
        return True
    # List[Any] / list[Any] / Dict[..., Any] / dict[..., Any]
    if isinstance(value, ast.Subscript):
        container = value.value
        cname = (
            container.id
            if isinstance(container, ast.Name)
            else getattr(container, "attr", None)
        )
        if cname in _WEAK_CONTAINER_NAMES:
            sl = value.slice
            leaves = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            # Weak only if every leaf is Any (no concrete model carried).
            if leaves and all(isinstance(le, ast.Name) and le.id == "Any" for le in leaves):
                return True
    return False


def _endpoint_label(decorator: ast.Call) -> str:
    method = decorator.func.attr if isinstance(decorator.func, ast.Attribute) else "?"
    path = ""
    if decorator.args and isinstance(decorator.args[0], ast.Constant):
        path = str(decorator.args[0].value)
    return f"{method.upper()} {path}".strip()


def detect_in_source(source: str, rel_path: str) -> List[Violation]:
    """Return the contract violations in a single python source string."""
    out: List[Violation] = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # R1 / R2 — body-model field defaults
        if isinstance(node, ast.ClassDef) and _is_base_model(node):
            for stmt in node.body:
                if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
                    continue
                field = stmt.target.id if isinstance(stmt.target, ast.Name) else "?"
                sym = f"{node.name}.{field}"
                if isinstance(stmt.value, ast.Tuple):
                    out.append(Violation("tuple_default_on_field", rel_path, sym,
                                         "model field default is a tuple (stray trailing comma?)"))
                # Form() as a field default, possibly wrapped in a tuple
                candidates = (
                    stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
                )
                if any(_calls(c, "Form") for c in candidates):
                    out.append(Violation("form_in_body_model", rel_path, sym,
                                         "Form() used as a default inside a Pydantic body model"))

        # R3 — weak response_model on router decorators
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                    continue
                if dec.func.attr not in _ROUTER_METHODS:
                    continue
                for kw in dec.keywords:
                    if kw.arg == "response_model" and _is_weak_response_model(kw.value):
                        out.append(Violation("weak_response_model", rel_path,
                                             _endpoint_label(dec),
                                             "response_model erases the schema in OpenAPI"))
    return out


def iter_router_files(api_root: Path) -> Iterator[Path]:
    for path in sorted(api_root.rglob("*.py")):
        name = path.name
        if "router" in name or name == "query_router.py":
            yield path


def detect_all(repo_root: Path) -> List[Violation]:
    api_root = repo_root / "cognee" / "api" / "v1"
    found: List[Violation] = []
    for path in iter_router_files(api_root):
        rel = path.relative_to(repo_root).as_posix()
        found.extend(detect_in_source(path.read_text(encoding="utf-8"), rel))
    return found


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[4]
    violations = detect_all(root)
    print(json.dumps(sorted(v.key() for v in violations), indent=2))
    print(f"\n{len(violations)} violations", flush=True)
