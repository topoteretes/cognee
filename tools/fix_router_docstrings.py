#!/usr/bin/env python3
"""
Auto-fix router docstring parameter mismatches found by check_router_docstrings.

For every issue the checker reports, rewrite the handler's docstring in place:

- missing docstring: generate one with a summary line and parameter sections
  built from the FastAPI/Pydantic metadata (Query/Form/Path/Field descriptions).
- parameter not documented: append a ``- **name** (type): description`` bullet
  to the matching ``## Path/Query/Request Parameters`` section, creating the
  section when absent. Descriptions come from the parameter's own metadata.
- documented parameter that does not exist: remove the stale bullet.

The goal is a mechanically correct starting point for human review — the
generated wording mirrors what the OpenAPI schema already shows. Intended to
run in CI (see the router docstring sync workflow), which commits the result
to a bot branch and opens a PR.

Exit codes: 0 = nothing to fix or all fixed, 2 = app import failed.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
import textwrap
import typing
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_router_docstrings import (  # noqa: E402
    actual_params,
    documented_params,
    iter_api_routes,
    normalize,
)

PATH_SECTION = "Path Parameters"
QUERY_SECTION = "Query Parameters"
REQUEST_SECTION = "Request Parameters"
SECTION_ORDER = [PATH_SECTION, QUERY_SECTION, REQUEST_SECTION]

# Headings that parameter sections should be inserted before.
TRAILING_HEADINGS = ("response", "error", "notes", "note", "example", "next steps")

FALLBACK_DESCRIPTION = "No description provided in code yet."

# Wording for parameter names that recur across cognee routers with a fixed
# meaning. Used only when the parameter declares no description of its own.
KNOWN_DESCRIPTIONS = {
    "dataset_id": "UUID of the dataset (from GET /api/v1/datasets).",
    "dataset_ids": "UUIDs of the datasets (from GET /api/v1/datasets).",
    "dataset_name": "Name of the target dataset.",
    "data_id": "UUID of the data item (from GET /api/v1/datasets/{dataset_id}/data).",
    "session_id": "Client-supplied session identifier — the same value passed as "
    "session_id to POST /api/v1/remember.",
    "agent_id": "The agent's user ID (from GET /api/v1/agents/list).",
    "agent_session_name": "Name of the agent connection.",
    "api_key_id": "UUID of the API key (from GET /api/v1/auth/api-keys).",
    "provider": "Key of a registered OAuth provider (see GET /api/v1/integrations/status).",
    "plugin_key": "Key of a known plugin (see GET /api/v1/integrations/status).",
    "skill_id": "ID of the skill (from GET /api/v1/skills/).",
    "proposal_id": "ID of the skill-improvement proposal.",
    "email": "Email address of the user.",
    "limit": "Maximum number of rows to return.",
    "offset": "Number of rows to skip for pagination.",
    "order_by": "Column to sort by.",
    "descending": "Sort in descending order.",
    "metadata": "Free-form metadata object.",
    "node_set": "Named node sets to tag the data with, for filtered retrieval.",
    "node_name": "Restrict the operation to these named node sets.",
    "top_k": "Maximum number of results to return.",
    "run_in_background": "Return immediately and continue processing server-side.",
}
KNOWN_DESCRIPTIONS = {normalize(key): value for key, value in KNOWN_DESCRIPTIONS.items()}


def _literal_values(annotation) -> list:
    """Values of any typing.Literal found inside the annotation."""
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))
    values: list = []
    for arg in typing.get_args(annotation):
        values.extend(_literal_values(arg))
    return values


def _describe(name: str, annotation, explicit: str | None, default) -> str:
    """Best mechanical description: code first, then vocabulary, then the type."""
    from pydantic_core import PydanticUndefined

    if explicit and explicit.strip():
        text = explicit.strip()
    elif normalize(name) in KNOWN_DESCRIPTIONS:
        text = KNOWN_DESCRIPTIONS[normalize(name)]
    else:
        literals = _literal_values(annotation)
        if literals:
            rendered = ", ".join(repr(value) for value in literals)
            text = f"One of: {rendered}."
        else:
            text = FALLBACK_DESCRIPTION

    has_default = default is not None and default is not ... and default is not PydanticUndefined
    if has_default and "default" not in text.lower():
        text = f"{text} Defaults to {default!r}."
    return text


def format_annotation(annotation) -> str:
    """Compact, readable type string for a parameter annotation.

    Recursively strips Annotated metadata (which can contain schema objects)
    and module paths, keeping only class names and type structure.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return "Any"
    if annotation is type(None):
        return "None"

    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return format_annotation(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        rendered = ", ".join(repr(value) for value in typing.get_args(annotation))
        return f"Literal[{rendered}]"
    if origin is typing.Union or str(origin) == "types.UnionType":
        args = list(typing.get_args(annotation))
        if type(None) in args and len(args) == 2:
            other = args[0] if args[1] is type(None) else args[1]
            return f"Optional[{format_annotation(other)}]"
        rendered = ", ".join(format_annotation(arg) for arg in args)
        return f"Union[{rendered}]"
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin)).capitalize()
        origin_name = {"List": "List", "Dict": "Dict", "Set": "Set", "Tuple": "Tuple"}.get(
            origin_name, origin_name
        )
        rendered = ", ".join(format_annotation(arg) for arg in typing.get_args(annotation))
        return f"{origin_name}[{rendered}]"

    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return re.sub(r"\w+(\.\w+)+\.", "", str(annotation))


@dataclass
class ParamDoc:
    name: str
    section: str
    type_text: str
    description: str

    def bullet(self) -> list[str]:
        text = f"- **{self.name}** ({self.type_text}): {self.description}"
        return textwrap.wrap(text, width=92, subsequent_indent="  ") or [text]


@dataclass
class EndpointFix:
    endpoint: object
    method: str
    path: str
    missing_docstring: bool = False
    add: list[ParamDoc] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)


def _param_metadata(parameter: inspect.Parameter):
    """The fastapi.params object attached to a signature parameter, if any."""
    from fastapi import params as fastapi_params

    candidates = [parameter.default]
    if typing.get_origin(parameter.annotation) is typing.Annotated:
        candidates.extend(typing.get_args(parameter.annotation)[1:])
    for candidate in candidates:
        if isinstance(candidate, (fastapi_params.Param, fastapi_params.Body)):
            return candidate
    return None


def _section_for(parameter: inspect.Parameter, route_path: str, name: str) -> str:
    from fastapi import params as fastapi_params

    meta = _param_metadata(parameter)
    if isinstance(meta, fastapi_params.Path):
        return PATH_SECTION
    if isinstance(meta, fastapi_params.Query):
        return QUERY_SECTION
    if meta is not None:
        return REQUEST_SECTION
    # No explicit marker: FastAPI treats path-template names as path params
    # and other scalars as query params.
    return PATH_SECTION if f"{{{name}}}" in route_path else QUERY_SECTION


def collect_param_docs(endpoint, route_path: str) -> dict[str, ParamDoc]:
    """Documentation source material for every client-facing parameter."""
    import fastapi
    from fastapi import params as fastapi_params
    from pydantic import BaseModel

    from check_router_docstrings import _is_dependency, _pydantic_models

    framework_types = (
        fastapi.Request,
        fastapi.Response,
        fastapi.BackgroundTasks,
        fastapi.WebSocket,
    )

    docs: dict[str, ParamDoc] = {}
    try:
        signature = inspect.signature(endpoint)
    except (TypeError, ValueError):
        return docs

    for name, parameter in signature.parameters.items():
        if _is_dependency(parameter):
            continue
        annotation = parameter.annotation
        base = annotation
        if typing.get_origin(annotation) is typing.Annotated:
            base = typing.get_args(annotation)[0]
        if inspect.isclass(base) and issubclass(base, framework_types):
            continue

        models = _pydantic_models(annotation)
        if models:
            for model in models:
                if not (inspect.isclass(model) and issubclass(model, BaseModel)):
                    continue
                for field_name, model_field in model.model_fields.items():
                    wire_name = model_field.alias or field_name
                    docs[normalize(wire_name)] = ParamDoc(
                        name=wire_name,
                        section=REQUEST_SECTION,
                        type_text=format_annotation(model_field.annotation),
                        description=_describe(
                            wire_name,
                            model_field.annotation,
                            model_field.description,
                            model_field.default,
                        ),
                    )
            continue

        meta = _param_metadata(parameter)
        alias = meta.alias if isinstance(meta, fastapi_params.Param) and meta.alias else None
        wire_name = alias or name
        description = getattr(meta, "description", None) if meta is not None else None
        default = meta.default if meta is not None else parameter.default
        if default is inspect.Parameter.empty:
            default = ...
        docs[normalize(wire_name)] = ParamDoc(
            name=wire_name,
            section=_section_for(parameter, route_path, name),
            type_text=format_annotation(annotation),
            description=_describe(wire_name, annotation, description, default),
        )
    return docs


def collect_fixes() -> list[EndpointFix]:
    from cognee.api.client import app

    fixes: list[EndpointFix] = []
    seen: set[int] = set()

    for path, route in iter_api_routes(app):
        if not route.include_in_schema:
            continue
        endpoint = inspect.unwrap(route.endpoint)
        module = getattr(endpoint, "__module__", "") or ""
        if not module.startswith("cognee."):
            continue
        if id(endpoint) in seen:
            continue
        seen.add(id(endpoint))

        method = ",".join(sorted(route.methods or []))
        docstring = inspect.getdoc(endpoint)
        canonical, accepted_norms = actual_params(endpoint)
        param_docs = collect_param_docs(endpoint, path)

        fix = EndpointFix(endpoint=endpoint, method=method, path=path)

        if not docstring:
            fix.missing_docstring = True
            fix.add = [
                param_docs[norm] for norm in sorted(param_docs) if norm in map(normalize, canonical)
            ]
            fixes.append(fix)
            continue

        bullets, tokens = documented_params(docstring)
        mentioned_norms = {normalize(name) for name in bullets | tokens}

        for name in sorted(canonical):
            norm = normalize(name)
            if norm not in mentioned_norms and norm in param_docs:
                fix.add.append(param_docs[norm])
        fix.remove = sorted(name for name in bullets if normalize(name) not in accepted_norms)

        if fix.add or fix.remove:
            fixes.append(fix)

    return fixes


# --------------------------------------------------------------------------- #
# Docstring text surgery
# --------------------------------------------------------------------------- #


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") or (
        stripped.endswith(":")
        and stripped.rstrip(":").strip().lower()
        in ("args", "arguments", "parameters", "returns", "raises", "response", "notes")
    )


def _heading_title(line: str) -> str:
    return line.strip().lstrip("#").strip().rstrip(":").strip().lower()


def _remove_stale_bullets(lines: list[str], stale: list[str]) -> list[str]:
    stale_norms = {normalize(name) for name in stale}
    result: list[str] = []
    skipping_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if skipping_indent is not None:
            indent = len(line) - len(line.lstrip())
            is_continuation = (
                stripped
                and indent > skipping_indent
                and not stripped.startswith("- ")
                and not _is_heading(line)
            )
            if is_continuation:
                continue
            skipping_indent = None
        if stripped.startswith("- **"):
            name = stripped[4:].split("**", 1)[0]
            if normalize(name) in stale_norms:
                skipping_indent = len(line) - len(line.lstrip())
                continue
        result.append(line)
    return result


def _insert_bullets(lines: list[str], additions: list[ParamDoc]) -> list[str]:
    by_section: dict[str, list[ParamDoc]] = defaultdict(list)
    for doc in additions:
        by_section[doc.section].append(doc)

    for section in SECTION_ORDER:
        docs = by_section.get(section)
        if not docs:
            continue
        bullet_lines: list[str] = []
        for doc in docs:
            bullet_lines.extend(doc.bullet())

        # Find an existing heading for this section (any parameters-flavoured
        # heading whose title contains the section's first word).
        target = None
        for index, line in enumerate(lines):
            if _is_heading(line) and section.lower() in _heading_title(line):
                target = index
                break
        if target is None and section == REQUEST_SECTION:
            # "Request Body" style headings also count as the request section.
            for index, line in enumerate(lines):
                if _is_heading(line) and "request body" in _heading_title(line):
                    target = index
                    break

        if target is not None:
            # Append at the end of the section: before the next heading.
            end = len(lines)
            for index in range(target + 1, len(lines)):
                if _is_heading(lines[index]):
                    end = index
                    break
            while end > target + 1 and not lines[end - 1].strip():
                end -= 1
            lines[end:end] = bullet_lines
            continue

        # No such section: create one before the first trailing heading.
        insert_at = len(lines)
        for index, line in enumerate(lines):
            if _is_heading(line) and _heading_title(line).startswith(TRAILING_HEADINGS):
                insert_at = index
                break
        block = [f"## {section}"] + bullet_lines + [""]
        if insert_at > 0 and lines[insert_at - 1].strip():
            block = [""] + block
        lines[insert_at:insert_at] = block

    return lines


def _rebuild_docstring(text: str, indent: str, summary_on_first_line: bool) -> list[str]:
    """Render docstring text back into source lines (with quotes)."""
    safe = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    lines = safe.splitlines() or [""]
    out: list[str] = []
    if summary_on_first_line and lines[0].strip():
        out.append(f'{indent}"""{lines[0]}')
        rest = lines[1:]
    else:
        out.append(f'{indent}"""')
        rest = lines if lines[0].strip() else lines[1:]
    for line in rest:
        out.append(f"{indent}{line}" if line.strip() else "")
    out.append(f'{indent}"""')
    return out


def _new_docstring_text(fix: EndpointFix) -> str:
    summary = fix.endpoint.__name__.replace("_", " ").strip().capitalize()
    lines = [f"{summary} — {fix.method} {fix.path}.", ""]
    lines = _insert_bullets(lines, fix.add)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def apply_fixes(fixes: list[EndpointFix]) -> dict[str, int]:
    """Group fixes per source file and rewrite the docstrings bottom-up."""
    per_file: dict[str, list[EndpointFix]] = defaultdict(list)
    for fix in fixes:
        source_file = inspect.getsourcefile(fix.endpoint)
        if source_file:
            per_file[source_file].append(fix)

    changed: dict[str, int] = {}
    for source_file, file_fixes in per_file.items():
        with open(source_file, encoding="utf-8") as handle:
            source = handle.read()
        source_lines = source.splitlines()
        tree = ast.parse(source)

        # Map function name -> AST nodes, to locate each endpoint's def.
        nodes_by_name: dict[str, list[ast.AST]] = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                nodes_by_name[node.name].append(node)

        # Bottom-up so earlier line numbers stay valid.
        located: list[tuple[int, ast.AST, EndpointFix]] = []
        for fix in file_fixes:
            candidates = nodes_by_name.get(fix.endpoint.__name__, [])
            if not candidates:
                continue
            firstline = fix.endpoint.__code__.co_firstlineno
            node = min(candidates, key=lambda n: abs(n.lineno - firstline))
            located.append((node.lineno, node, fix))
        located.sort(key=lambda item: item[0], reverse=True)

        for _, node, fix in located:
            body_first = node.body[0]
            has_docstring = (
                isinstance(body_first, ast.Expr)
                and isinstance(body_first.value, ast.Constant)
                and isinstance(body_first.value.value, str)
            )
            indent = " " * body_first.col_offset

            if fix.missing_docstring or not has_docstring:
                text = _new_docstring_text(fix)
                new_lines = _rebuild_docstring(text, indent, summary_on_first_line=True)
                source_lines[body_first.lineno - 1 : body_first.lineno - 1] = new_lines
            else:
                original = body_first.value.value
                summary_on_first_line = not original.startswith(("\n", "\r"))
                text_lines = inspect.cleandoc(original).splitlines()
                text_lines = _remove_stale_bullets(text_lines, fix.remove)
                text_lines = _insert_bullets(text_lines, fix.add)
                new_lines = _rebuild_docstring("\n".join(text_lines), indent, summary_on_first_line)
                source_lines[body_first.lineno - 1 : body_first.end_lineno] = new_lines

        with open(source_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(source_lines) + "\n")
        changed[source_file] = len(located)

    return changed


def main() -> int:
    os.environ.setdefault("ENV", "dev")

    try:
        fixes = collect_fixes()
    except Exception as exc:
        print(f"Failed to import cognee API app: {exc}", file=sys.stderr)
        return 2

    if not fixes:
        print("No docstring fixes needed.")
        return 0

    changed = apply_fixes(fixes)
    for fix in fixes:
        actions = []
        if fix.missing_docstring:
            actions.append("docstring added")
        if fix.add and not fix.missing_docstring:
            actions.append(f"documented: {', '.join(doc.name for doc in fix.add)}")
        if fix.remove:
            actions.append(f"removed stale: {', '.join(fix.remove)}")
        print(f"{fix.method} {fix.path}  ->  {'; '.join(actions)}")

    print(
        f"\nFixed {len(fixes)} endpoints across {len(changed)} files "
        f"({sum(changed.values())} docstrings rewritten)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
