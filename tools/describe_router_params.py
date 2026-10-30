#!/usr/bin/env python3
"""
Generate parameter descriptions for router docstring placeholder bullets.

tools/fix_router_docstrings.py leaves ``No description provided in code yet.``
in generated bullets when a parameter has no description= metadata and no
vocabulary entry (Literal parameters keep it too, followed by their values),
and writes a stub summary line (``Name — METHOD /path.``) for handlers that
had no docstring. This script finds both under cognee/api, gathers each
endpoint's source plus the source of the models and cognee helpers it uses,
asks Claude for the descriptions and summaries, and rewrites them in place.

Runs on plain text and AST only — it does not import cognee, so it needs no
project environment; only the ``anthropic`` package.

Exit codes: 0 = nothing to do or descriptions applied (a missing API key is
reported and skipped, not failed), 1 = the Claude call failed.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PLACEHOLDER = "No description provided in code yet."
BULLET_RE = re.compile(
    r"^(?P<indent>\s*)- \*\*(?P<name>\w+)\*\* \((?P<type>[^)]*)\): (?P<rest>.*)$"
)
API_ROOT = Path("cognee/api")

# The stub summary fix_router_docstrings.py writes for a handler with no docstring.
STUB_SUMMARY_RE = re.compile(
    r'^(?P<indent>\s*)"""[A-Z][\w ]* — [A-Z]+(?:,[A-Z]+)* /\S*\.(?P<close>""")?$'
)
# Names that resolve to cognee code but say nothing about a route's parameters.
SKIP_CONTEXT = {"User", "get_authenticated_user", "log_usage", "new_span"}
MAX_SYMBOL_CHARS = 5000
MAX_CONTEXT_CHARS = 24000

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You write the REST API reference for cognee, an open-source AI memory platform
(knowledge graphs + vector search for AI agents): parameter descriptions and,
where a handler has only a stub, its summary.

For each endpoint you receive the handler source and the source of the request and
response models and cognee helpers it uses (related_source). Parameter descriptions
follow the style of these existing ones:

- "UUID of the dataset (from GET /api/v1/datasets)."
- "Inline SKILL.md markdown to ingest as a Skill node."
- "Time window filtered on last_activity_at: 24h, 7d, 30d, or all."
- "Maximum number of rows to return."

Evidence (these rules outrank everything below):
- State only what the source you were given shows. When behaviour depends on code
  you were not given (a function that is called but not included, a setting, an
  exception class, a registry or cache), leave that behaviour out. A shorter true
  description is always better than a complete-sounding guess.
- Do not write "only", "never", "always", "at most", "returns the existing" or
  similar absolute claims unless the code you were given enforces them.
- How FastAPI behaves, so you do not have to guess:
  - Request validation failures return 422 before the handler runs: a missing or
    ill-typed parameter, a Field limit, or a ValueError raised in a Pydantic
    model validator.
  - HTTPException(status, ...) raised in the shown code returns that status.
  - Any other exception that the shown code does not catch: do not claim a status.
  - A handler that returns None responds with its success status and a JSON null
    body, not an empty body. The success status is 200 unless the decorator sets
    status_code=.

Rules for parameter descriptions:
- One or two sentences, at most 220 characters, ending with a period.
- Say what the parameter does in THIS endpoint. Read the models' validators, Field
  limits, comments and docstrings: they hold the rules callers need.
- Where the source shows it, include what omitting the parameter means, what each
  allowed value selects (when values are listed), what happens when a limit is
  reached, and which other parameters it requires or excludes.
- Do not restate the type, the parameter name, or anything in "already_rendered"
  (defaults, allowed values and numeric limits are rendered after your text).
- Plain, factual reference tone; no marketing language.

Rules for summaries (only for endpoints with "needs_summary": true):
- summary: one imperative line, at most 90 characters, ending with a period,
  e.g. "List the conflicts in a dataset.". Never repeat the method or path.
- details: zero to two sentences. Name an error status only when endpoint_source
  itself raises HTTPException with it. Describe the response only from the
  response_model or a dict literal the handler returns. Leave details empty when
  neither applies; most endpoints need none.
"""


@dataclass
class Placeholder:
    file: Path
    start: int  # index of the bullet line
    end: int  # index one past the last continuation line
    indent: str
    name: str
    type_text: str
    suffix: str  # text after the placeholder sentence, e.g. "Defaults to True."

    @property
    def key(self) -> str:
        return f"{self.file}::{self.start + 1}::{self.name}"


def find_placeholders(root: Path) -> list[Placeholder]:
    found: list[Placeholder] = []
    for path in sorted(root.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = BULLET_RE.match(lines[index])
            if not match:
                index += 1
                continue
            # Swallow wrapped continuation lines belonging to this bullet.
            end = index + 1
            while end < len(lines):
                nxt = lines[end]
                deeper = len(nxt) - len(nxt.lstrip()) > len(match.group("indent"))
                if nxt.strip() and deeper and not nxt.lstrip().startswith(("-", "#")):
                    end += 1
                else:
                    break
            full_rest = " ".join(
                [match.group("rest")] + [lines[i].strip() for i in range(index + 1, end)]
            )
            if PLACEHOLDER in full_rest:
                found.append(
                    Placeholder(
                        file=path,
                        start=index,
                        end=end,
                        indent=match.group("indent"),
                        name=match.group("name"),
                        type_text=match.group("type"),
                        suffix=full_rest.split(PLACEHOLDER, 1)[1].strip(),
                    )
                )
            index = end
    return found


def enclosing_source(path: Path, line_index: int, max_chars: int = 6000) -> str:
    """Source of the innermost function containing the given line (with decorators)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lineno = line_index + 1
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            if start <= lineno <= (node.end_lineno or start) and (best is None or start > best[0]):
                best = (start, node.end_lineno)
    if best is None:
        return ""
    lines = source.splitlines()[best[0] - 1 : best[1]]
    return "\n".join(lines)[:max_chars]


@dataclass
class StubSummary:
    file: Path
    line: int  # index of the stub summary line
    indent: str
    closes: bool  # the stub line also closes the docstring

    @property
    def key(self) -> str:
        return f"{self.file}::{self.line + 1}"


def find_stub_summaries(root: Path) -> list[StubSummary]:
    found: list[StubSummary] = []
    for path in sorted(root.rglob("*.py")):
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            match = STUB_SUMMARY_RE.match(line)
            if match:
                found.append(
                    StubSummary(
                        file=path,
                        line=index,
                        indent=match.group("indent"),
                        closes=bool(match.group("close")),
                    )
                )
    return found


def _module_file(module: str) -> Path | None:
    if not module.startswith("cognee"):
        return None
    base = Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _absolute_module(path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = list(path.with_suffix("").parts[: -node.level])
    return ".".join(package + ([node.module] if node.module else []))


def resolve_symbol(path: Path, name: str, depth: int = 0) -> tuple[Path, ast.AST] | None:
    """Where a name used in ``path`` is defined: a module-level def/class in the
    same file, or one imported from a cognee module (following re-exports)."""
    if depth > 4:
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return path, node
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    target = _module_file(_absolute_module(path, node))
                    if target is not None:
                        return resolve_symbol(target, alias.name, depth + 1)
    return None


def _node_source(path: Path, node: ast.AST) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1 : node.end_lineno])[:MAX_SYMBOL_CHARS]


def _used_names(node: ast.AST) -> list[str]:
    """Names a handler (or model) refers to: annotations, decorator arguments
    such as response_model, and the functions it calls — in source order."""
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in names:
            names.append(child.id)
    return names


def related_source(path: Path, line_index: int) -> str:
    """Source of the cognee models and helpers the enclosing handler uses.

    One hop from the handler, plus the classes those models refer to (a model
    nested in a request body carries its own validators).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    handler = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            inside = start <= line_index + 1 <= (node.end_lineno or start)
            if inside and (handler is None or node.lineno > handler.lineno):
                handler = node
    if handler is None:
        return ""
    # The decorator (response_model=...) belongs to the handler but sits outside it.
    queue = [(name, path, 0) for name in _used_names(handler)]
    for decorator in handler.decorator_list:
        queue += [(name, path, 0) for name in _used_names(decorator)]

    seen: set[tuple[str, str]] = set()
    chunks: list[str] = []
    total = 0
    while queue:
        name, origin, hop = queue.pop(0)
        if name in SKIP_CONTEXT or name == handler.name:
            continue
        found = resolve_symbol(origin, name)
        if found is None:
            continue
        file, node = found
        marker = (str(file), getattr(node, "name", name))
        if marker in seen:
            continue
        seen.add(marker)
        chunk = f"# {file}\n{_node_source(file, node)}"
        if total + len(chunk) > MAX_CONTEXT_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
        if isinstance(node, ast.ClassDef) and hop == 0:
            queue += [(child, file, 1) for child in _used_names(node)]
    return "\n\n".join(chunks)


def _endpoint_key(path: Path, line_index: int) -> str:
    return f"{path}::{enclosing_line(path, line_index)}"


def enclosing_line(path: Path, line_index: int) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    best = 0
    for node in ast.walk(tree):
        is_function = isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        if is_function and node.lineno <= line_index + 1 <= (node.end_lineno or node.lineno):
            best = max(best, node.lineno)
    return best


def generate_descriptions(
    placeholders: list[Placeholder], stubs: list[StubSummary]
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """(description per placeholder key, (summary, details) per stub key)."""
    import anthropic

    endpoints: dict[str, dict] = {}

    def endpoint(path: Path, line_index: int) -> dict:
        key = _endpoint_key(path, line_index)
        if key not in endpoints:
            endpoints[key] = {
                "endpoint": key,
                "needs_summary": False,
                "endpoint_source": enclosing_source(path, line_index),
                "related_source": related_source(path, line_index),
                "parameters": [],
            }
        return endpoints[key]

    for item in placeholders:
        endpoint(item.file, item.start)["parameters"].append(
            {
                "key": item.key,
                "parameter": item.name,
                "type": item.type_text,
                "already_rendered": item.suffix,
            }
        )
    stub_keys: dict[str, str] = {}
    for stub in stubs:
        entry = endpoint(stub.file, stub.line)
        entry["needs_summary"] = True
        stub_keys[entry["endpoint"]] = stub.key

    schema = {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["key", "description"],
                    "additionalProperties": False,
                },
            },
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string"},
                        "summary": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["endpoint", "summary", "details"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["descriptions", "summaries"],
        "additionalProperties": False,
    }

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        # Safety classifiers can decline a request; the server-side default
        # fallback re-runs it on the recommended substitute model instead.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"format": {"type": "json_schema", "schema": schema}},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a description for each parameter (one entry per parameter "
                    "key) and a summary for each endpoint with needs_summary set (one "
                    "entry per such endpoint).\n\n" + json.dumps(list(endpoints.values()), indent=2)
                ),
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (stop_reason=refusal)")

    text = next(block.text for block in response.content if block.type == "text")
    data = json.loads(text)
    descriptions = {entry["key"]: entry["description"].strip() for entry in data["descriptions"]}
    summaries = {
        stub_keys[entry["endpoint"]]: (entry["summary"].strip(), entry["details"].strip())
        for entry in data["summaries"]
        if entry["endpoint"] in stub_keys
    }
    return descriptions, summaries


Edit = tuple[int, int, list[str]]  # replace lines[start:end] with the new lines


def summary_edits(
    stubs: list[StubSummary], summaries: dict[str, tuple[str, str]]
) -> dict[Path, list[Edit]]:
    edits: dict[Path, list[Edit]] = {}
    for stub in stubs:
        summary, details = summaries.get(stub.key, ("", ""))
        if not summary:
            continue
        if not summary.endswith("."):
            summary += "."
        new = [f'{stub.indent}"""{summary}']
        if details:
            new += [""] + textwrap.wrap(
                details,
                width=max(60, 100 - len(stub.indent)),
                initial_indent=stub.indent,
                subsequent_indent=stub.indent,
            )
        if stub.closes:
            new.append(f'{stub.indent}"""')
        edits.setdefault(stub.file, []).append((stub.line, stub.line + 1, new))
    return edits


def description_edits(
    placeholders: list[Placeholder], descriptions: dict[str, str]
) -> dict[Path, list[Edit]]:
    edits: dict[Path, list[Edit]] = {}
    for item in placeholders:
        description = descriptions.get(item.key, "").strip()
        if not description:
            continue
        if not description.endswith("."):
            description += "."
        body = f"- **{item.name}** ({item.type_text}): {description}"
        if item.suffix:
            body += f" {item.suffix}"
        wrapped = textwrap.wrap(
            body,
            width=max(60, 100 - len(item.indent)),
            initial_indent=item.indent,
            subsequent_indent=item.indent + "  ",
        )
        edits.setdefault(item.file, []).append((item.start, item.end, wrapped))
    return edits


def apply_edits(*edit_sets: dict[Path, list[Edit]]) -> int:
    """Apply line-range replacements, bottom-up per file so indices stay valid."""
    per_file: dict[Path, list[Edit]] = {}
    for edit_set in edit_sets:
        for path, edits in edit_set.items():
            per_file.setdefault(path, []).extend(edits)

    applied = 0
    for path, edits in per_file.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        for start, end, new in sorted(edits, key=lambda edit: edit[0], reverse=True):
            lines[start:end] = new
            applied += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return applied


def main() -> int:
    placeholders = find_placeholders(API_ROOT)
    stubs = find_stub_summaries(API_ROOT)
    if not placeholders and not stubs:
        print("No placeholder descriptions or stub summaries found — nothing to do.")
        return 0

    print(f"Found {len(placeholders)} placeholder bullets:")
    for item in placeholders:
        print(f"  {item.key} ({item.type_text})")
    print(f"Found {len(stubs)} stub summaries:")
    for stub in stubs:
        print(f"  {stub.key}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — leaving placeholders in place.")
        return 0

    try:
        descriptions, summaries = generate_descriptions(placeholders, stubs)
    except Exception as exc:
        logger.debug("Exiting with status 1 after error in main", exc_info=True)
        print(f"Description generation failed: {exc}", file=sys.stderr)
        return 1

    by_description = description_edits(placeholders, descriptions)
    by_summary = summary_edits(stubs, summaries)
    apply_edits(by_description, by_summary)
    applied = sum(len(edits) for edits in by_description.values())
    applied_summaries = sum(len(edits) for edits in by_summary.values())
    print(f"\nApplied {applied} generated descriptions and {applied_summaries} summaries.")
    if len(placeholders) - applied:
        print(
            f"{len(placeholders) - applied} placeholders received no description "
            "and were left in place."
        )
    if len(stubs) - applied_summaries:
        print(f"{len(stubs) - applied_summaries} stub summaries were left in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
