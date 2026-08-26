#!/usr/bin/env python3
"""
Generate parameter descriptions for router docstring placeholder bullets.

tools/fix_router_docstrings.py leaves ``No description provided in code yet.``
in generated bullets when a parameter has no description= metadata, no
vocabulary entry, and no Literal values to derive from. This script finds
those placeholder bullets under cognee/api, gathers each parameter's endpoint
source as context, asks Claude for a one-sentence description of each, and
rewrites the bullets in place.

Runs on plain text and AST only — it does not import cognee, so it needs no
project environment; only the ``anthropic`` package.

Exit codes: 0 = nothing to do or descriptions applied (a missing API key is
reported and skipped, not failed), 1 = the Claude call failed.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "No description provided in code yet."
BULLET_RE = re.compile(
    r"^(?P<indent>\s*)- \*\*(?P<name>\w+)\*\* \((?P<type>[^)]*)\): (?P<rest>.*)$"
)
API_ROOT = Path("cognee/api")

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You write one-line parameter descriptions for the REST API reference of cognee,
an open-source AI memory platform (knowledge graphs + vector search for AI agents).

For each parameter you receive its name, type, and the source code of the FastAPI
endpoint handler it belongs to. Write a description the way these existing cognee
descriptions are written:

- "UUID of the dataset (from GET /api/v1/datasets)."
- "Inline SKILL.md markdown to ingest as a Skill node."
- "Time window filtered on last_activity_at: 24h, 7d, 30d, or all."
- "Maximum number of rows to return."

Rules:
- One sentence, at most 110 characters, ending with a period.
- Describe what the parameter means and does in THIS endpoint, based on how the
  source code actually uses it — never guess beyond the code.
- Do not restate the type or the parameter name.
- Do not mention defaults (they are rendered separately).
- Plain, factual reference tone; no marketing language.
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
            if start <= lineno <= (node.end_lineno or start):
                if best is None or start > best[0]:
                    best = (start, node.end_lineno)
    if best is None:
        return ""
    lines = source.splitlines()[best[0] - 1 : best[1]]
    return "\n".join(lines)[:max_chars]


def generate_descriptions(placeholders: list[Placeholder]) -> dict[str, str]:
    import anthropic

    items = [
        {
            "key": item.key,
            "parameter": item.name,
            "type": item.type_text,
            "file": str(item.file),
            "endpoint_source": enclosing_source(item.file, item.start),
        }
        for item in placeholders
    ]

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
            }
        },
        "required": ["descriptions"],
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
                    "Write a description for each of these parameters. Return one "
                    "entry per input key.\n\n" + json.dumps(items, indent=2)
                ),
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (stop_reason=refusal)")

    text = next(block.text for block in response.content if block.type == "text")
    data = json.loads(text)
    return {entry["key"]: entry["description"].strip() for entry in data["descriptions"]}


def apply_descriptions(placeholders: list[Placeholder], descriptions: dict[str, str]) -> int:
    applied = 0
    by_file: dict[Path, list[Placeholder]] = {}
    for item in placeholders:
        by_file.setdefault(item.file, []).append(item)

    for path, items in by_file.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        # Bottom-up so earlier line indices stay valid.
        for item in sorted(items, key=lambda entry: entry.start, reverse=True):
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
            lines[item.start : item.end] = wrapped
            applied += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return applied


def main() -> int:
    placeholders = find_placeholders(API_ROOT)
    if not placeholders:
        print("No placeholder descriptions found — nothing to do.")
        return 0

    print(f"Found {len(placeholders)} placeholder bullets:")
    for item in placeholders:
        print(f"  {item.key} ({item.type_text})")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — leaving placeholders in place.")
        return 0

    try:
        descriptions = generate_descriptions(placeholders)
    except Exception as exc:
        print(f"Description generation failed: {exc}", file=sys.stderr)
        return 1

    applied = apply_descriptions(placeholders, descriptions)
    missing = len(placeholders) - applied
    print(f"\nApplied {applied} generated descriptions.")
    if missing:
        print(f"{missing} placeholders received no description and were left in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
