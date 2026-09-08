#!/usr/bin/env python3
"""
Auto-fix drift between ``tools/spec_extras.json`` and the live FastAPI app.

Companion to ``check_spec_extras.py``, in the same shape as
``fix_router_docstrings.py``: the checker reports, this rewrites. Two fixes are
derivable from the app and applied mechanically, one needs prose and is written
by Claude, and one is left for a human because guessing would be wrong.

Mechanical:

- **Tag has no blurb.** Insert a placeholder entry so the tag is present in the
  data file, then let ``--describe`` fill it in.
- **Blurb for a tag no route uses.** Delete it. The tag was renamed or its
  endpoints were removed, so the entry can only ever be dead weight.

With ``--describe`` (needs ``ANTHROPIC_API_KEY``):

- **Placeholder blurbs.** One structured-output call writes them all, given the
  tag plus the endpoints carrying it — their method, path, summary and the first
  lines of their description. Prose the app cannot supply.

Reported, never guessed:

- **An example pinned to a route the API no longer exposes.** The path or method
  changed, so the fix is a judgement call about which route the sample belongs to
  now. Inventing a mapping here would silently attach a stale sample body to a
  live endpoint.
- **Malformed ``servers``.** A base URL is a deployment fact.

Writes ``tools/spec_extras.json`` in place, keys sorted, so a rerun with no drift
is a no-op and the diff a reviewer sees is only ever the changed entries.

Exit codes: 0 = nothing to fix or all fixed, 1 = drift left that needs a human,
2 = app import failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PLACEHOLDER = "No description provided yet."
MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You write one-sentence group descriptions for an HTTP API reference.

Each input is a tag from cognee's OpenAPI spec plus the endpoints carrying it.
Write a description of what that group of endpoints is for.

Rules:
- One sentence, ending in a period. Aim for 8-16 words.
- Describe what the endpoints do, grounded only in the endpoints given. Never
  invent capabilities that are not evidenced by the paths and summaries.
- Start with a noun phrase, matching the existing house style: "Dataset
  management endpoints for listing, creating, and deleting datasets.",
  "Search endpoints for querying the knowledge graph."
- No marketing language, no "This group contains", no restating the tag name
  alone ("Slack endpoints." is useless — say what they do).
- If the endpoints are internal or diagnostic, say so plainly.
"""


def load_app_schema() -> dict:
    os.environ.setdefault("ENV", "dev")
    from cognee.api.client import app  # pylint: disable=import-outside-toplevel

    return app.openapi()


def tag_usage(spec: dict) -> dict[str, list[dict]]:
    """tag -> the operations carrying it, with just enough context to describe it."""
    usage: dict[str, list[dict]] = {}
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags") or [
                "health" if path == "/" or path.startswith("/health") else "untagged"
            ]
            description = (operation.get("description") or "").strip()
            for tag in tags:
                usage.setdefault(tag, []).append(
                    {
                        "method": method.upper(),
                        "path": path,
                        "summary": (operation.get("summary") or "").strip(),
                        # First paragraph only: full docstrings run to hundreds of
                        # lines and the tag blurb only needs the gist.
                        "description": description.split("\n\n")[0][:400],
                    }
                )
    return usage


def existing_routes(spec: dict) -> set[str]:
    return {
        f"{method.upper()} {path}"
        for path, methods in spec.get("paths", {}).items()
        for method, operation in methods.items()
        if isinstance(operation, dict)
    }


def generate_descriptions(pending: dict[str, list[dict]]) -> dict[str, str]:
    """Ask Claude for one blurb per tag. Raises if the call fails or is refused."""
    import anthropic

    items = [
        {"tag": tag, "endpoints": endpoints[:12]} for tag, endpoints in sorted(pending.items())
    ]
    schema = {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["tag", "description"],
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
        max_tokens=8000,
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
                    "Write a group description for each of these API tags. Return one "
                    "entry per input tag.\n\n" + json.dumps(items, indent=2)
                ),
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (stop_reason=refusal)")

    text = next(block.text for block in response.content if block.type == "text")
    return {
        entry["tag"]: entry["description"].strip() for entry in json.loads(text)["descriptions"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Fill placeholder blurbs with Claude (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    from sync_release_docs import EXTRAS_PATH, load_extras  # noqa: E402

    try:
        spec = load_app_schema()
    except Exception as exc:
        print(f"Failed to import cognee API app: {exc}", file=sys.stderr)
        return 2

    extras = load_extras()
    tag_descriptions = dict(extras["tag_descriptions"])
    usage = tag_usage(spec)
    used_tags = set(usage)

    changed: list[str] = []

    # Mechanical fix 1: a blurb for a tag nothing uses is dead weight.
    for tag in sorted(tag_descriptions.keys() - used_tags):
        del tag_descriptions[tag]
        changed.append(f"removed dead tag description: {tag}")

    # Mechanical fix 2: give every used tag an entry, so the gap is visible in
    # the data file and fillable in one pass.
    for tag in sorted(used_tags - tag_descriptions.keys()):
        tag_descriptions[tag] = PLACEHOLDER
        changed.append(f"added placeholder for tag: {tag}")

    # Claude fix: turn the placeholders into real prose.
    placeholders = {tag for tag, text in tag_descriptions.items() if text == PLACEHOLDER}
    if placeholders and args.describe:
        pending = {tag: usage[tag] for tag in sorted(placeholders)}
        written = generate_descriptions(pending)
        for tag in sorted(placeholders):
            description = written.get(tag, "").strip()
            if not description:
                print(f"WARNING: Claude returned no description for tag {tag!r}", file=sys.stderr)
                continue
            tag_descriptions[tag] = description
            changed.append(f"described tag: {tag}")
        placeholders = {tag for tag, t in tag_descriptions.items() if t == PLACEHOLDER}

    extras["tag_descriptions"] = dict(sorted(tag_descriptions.items()))

    if changed:
        with EXTRAS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(extras, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Updated {EXTRAS_PATH.name}:")
        for line in changed:
            print(f"    {line}")
    else:
        print("No spec extras fixes needed.")

    # Everything below is reported, not fixed — see the module docstring.
    needs_human = False

    orphaned = sorted(extras["request_examples"].keys() - existing_routes(spec))
    if orphaned:
        needs_human = True
        print("\nNeeds a human — request examples pinned to routes the API no longer exposes:")
        for route in orphaned:
            print(f"    {route}")
        print(f"    Decide which route each sample belongs to now, in {EXTRAS_PATH.name}.")

    if placeholders:
        needs_human = True
        print("\nNeeds a human — tags still holding a placeholder blurb:")
        for tag in sorted(placeholders):
            print(f"    {tag}")
        print(
            "    Rerun with --describe and ANTHROPIC_API_KEY set, or write them by hand."
            if not args.describe
            else "    Claude did not return a description for these."
        )

    return 1 if needs_human else 0


if __name__ == "__main__":
    raise SystemExit(main())
