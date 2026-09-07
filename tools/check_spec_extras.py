#!/usr/bin/env python3
"""
Check the docs extras in ``tools/spec_extras.json`` against the live FastAPI app.

``enhance_spec`` in ``sync_release_docs.py`` adds what FastAPI does not emit but
the published API reference needs: ``servers``, tag blurbs and request examples.
Two of those are keyed off things that move — the tags routes carry, the routes
themselves — and nothing else in the repo watches them. ``router_docstring_sync``
does not: it only ever stages ``cognee``, and a tag blurb is not derivable from a
docstring or Pydantic metadata anyway.

This is the detector half; ``fix_spec_extras.py`` does the fixing. It reads the
extras through ``sync_release_docs`` so they stay single-source, but derives what
it compares them against — the tags in use, the routes that exist — straight from
the app's own schema, so a bug in ``enhance_spec`` cannot hide from its own check.

What it reports:

- a route tag with no blurb, which renders as a bare sidebar group
- a blurb for a tag no route uses, left behind by a rename
- a placeholder blurb the describe stage never filled
- a request example pinned to a route the API no longer exposes, so the sample
  silently stopped reaching the reference
- a malformed ``servers`` entry, which would leave the docs playground with no
  usable base URL

Exit codes mirror ``check_router_docstrings.py``:
0 = no issues, 1 = issues found, 2 = app import failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_app_schema() -> dict:
    """The app's own OpenAPI schema, before any of the extras are applied."""
    os.environ.setdefault("ENV", "dev")
    from cognee.api.client import app  # pylint: disable=import-outside-toplevel

    return app.openapi()


def route_facts(spec: dict) -> tuple[set[str], set[str]]:
    """(tags that will appear in the spec, "METHOD /path" for every operation).

    Includes the ``health``/``untagged`` names ``enhance_spec``'s fallback invents
    for operations that declare no tags — those are real sidebar groups even
    though no route names them.
    """
    tags: set[str] = set()
    routes: set[str] = set()
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            routes.add(f"{method.upper()} {path}")
            tags.update(
                operation.get("tags")
                or ["health" if path == "/" or path.startswith("/health") else "untagged"]
            )
    return tags, routes


def check_servers(servers: list) -> list[str]:
    """Structural problems that would break the docs playground."""
    if not servers:
        return ["servers is empty — the docs playground would have no base URL"]
    problems = []
    for entry in servers:
        if not isinstance(entry, dict):
            problems.append(f"servers entry is not an object: {entry!r}")
            continue
        url = entry.get("url", "")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            problems.append(f"servers entry has no absolute http(s) url: {url!r}")
        if not entry.get("description"):
            problems.append(f"servers entry {url!r} has no description")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print the report but always exit 0",
    )
    args = parser.parse_args()

    from fix_spec_extras import PLACEHOLDER  # pylint: disable=import-outside-toplevel
    from sync_release_docs import load_extras  # pylint: disable=import-outside-toplevel

    # Read the file rather than using sync_release_docs' module-level constants:
    # those are bound once at import, so a checker running in the same process as
    # the fixer would grade the pre-fix data and wrongly report drift.
    extras = load_extras()
    servers = extras["servers"]
    tag_descriptions = extras["tag_descriptions"]
    request_examples = extras["request_examples"]

    try:
        spec = load_app_schema()
    except Exception as exc:
        print(f"Failed to import cognee API app: {exc}", file=sys.stderr)
        return 2

    used_tags, routes = route_facts(spec)

    undescribed = sorted(used_tags - tag_descriptions.keys())
    stale = sorted(tag_descriptions.keys() - used_tags)
    placeholders = sorted(tag for tag, text in tag_descriptions.items() if text == PLACEHOLDER)
    orphaned = sorted(request_examples.keys() - routes)
    server_problems = check_servers(servers)

    for tag in undescribed:
        print(f"tag has no description:              {tag}")
    for tag in stale:
        print(f"description for unused tag:          {tag}")
    for tag in placeholders:
        print(f"description is still a placeholder:  {tag}")
    for route in orphaned:
        print(f"example pinned to a missing route:   {route}")
    for problem in server_problems:
        print(f"servers: {problem}")

    issues = (
        len(undescribed) + len(stale) + len(placeholders) + len(orphaned) + len(server_problems)
    )
    described = len(used_tags) - len(undescribed) - len(placeholders)
    print(
        f"\nChecked {len(used_tags)} route tags ({described} described), "
        f"{len(request_examples)} request examples and {len(servers)} servers "
        f"against {len(routes)} routes: {issues} issue(s)."
    )

    if issues and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
