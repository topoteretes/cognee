#!/usr/bin/env python3
"""
Sync release docs artifacts into a checked-out cognee-docs repository.

This script is intended for CI usage from the core `cognee` repository:
1) Generate OpenAPI spec from the current codebase.
2) Enhance it with the docs-facing extras (servers, tag descriptions, request
   examples, security schemes) — see `enhance_spec`.
3) Copy spec to docs repo.
4) Prepend changelog entry for the release.

This script is the *single* generator of `cognee_openapi_spec.json`. The docs
repo used to regenerate the same file on a Wednesday cron
(`.github/scripts/generate-api-docs.sh`) and add step 2 itself, so every
release stripped what the cron had added and every cron put it back. That
script is gone; the docs repo now only fetches this output and opens a PR.
Anything the published API reference needs that FastAPI does not emit belongs
in `enhance_spec` below.

Note that the extras live here rather than in the FastAPI app, so a running
server's `/openapi.json` does not carry them — only the published spec does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_CHANGELOG_TEXT = """---
title: "Changelog"
description: "Recent Cognee releases"
icon: "scroll-text"
---

Cognee releases with highlights and links to the full release notes on GitHub.
"""


# Docs-facing extras. FastAPI does not emit any of this, and Mintlify reads all
# of it from the spec: `servers` drives the interactive playground's base URL,
# `tag_descriptions` supplies the sidebar group blurbs, and `request_examples`
# gives each endpoint a runnable sample body.
#
# They live in a JSON data file rather than as literals here because two of the
# three are machine-maintained: the spec extras sync workflow rewrites them via
# tools/fix_spec_extras.py. Editing JSON is a load-mutate-dump, with no source
# splicing and nothing for the formatter to disagree with.
EXTRAS_PATH = Path(__file__).resolve().parent / "spec_extras.json"


def load_extras() -> dict:
    """The extras data file. Fails loudly — a silent default would publish a
    spec missing its servers and blurbs, which is worse than a failed release."""
    try:
        with EXTRAS_PATH.open(encoding="utf-8") as handle:
            extras = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read docs extras from {EXTRAS_PATH}: {exc}") from exc

    missing = {"servers", "tag_descriptions", "request_examples"} - extras.keys()
    if missing:
        raise RuntimeError(f"{EXTRAS_PATH} is missing required key(s): {sorted(missing)}")
    return extras


_EXTRAS = load_extras()

# Base URLs for the docs playground. Not machine-maintained — a new hosted API
# URL is a human decision — but checked for shape by tools/check_spec_extras.py.
SERVERS = _EXTRAS["servers"]

# Sidebar blurbs, keyed by the tag as it appears on the route. A tag with no
# entry still groups correctly, it just renders without a description.
TAG_DESCRIPTIONS = _EXTRAS["tag_descriptions"]

# Sample request bodies, keyed by "METHOD /path" — the API contract, which is
# stable across handler renames. Keying by FastAPI's generated operationId
# instead would embed the function name, so a rename would silently orphan the
# example. A key whose path or method no longer exists is a real API change and
# `enhance_spec` raises rather than dropping the sample quietly.
REQUEST_EXAMPLES = _EXTRAS["request_examples"]


def _cookie_scheme() -> dict:
    """The cookie auth scheme, with the cookie name the app actually sets.

    Read from the transport rather than hardcoded: the docs cron used to publish
    ``fastapiusersauth`` (fastapi-users' default) while cognee's transport sets
    ``auth_token``, so the reference documented a cookie that never existed.
    """
    from cognee.modules.users.authentication.default import default_transport

    return {"type": "apiKey", "in": "cookie", "name": default_transport.cookie_name}


def enhance_spec(spec: dict) -> dict:
    """Add the docs-facing extras FastAPI does not generate. Mutates and returns spec.

    Assignment order matters: `servers` and `tags` are appended after the keys
    FastAPI produced, which is the key order the published spec already has.
    """
    spec["servers"] = SERVERS

    # All three transports are real: APIKeyHeader (X-Api-Key), BearerTransport,
    # and CookieTransport. The app declares the first two; the cookie one is
    # only ever reflected in the published spec, so it is added here.
    schemes = spec.setdefault("components", {}).setdefault("securitySchemes", {})
    schemes["ApiKeyAuth"] = {"type": "apiKey", "in": "header", "name": "X-Api-Key"}
    schemes["BearerAuth"] = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    schemes["CookieAuth"] = _cookie_scheme()

    operations = [
        operation
        for methods in spec.get("paths", {}).values()
        for operation in methods.values()
        if isinstance(operation, dict)
    ]

    # Tag anything untagged so it does not land in an unnamed sidebar group.
    for path, methods in spec.get("paths", {}).items():
        for operation in methods.values():
            if not isinstance(operation, dict) or operation.get("tags"):
                continue
            operation["tags"] = (
                ["health"] if path == "/" or path.startswith("/health") else ["untagged"]
            )

    used_tags = {tag for operation in operations for tag in operation.get("tags", [])}
    spec["tags"] = [
        {"name": tag, "description": TAG_DESCRIPTIONS.get(tag, "")} for tag in sorted(used_tags)
    ]

    # Neither of these is fatal — the reference still builds — but both mean the
    # sidebar quietly lost a blurb, which is exactly the drift this script exists
    # to stop. Nothing else watches TAG_DESCRIPTIONS, so say it out loud.
    if undescribed := sorted(used_tags - TAG_DESCRIPTIONS.keys()):
        print(
            f"WARNING: {len(undescribed)} tag(s) have no description in "
            f"TAG_DESCRIPTIONS: {', '.join(undescribed)}",
            file=sys.stderr,
        )
    if stale := sorted(TAG_DESCRIPTIONS.keys() - used_tags):
        print(
            f"WARNING: TAG_DESCRIPTIONS describes tag(s) no route uses: {', '.join(stale)}",
            file=sys.stderr,
        )

    # An example pinned to a path/method the API no longer exposes would vanish
    # from the reference without a trace. Fail the release sync instead.
    by_route = {
        f"{method.upper()} {path}": operation
        for path, methods in spec.get("paths", {}).items()
        for method, operation in methods.items()
        if isinstance(operation, dict)
    }
    if orphaned := sorted(REQUEST_EXAMPLES.keys() - by_route.keys()):
        raise RuntimeError(
            f"request_examples references route(s) absent from the spec: {', '.join(orphaned)}. "
            f"The path or method changed — update request_examples in {EXTRAS_PATH.name}."
        )
    for route, example in REQUEST_EXAMPLES.items():
        content = by_route[route].get("requestBody", {}).get("content", {})
        for media in content.values():
            media.setdefault("example", example)

    return spec


def generate_openapi_spec(output_path: Path) -> None:
    """
    Generate the enhanced OpenAPI schema from the cognee FastAPI app.

    The app's own schema plus the docs-facing extras from `enhance_spec` — this
    is the file the published API reference is built from.
    """
    try:
        # Avoid prod-only initialization behavior for CI schema generation.
        os.environ.setdefault("ENV", "dev")
        from cognee.api.client import app  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - runtime import environment specific
        raise RuntimeError(f"Failed to import cognee API app: {exc}") from exc

    spec = enhance_spec(app.openapi())
    output_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def read_release_body(path: Path) -> str:
    body = path.read_text(encoding="utf-8").strip()
    return body if body else "_No release notes provided._"


def format_release_date(published_at: str) -> str:
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except ValueError:
        return published_at


def build_changelog_entry(tag: str, release_url: str, release_date: str, release_body: str) -> str:
    return (
        f"## {tag}\n\n"
        f"**Released:** {release_date}  \n"
        f"**[View on GitHub]({release_url})**\n\n"
        f"{release_body}\n\n"
        "---\n"
    )


def split_frontmatter(content: str) -> tuple[str, str]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", content

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break

    if end_idx is None:
        return "", content

    frontmatter = "".join(lines[: end_idx + 1]).rstrip() + "\n\n"
    body = "".join(lines[end_idx + 1 :]).lstrip("\n")
    return frontmatter, body


def changelog_has_tag(content: str, tag: str) -> bool:
    pattern = rf"^##\s+{re.escape(tag)}\s*$"
    return re.search(pattern, content, flags=re.MULTILINE) is not None


def prepend_entry_to_changelog(existing: str, entry: str) -> str:
    frontmatter, body = split_frontmatter(existing)

    first_h2 = re.search(r"^##\s+", body, flags=re.MULTILINE)
    if first_h2:
        intro = body[: first_h2.start()].rstrip()
        existing_entries = body[first_h2.start() :].lstrip("\n")
    else:
        intro = body.rstrip()
        existing_entries = ""

    parts = []
    if intro:
        parts.append(intro)
    parts.append(entry.rstrip())
    if existing_entries:
        parts.append(existing_entries.rstrip())

    updated_body = "\n\n".join(parts).rstrip() + "\n"
    return frontmatter + updated_body


def copy_if_changed(source: Path, target: Path) -> bool:
    source_bytes = source.read_bytes()
    if target.exists() and target.read_bytes() == source_bytes:
        return False
    target.write_bytes(source_bytes)
    return True


def update_changelog_if_needed(
    changelog_path: Path, tag: str, release_url: str, published_at: str, release_body: str
) -> bool:
    existing = (
        changelog_path.read_text(encoding="utf-8")
        if changelog_path.exists()
        else DEFAULT_CHANGELOG_TEXT
    )

    if changelog_has_tag(existing, tag):
        return False

    release_date = format_release_date(published_at)
    entry = build_changelog_entry(tag, release_url, release_date, release_body)
    updated = prepend_entry_to_changelog(existing, entry)

    if updated == existing:
        return False

    changelog_path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync release docs artifacts into cognee-docs repo"
    )
    parser.add_argument(
        "--docs-repo", required=True, type=Path, help="Path to checked-out docs repo"
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.5.4")
    parser.add_argument("--release-url", required=True, help="GitHub release URL")
    parser.add_argument(
        "--published-at", required=True, help="Release publish timestamp (ISO 8601)"
    )
    parser.add_argument(
        "--release-body-file",
        required=True,
        type=Path,
        help="Path to file containing GitHub release body markdown",
    )
    parser.add_argument(
        "--openapi-output",
        default="cognee_openapi_spec.json",
        type=Path,
        help="Where to write generated OpenAPI spec in core repo checkout",
    )
    parser.add_argument(
        "--docs-openapi-file",
        default="cognee_openapi_spec.json",
        help="OpenAPI target file path relative to docs repo",
    )
    parser.add_argument(
        "--docs-changelog-file",
        default="changelog.mdx",
        help="Changelog target file path relative to docs repo",
    )
    parser.add_argument(
        "--skip-openapi-generation",
        action="store_true",
        help="Skip OpenAPI generation and only sync existing openapi-output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    docs_repo: Path = args.docs_repo
    if not docs_repo.exists():
        print(f"Docs repo path does not exist: {docs_repo}", file=sys.stderr)
        return 2

    if not args.skip_openapi_generation:
        generate_openapi_spec(args.openapi_output)

    if not args.openapi_output.exists():
        print(f"OpenAPI source file does not exist: {args.openapi_output}", file=sys.stderr)
        return 2

    release_body = read_release_body(args.release_body_file)
    docs_openapi_path = docs_repo / args.docs_openapi_file
    docs_changelog_path = docs_repo / args.docs_changelog_file

    openapi_changed = copy_if_changed(args.openapi_output, docs_openapi_path)
    changelog_changed = update_changelog_if_needed(
        docs_changelog_path,
        tag=args.tag,
        release_url=args.release_url,
        published_at=args.published_at,
        release_body=release_body,
    )

    print(f"openapi_changed={str(openapi_changed).lower()}")
    print(f"changelog_changed={str(changelog_changed).lower()}")
    print(f"changes_made={str(openapi_changed or changelog_changed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
