#!/usr/bin/env python3
"""
Sync release docs artifacts into a checked-out cognee-docs repository.

This script is intended for CI usage from the core `cognee` repository:
1) Generate OpenAPI spec from the current codebase.
2) Copy spec to docs repo.
3) Prepend changelog entry for the release.
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


def generate_openapi_spec(output_path: Path) -> None:
    """
    Generate OpenAPI schema from cognee FastAPI app and write it to output_path.
    """
    try:
        # Avoid prod-only initialization behavior for CI schema generation.
        os.environ.setdefault("ENV", "dev")
        from cognee.api.client import app  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - runtime import environment specific
        raise RuntimeError(f"Failed to import cognee API app: {exc}") from exc

    spec = app.openapi()
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
