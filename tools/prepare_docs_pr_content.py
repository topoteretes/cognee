#!/usr/bin/env python3
"""Prepare pull request title, body, and changed-file outputs for docs drafts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare docs PR content")
    parser.add_argument("--notes-json", required=True, type=Path)
    parser.add_argument("--assessment-json", required=True, type=Path)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--short-sha", required=True)
    parser.add_argument("--default-pr-title", required=True)
    parser.add_argument("--docs-root", default="docs-repo")
    return parser.parse_args()


def write_multiline_output(name: str, lines: list[str]) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{name}<<EOF\n")
        fh.write("\n".join(lines) + "\n")
        fh.write("EOF\n")


def main() -> None:
    args = parse_args()
    notes = json.loads(args.notes_json.read_text())
    assessment = json.loads(args.assessment_json.read_text())
    summary = notes.get("summary", "").strip()
    highlights = notes.get("highlights", [])
    reason = assessment.get("reason", "")

    changed_files = [
        line.rstrip()
        for line in subprocess.check_output(
            ["git", "-C", args.docs_root, "status", "--short"],
            text=True,
        ).splitlines()
        if line.strip()
    ]
    normalized_files = [entry[3:] if len(entry) > 3 else entry for entry in changed_files]

    pr_body_lines = [
        "## Summary",
        "",
        f"Automated documentation draft for merged branch `{args.branch_name}` (`{args.short_sha}`).",
        "",
        summary
        or "This PR updates existing docs to reflect the branch's user-facing documentation impact based on the source diff and current docs structure.",
        "",
        "## Why This PR Exists",
        "",
        reason
        or "The merged branch appears to change behavior, configuration, API usage, or developer-facing semantics that are represented in the docs.",
        "",
        "## Source",
        "",
        f"- Branch: `{args.branch_name}`",
        f"- Merge short SHA: `{args.short_sha}`",
    ]
    if highlights:
        pr_body_lines.extend(["", "## Branch Highlights", ""])
        pr_body_lines.extend([f"- {item}" for item in highlights])
    if normalized_files:
        pr_body_lines.extend(["", "## Documentation Files Updated", ""])
        pr_body_lines.extend([f"- `{item}`" for item in normalized_files])

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"pr_title={args.default_pr_title}\n")
    write_multiline_output("pr_body", pr_body_lines)
    write_multiline_output("changed_files", normalized_files)


if __name__ == "__main__":
    main()
