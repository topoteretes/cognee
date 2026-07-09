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


def read_markdown_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").rstrip().splitlines()


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
    branch_notes_dir = args.notes_json.parent

    generated_markdown_files = [
        ("branch_notes.md", branch_notes_dir / "branch_notes.md"),
        ("docs_assessment.md", branch_notes_dir / "docs_assessment.md"),
        ("docs_edit_scope.md", branch_notes_dir / "docs_edit_scope.md"),
        ("docs_scope_plan.md", branch_notes_dir / "docs_scope_plan.md"),
    ]

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

    generated_markdown = [
        (label, read_markdown_file(path))
        for label, path in generated_markdown_files
        if path.is_file()
    ]
    if generated_markdown:
        pr_body_lines.extend(["", "## Generated Planning Markdown", ""])
        pr_body_lines.append(
            "The workflow also generated the following planning files in `branch-dev-notes/`."
        )
        pr_body_lines.append(
            "Their contents are included below so the PR captures the Claude-produced planning artifacts."
        )
        for label, lines in generated_markdown:
            pr_body_lines.extend(["", f"### {label}", ""])
            pr_body_lines.extend(lines or ["(empty file)"])

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"pr_title={args.default_pr_title}\n")
    write_multiline_output("pr_body", pr_body_lines)
    write_multiline_output("changed_files", normalized_files)


if __name__ == "__main__":
    main()
