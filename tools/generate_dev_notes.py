#!/usr/bin/env python3
"""
Generate daily dev notes from merge commits on a target branch.

Matches the LLM integration style used by tools/generate_release_notes.py:
- uses litellm + instructor directly
- reads LLM_API_KEY / LLM_MODEL from the environment
- falls back to deterministic notes when LLM is unavailable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def run_git_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"Error running git command: {' '.join(command)}", file=sys.stderr)
        print(exc.stderr, file=sys.stderr)
        raise SystemExit(1) from exc


@dataclass
class MergeRecord:
    branch_name: str
    merge_sha: str
    short_sha: str
    subject: str
    first_parent: str
    second_parent: str
    changed_files: list[str]
    commit_subjects: list[str]
    diff_stat: str


def infer_branch_name(subject: str, body: str, merge_sha: str) -> str:
    pr_match = re.search(r"Merge pull request #\d+ from [^/]+/(.+)", subject)
    if pr_match:
        return pr_match.group(1).strip()

    branch_match = re.search(r"Merge branch '([^']+)'", subject)
    if branch_match:
        return branch_match.group(1).strip()

    body_match = re.search(r"from [^/]+/(.+)", body)
    if body_match:
        return body_match.group(1).strip()

    return f"merge-{merge_sha[:7]}"


def collect_merge_records(target_branch: str, start_date: str, end_date: str) -> list[MergeRecord]:
    raw_log = run_git_command(
        [
            "git",
            "log",
            target_branch,
            "--merges",
            "--first-parent",
            f"--since={start_date}",
            f"--until={end_date}",
            "--pretty=format:%H%x1f%P%x1f%s%x1f%b%x1e",
        ]
    )

    records: list[MergeRecord] = []
    for record in raw_log.split("\x1e"):
        if not record.strip():
            continue
        merge_sha, parents, subject, body = record.strip().split("\x1f", 3)
        parent_parts = parents.split()
        if len(parent_parts) < 2:
            continue
        first_parent, second_parent = parent_parts[0], parent_parts[1]
        branch_name = infer_branch_name(subject, body, merge_sha)

        changed_files = [
            line
            for line in run_git_command(
                ["git", "diff", "--name-only", first_parent, second_parent]
            ).splitlines()
            if line.strip()
        ]
        commit_subjects = [
            line
            for line in run_git_command(
                [
                    "git",
                    "log",
                    "--no-merges",
                    "--pretty=format:- %s (%h)",
                    f"{first_parent}..{second_parent}",
                ]
            ).splitlines()
            if line.strip()
        ]
        diff_stat = run_git_command(["git", "diff", "--stat", first_parent, second_parent])

        records.append(
            MergeRecord(
                branch_name=branch_name,
                merge_sha=merge_sha,
                short_sha=merge_sha[:7],
                subject=subject,
                first_parent=first_parent,
                second_parent=second_parent,
                changed_files=changed_files,
                commit_subjects=commit_subjects,
                diff_stat=diff_stat,
            )
        )

    return records


async def generate_notes_with_llm(
    merge_records: list[MergeRecord], target_branch: str, start_date: str, end_date: str
) -> Any:
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        print(f"Error: Required dependencies not available: {exc}", file=sys.stderr)
        return None

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        print("Warning: LLM_API_KEY not set, skipping LLM generation.", file=sys.stderr)
        return None

    class BranchSummary(BaseModel):
        branch_name: str = Field(description="Merged branch name")
        summary: str = Field(description="Short summary of the branch changes")
        user_impact: str = Field(description="How users are likely affected")
        notable_files: list[str] = Field(description="Most relevant changed files")

    class DevNotes(BaseModel):
        title: str = Field(description="Human-friendly title for the daily dev notes")
        summary: str = Field(description="High-level summary of the day's merged changes")
        highlights: list[str] = Field(description="3-6 daily highlights")
        merged_branches: list[BranchSummary] = Field(description="Summary per merged branch")
        documentation_signals: list[str] = Field(
            description="Signals suggesting what areas may need docs attention"
        )

    merge_payload = [
        {
            "branch_name": record.branch_name,
            "merge_sha": record.merge_sha,
            "subject": record.subject,
            "changed_files": record.changed_files[:30],
            "commit_subjects": record.commit_subjects[:20],
            "diff_stat": record.diff_stat,
        }
        for record in merge_records
    ]

    system_prompt = """You are a technical writer creating daily dev notes for Cognee.

Analyze merged branches on dev and produce concise, user-focused notes.

Guidelines:
- Focus on user-facing changes, APIs, integrations, setup, and operational impact
- Group related changes clearly
- Keep summaries concrete and useful
- Highlight documentation-relevant signals when they appear
- Avoid unnecessary file-level detail unless it is especially relevant
"""

    user_prompt = f"""Generate daily dev notes for Cognee.

Target branch: {target_branch}
Time window start: {start_date}
Time window end: {end_date}

Merged branch data:
{json.dumps(merge_payload, indent=2)}
"""

    try:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=DevNotes,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        print(f"Warning: LLM generation failed: {exc}", file=sys.stderr)
        return None


def generate_fallback_notes(
    merge_records: list[MergeRecord], target_branch: str, start_date: str, end_date: str
) -> dict[str, Any]:
    return {
        "title": f"Daily dev notes for {target_branch}",
        "summary": (
            f"{len(merge_records)} branches were merged into {target_branch} "
            f"between {start_date} and {end_date}."
        ),
        "highlights": [record.subject for record in merge_records[:5]],
        "merged_branches": [
            {
                "branch_name": record.branch_name,
                "summary": record.subject,
                "user_impact": "Needs review based on merged branch changes.",
                "notable_files": record.changed_files[:10],
            }
            for record in merge_records
        ],
        "documentation_signals": [
            "Review API, CLI, integration, setup, and operational changes mentioned in merged branches."
        ],
    }


def format_notes_markdown(
    notes: Any, merge_records: list[MergeRecord], target_branch: str, start_date: str, end_date: str
) -> str:
    lines = [
        f"# {notes.title if hasattr(notes, 'title') else notes.get('title', 'Daily dev notes')}",
        "",
        f"**Branch:** {target_branch}",
        f"**Time window (UTC):** {start_date} to {end_date}",
        "",
        "## Summary",
        "",
        notes.summary if hasattr(notes, "summary") else notes.get("summary", ""),
        "",
    ]

    highlights = notes.highlights if hasattr(notes, "highlights") else notes.get("highlights", [])
    if highlights:
        lines.extend(["## Highlights", ""])
        lines.extend([f"- {item}" for item in highlights])
        lines.append("")

    merged_branches = (
        notes.merged_branches if hasattr(notes, "merged_branches") else notes.get("merged_branches", [])
    )
    if merged_branches:
        lines.extend(["## Merged Branches", ""])
        for item in merged_branches:
            branch_name = item.branch_name if hasattr(item, "branch_name") else item.get("branch_name", "Unknown branch")
            summary = item.summary if hasattr(item, "summary") else item.get("summary", "")
            user_impact = item.user_impact if hasattr(item, "user_impact") else item.get("user_impact", "")
            notable_files = item.notable_files if hasattr(item, "notable_files") else item.get("notable_files", [])
            lines.append(f"### {branch_name}")
            lines.append("")
            lines.append(summary)
            lines.append("")
            lines.append(f"- User impact: {user_impact}")
            if notable_files:
                lines.append("- Notable files:")
                lines.extend([f"  - {path}" for path in notable_files])
            lines.append("")

    documentation_signals = (
        notes.documentation_signals
        if hasattr(notes, "documentation_signals")
        else notes.get("documentation_signals", [])
    )
    if documentation_signals:
        lines.extend(["## Documentation Signals", ""])
        lines.extend([f"- {item}" for item in documentation_signals])
        lines.append("")

    lines.extend(["## Raw Merge List", ""])
    for record in merge_records:
        lines.append(f"- {record.branch_name} ({record.short_sha}): {record.subject}")
    lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate current-day dev notes from merged branches")
    parser.add_argument("--target-branch", default="origin/dev")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


async def main():
    args = parse_args()
    merge_records = collect_merge_records(args.target_branch, args.start_date, args.end_date)

    if merge_records:
        notes = await generate_notes_with_llm(
            merge_records, args.target_branch, args.start_date, args.end_date
        )
        if notes is None:
            notes = generate_fallback_notes(
                merge_records, args.target_branch, args.start_date, args.end_date
            )
    else:
        notes = {
            "title": f"Daily dev notes for {args.target_branch}",
            "summary": (
                f"No branches were merged into {args.target_branch} "
                f"between {args.start_date} and {args.end_date}."
            ),
            "highlights": [],
            "merged_branches": [],
            "documentation_signals": [],
        }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            notes.model_dump() if hasattr(notes, "model_dump") else notes,
            indent=2,
        )
        + "\n"
    )
    args.markdown_output.write_text(
        format_notes_markdown(notes, merge_records, args.target_branch, args.start_date, args.end_date)
    )

    print(args.markdown_output.read_text())


if __name__ == "__main__":
    asyncio.run(main())
