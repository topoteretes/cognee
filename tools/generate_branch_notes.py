#!/usr/bin/env python3
"""
Generate notes for a single merged branch.

Matches the LLM integration style used by tools/generate_release_notes.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

try:
    from tools.merge_branch_diff import (
        get_branch_changed_files,
        get_branch_commit_subjects,
        get_branch_diff_stat,
    )
except ModuleNotFoundError:
    from merge_branch_diff import (
        get_branch_changed_files,
        get_branch_commit_subjects,
        get_branch_diff_stat,
    )


def collect_branch_payload(
    first_parent: str, second_parent: str, branch_name: str, merge_sha: str
) -> dict[str, Any]:
    changed_files = get_branch_changed_files(first_parent, second_parent)
    commit_subjects = get_branch_commit_subjects(first_parent, second_parent)
    diff_stat = get_branch_diff_stat(first_parent, second_parent)
    return {
        "branch_name": branch_name,
        "merge_sha": merge_sha,
        "first_parent": first_parent,
        "second_parent": second_parent,
        "changed_files": changed_files,
        "commit_subjects": commit_subjects,
        "diff_stat": diff_stat,
    }


async def generate_notes_with_llm(payload: dict[str, Any]) -> Any:
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(f"Required dependencies not available: {exc}") from exc

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")

    class BranchNotes(BaseModel):
        title: str = Field(description="Title for the branch notes")
        summary: str = Field(description="Summary of the merged branch")
        highlights: list[str] = Field(description="Top highlights from this branch")
        user_impact: str = Field(description="Likely user-facing impact")
        notable_files: list[str] = Field(description="Most relevant changed files")
        documentation_signals: list[str] = Field(
            description="What documentation areas may be affected"
        )

    system_prompt = """You are a technical writer creating notes for a single merged branch in Cognee.

Focus on user-facing changes, APIs, integrations, setup, and operational impact.
Keep the output concise and concrete.
"""

    user_prompt = (
        f"Generate notes for this merged branch.\n\nBranch data:\n{json.dumps(payload, indent=2)}\n"
    )

    try:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=BranchNotes,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc


def format_markdown(notes: Any, payload: dict[str, Any]) -> str:
    def get(name: str, default=""):
        return getattr(
            notes, name, notes.get(name, default) if isinstance(notes, dict) else default
        )

    highlights = get("highlights", [])
    notable_files = get("notable_files", [])
    documentation_signals = get("documentation_signals", [])
    summary = str(get("summary", "")).strip()
    branch_summary_prefix = f"Merged branch `{payload['branch_name']}`:"
    if summary:
        if payload["branch_name"].lower() not in summary.lower():
            summary = f"{branch_summary_prefix} {summary}"
    else:
        summary = branch_summary_prefix

    lines = [
        f"# {get('title', 'Branch notes')}",
        "",
        f"**Branch:** {payload['branch_name']}",
        f"**Merge SHA:** {payload['merge_sha']}",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## User Impact",
        "",
        get("user_impact", ""),
        "",
    ]
    if highlights:
        lines.extend(["## Highlights", ""])
        lines.extend([f"- {item}" for item in highlights])
        lines.append("")
    if notable_files:
        lines.extend(["## Notable Files", ""])
        lines.extend([f"- {item}" for item in notable_files])
        lines.append("")
    if documentation_signals:
        lines.extend(["## Documentation Signals", ""])
        lines.extend([f"- {item}" for item in documentation_signals])
        lines.append("")
    lines.extend(["## Raw Commit Subjects", ""])
    lines.extend(payload["commit_subjects"] or ["No non-merge commits found"])
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate notes for a single merged branch")
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--merge-sha", required=True)
    parser.add_argument("--first-parent", required=True)
    parser.add_argument("--second-parent", required=True)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


async def main():
    args = parse_args()
    payload = collect_branch_payload(
        args.first_parent, args.second_parent, args.branch_name, args.merge_sha
    )
    notes = await generate_notes_with_llm(payload)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(notes.model_dump() if hasattr(notes, "model_dump") else notes, indent=2) + "\n"
    )
    args.markdown_output.write_text(format_markdown(notes, payload))
    print(args.markdown_output.read_text())


if __name__ == "__main__":
    asyncio.run(main())
