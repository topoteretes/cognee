#!/usr/bin/env python3
"""Prepare source and documentation candidates for automated docs edits."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


MAX_SOURCE_FILES_TO_INSPECT = 8
MAX_DOC_FILES_TO_CONSIDER = 5

IGNORED_KEYWORD_PARTS = {
    "cognee",
    "__init__.py",
    "models",
    "operations",
    "modules",
    "tasks",
    "routers",
    "shared",
    "infrastructure",
}

IGNORED_SOURCE_PARTS = {
    ".github",
    ".venv",
    "__pycache__",
    "node_modules",
    "tests",
    "test",
    "testing",
}

IGNORED_SOURCE_SUFFIXES = {
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".pyc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare docs edit scope candidates")
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/repo form")
    parser.add_argument("--pr-number", required=True, type=int, help="Pull request number to inspect")
    parser.add_argument("--docs-root", default="docs-repo", type=Path)
    parser.add_argument("--notes-json", required=True, type=Path)
    parser.add_argument("--assessment-json", required=True, type=Path)
    parser.add_argument("--scope-json-output", required=True, type=Path)
    parser.add_argument("--scope-markdown-output", required=True, type=Path)
    return parser.parse_args()


def github_api_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "prepare_docs_edit_scope.py",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def get_pr_changed_files(repo: str, pr_number: int) -> list[str]:
    changed_files: list[str] = []
    for page in range(1, 11):
        payload = github_api_json(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
            f"?per_page=100&page={page}"
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected GitHub API response for PR #{pr_number} files.")
        if not payload:
            break

        for item in payload:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                changed_files.append(filename)

        if len(payload) < 100:
            break

    return changed_files


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return payload


def collect_changed_files(args: argparse.Namespace) -> list[str]:
    return get_pr_changed_files(args.repo, args.pr_number)


def collect_keywords(changed_files: list[str]) -> list[str]:
    keywords: list[str] = []
    for changed_file in changed_files:
        path = Path(changed_file)
        stem = path.stem.lower()
        if len(stem) >= 3 and stem not in IGNORED_KEYWORD_PARTS:
            keywords.append(stem)
        for part in path.parts:
            token = part.lower()
            if len(token) >= 4 and token not in IGNORED_KEYWORD_PARTS:
                keywords.append(token)

    unique_keywords: list[str] = []
    for keyword in keywords:
        if keyword not in unique_keywords:
            unique_keywords.append(keyword)
    return unique_keywords


def get_candidate_doc_files(docs_root: Path, keywords: list[str]) -> list[str]:
    doc_files = sorted(
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}
    )

    scored: list[tuple[int, str]] = []
    for doc_file in doc_files:
        haystack = doc_file.lower()
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > 0:
            scored.append((score, doc_file))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_file for _, doc_file in scored[:5]]


def is_ignored_source_file(changed_file: str) -> bool:
    path = Path(changed_file)
    parts = {part.lower() for part in path.parts}
    if parts & IGNORED_SOURCE_PARTS:
        return True
    return path.suffix.lower() in IGNORED_SOURCE_SUFFIXES


def score_source_file(changed_file: str, notes: dict[str, Any], assessment: dict[str, Any]) -> int:
    path = Path(changed_file)
    lower_path = changed_file.lower()
    score = 0

    if lower_path.startswith("cognee/api/"):
        score += 60
    if lower_path.startswith("cognee/cli/"):
        score += 55
    if lower_path.startswith("examples/"):
        score += 50
    if "config" in lower_path or "settings" in lower_path:
        score += 45
    if "/routers/" in lower_path or lower_path.endswith("router.py"):
        score += 35
    if "/models/" in lower_path:
        score += 30
    if lower_path.startswith("cognee/modules/"):
        score += 25
    if lower_path.startswith("cognee/infrastructure/"):
        score += 20
    if path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx"}:
        score += 10

    evidence_text = " ".join(
        str(value)
        for value in [
            notes.get("summary", ""),
            notes.get("user_impact", ""),
            " ".join(notes.get("documentation_signals", []) or []),
            assessment.get("reason", ""),
            " ".join(assessment.get("candidate_areas", []) or []),
            " ".join(assessment.get("recommended_next_steps", []) or []),
        ]
    ).lower()
    for keyword in collect_keywords([changed_file]):
        if keyword in evidence_text:
            score += 5

    return score


def get_source_files_to_inspect(
    changed_files: list[str],
    notes: dict[str, Any],
    assessment: dict[str, Any],
) -> list[str]:
    scored = [
        (score_source_file(changed_file, notes, assessment), changed_file)
        for changed_file in changed_files
        if not is_ignored_source_file(changed_file)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [changed_file for _, changed_file in scored[:MAX_SOURCE_FILES_TO_INSPECT]]


def describe_source_file(changed_file: str) -> str:
    lower_path = changed_file.lower()
    if lower_path.startswith("cognee/api/"):
        return "public API surface"
    if lower_path.startswith("cognee/cli/"):
        return "CLI surface"
    if lower_path.startswith("examples/"):
        return "example or usage pattern"
    if "config" in lower_path or "settings" in lower_path:
        return "configuration/defaults surface"
    if lower_path.startswith("cognee/infrastructure/"):
        return "provider or adapter behavior"
    if lower_path.startswith("cognee/modules/"):
        return "module behavior or extension semantics"
    return "possible source evidence"


def build_scope_payload(
    pr_number: int,
    changed_files: list[str],
    candidate_doc_files: list[str],
    notes: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    source_files_to_inspect = get_source_files_to_inspect(changed_files, notes, assessment)
    ignored_files = [
        changed_file for changed_file in changed_files if changed_file not in source_files_to_inspect
    ]
    return {
        "pr_number": pr_number,
        "notes_summary": notes.get("summary", ""),
        "notes_user_impact": notes.get("user_impact", ""),
        "documentation_signals": notes.get("documentation_signals", []) or [],
        "needs_documentation_update": bool(assessment.get("needs_documentation_update")),
        "assessment_reason": assessment.get("reason", ""),
        "assessment_candidate_areas": assessment.get("candidate_areas", []) or [],
        "assessment_recommended_next_steps": assessment.get("recommended_next_steps", []) or [],
        "changed_files_count": len(changed_files),
        "changed_files": changed_files,
        "source_files_to_inspect": source_files_to_inspect,
        "source_file_classifications": [
            {
                "path": changed_file,
                "classification": describe_source_file(changed_file),
            }
            for changed_file in source_files_to_inspect
        ],
        "candidate_doc_files": candidate_doc_files[:MAX_DOC_FILES_TO_CONSIDER],
        "out_of_scope_files": ignored_files,
        "limits": {
            "max_source_files_to_inspect": MAX_SOURCE_FILES_TO_INSPECT,
            "max_doc_files_to_consider": MAX_DOC_FILES_TO_CONSIDER,
        },
    }


def format_markdown_scope(payload: dict[str, Any]) -> str:
    lines = [
        "# Prepared Documentation Edit Scope",
        "",
        f"PR: #{payload['pr_number']}",
        f"Changed files: {payload['changed_files_count']}",
        "",
        "## Assessment",
        "",
        f"Needs documentation update: `{str(payload['needs_documentation_update']).lower()}`",
        "",
        str(payload["assessment_reason"]).strip() or "No assessment reason provided.",
        "",
        "## Branch Notes Summary",
        "",
        str(payload["notes_summary"]).strip() or "No branch summary provided.",
        "",
        "## User Impact",
        "",
        str(payload["notes_user_impact"]).strip() or "No user impact provided.",
        "",
        "## Source Files To Inspect",
        "",
    ]
    source_classifications = payload["source_file_classifications"]
    if source_classifications:
        for item in source_classifications:
            lines.append(f"- `{item['path']}`: {item['classification']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Candidate Documentation Files", ""])
    candidate_doc_files = payload["candidate_doc_files"]
    if candidate_doc_files:
        lines.extend([f"- `{doc_file}`" for doc_file in candidate_doc_files])
    else:
        lines.append("- None")

    lines.extend(["", "## Documentation Signals", ""])
    documentation_signals = payload["documentation_signals"]
    if documentation_signals:
        lines.extend([f"- {signal}" for signal in documentation_signals])
    else:
        lines.append("- None")

    lines.extend(["", "## Candidate Areas", ""])
    candidate_areas = payload["assessment_candidate_areas"]
    if candidate_areas:
        lines.extend([f"- {area}" for area in candidate_areas])
    else:
        lines.append("- None")

    lines.extend(["", "## Recommended Next Steps", ""])
    next_steps = payload["assessment_recommended_next_steps"]
    if next_steps:
        lines.extend([f"- {step}" for step in next_steps])
    else:
        lines.append("- None")

    lines.extend(["", "## Out Of Scope Files", ""])
    out_of_scope_files = payload["out_of_scope_files"]
    if out_of_scope_files:
        lines.extend([f"- `{changed_file}`" for changed_file in out_of_scope_files])
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def write_scope_files(payload: dict[str, Any], json_output: Path, markdown_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(format_markdown_scope(payload), encoding="utf-8")


def write_multiline_output(name: str, lines: list[str]) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{name}<<EOF\n")
        fh.write("\n".join(lines) + "\n")
        fh.write("EOF\n")


def main() -> None:
    args = parse_args()
    notes = read_json(args.notes_json)
    assessment = read_json(args.assessment_json)
    changed_files = collect_changed_files(args)
    keywords = collect_keywords(changed_files)
    candidate_doc_files = get_candidate_doc_files(args.docs_root, keywords)
    scope_payload = build_scope_payload(
        args.pr_number,
        changed_files,
        candidate_doc_files,
        notes,
        assessment,
    )
    write_scope_files(scope_payload, args.scope_json_output, args.scope_markdown_output)

    write_multiline_output("changed_source_files", changed_files)
    write_multiline_output("candidate_doc_files", candidate_doc_files)
    write_multiline_output("source_files_to_inspect", scope_payload["source_files_to_inspect"])
    write_multiline_output("docs_files_to_consider", scope_payload["candidate_doc_files"])
    write_multiline_output("out_of_scope_files", scope_payload["out_of_scope_files"])


if __name__ == "__main__":
    main()
