#!/usr/bin/env python3
"""Prepare source and documentation candidates for automated docs edits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from tools.merge_branch_diff import get_branch_changed_files
except ModuleNotFoundError:
    from merge_branch_diff import get_branch_changed_files


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare docs edit scope candidates")
    parser.add_argument("--first-parent", required=True)
    parser.add_argument("--second-parent", required=True)
    parser.add_argument("--docs-root", default="docs-repo", type=Path)
    return parser.parse_args()


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


def write_multiline_output(name: str, lines: list[str]) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{name}<<EOF\n")
        fh.write("\n".join(lines) + "\n")
        fh.write("EOF\n")


def main() -> None:
    args = parse_args()
    changed_files = get_branch_changed_files(args.first_parent, args.second_parent)
    keywords = collect_keywords(changed_files)
    candidate_doc_files = get_candidate_doc_files(args.docs_root, keywords)

    write_multiline_output("changed_source_files", changed_files)
    write_multiline_output("candidate_doc_files", candidate_doc_files)


if __name__ == "__main__":
    main()
