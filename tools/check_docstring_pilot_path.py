#!/usr/bin/env python3
"""
Check that a docstring-pilot target is one of the ten selected files.

RES-28 deliberately scopes the docstring automation to a fixed list of files
chosen by a human. The list lives in the markdown table of
``.claude/skills/docstring-author/pilot_files.md`` — the same file the
``docstring-author`` skill reads — so the workflow's allowlist and the agent's
instructions cannot drift apart. Any backticked path in a table row counts as selected.

Parsing the allowlist out of prose rather than keeping a second machine-readable
copy is the point: a duplicate list is a list that goes stale, and the failure
mode there is the automation editing a file nobody selected.

Rejection is reported to stdout and, when running under Actions, appended to the
step summary, because a dispatch with a bad ``path`` is a human typo and the
reason has to be visible without opening the job log.

Exit codes: 0 = path is selected, 1 = path is rejected, 2 = allowlist unreadable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import PurePosixPath

DEFAULT_ALLOWLIST = ".claude/skills/docstring-author/pilot_files.md"

# Only rows of a markdown table count, so the prose above the tables can mention
# paths (CLAUDE.md, the workflow itself) without silently widening the allowlist.
TABLE_ROW_RE = re.compile(r"^\s*\|")
BACKTICKED_RE = re.compile(r"`([^`]+)`")


def read_allowlist(allowlist_path: str) -> list[str]:
    """Selected files, in the order they appear in the allowlist's tables."""
    try:
        text = open(allowlist_path, encoding="utf-8").read()
    except OSError as error:
        print(f"Could not read the allowlist at {allowlist_path}: {error}", file=sys.stderr)
        raise SystemExit(2)

    selected: list[str] = []
    for line in text.splitlines():
        if not TABLE_ROW_RE.match(line):
            continue
        # The first backticked path in a row is the selected file; the reason
        # cell after it quotes symbols and other files as evidence, and must not
        # widen the allowlist.
        for candidate in BACKTICKED_RE.findall(line):
            if candidate.endswith(".py") and "/" in candidate:
                if candidate not in selected:
                    selected.append(candidate)
                break
    return selected


def normalize(path: str) -> str:
    """Repo-relative POSIX form, or "" for anything that escapes the repo.

    Dispatch inputs are free text. ``./cognee/x.py`` and ``cognee/x.py`` are the
    same file and both should pass; an absolute path or one climbing out with
    ``..`` should not resolve to a selected file by accident.
    """
    cleaned = path.strip().strip("`").replace("\\", "/")
    if not cleaned or cleaned.startswith("/"):
        return ""
    parts = [p for p in PurePosixPath(cleaned).parts if p not in (".",)]
    if any(p == ".." for p in parts):
        return ""
    return "/".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Candidate file path from the dispatch input")
    parser.add_argument(
        "--allowlist",
        default=DEFAULT_ALLOWLIST,
        help=f"Markdown file holding the selection table (default: {DEFAULT_ALLOWLIST})",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Directory the path is resolved against when checking the file exists",
    )
    args = parser.parse_args()

    selected = read_allowlist(args.allowlist)
    if len(selected) != 10:
        # The pilot is defined as ten files. A table that no longer yields ten
        # means the allowlist was edited or the parse broke, and either way the
        # automation should stop rather than run against a list it misread.
        return reject(
            f"The allowlist at `{args.allowlist}` yielded {len(selected)} files, not the "
            f"ten the pilot is defined as. Parsed: {selected or 'nothing'}.",
            selected,
        )

    candidate = normalize(args.path)
    if not candidate:
        return reject(
            f"`{args.path}` is not a repo-relative path. Pass one of the selected files "
            "exactly as it appears below.",
            selected,
        )

    if candidate not in selected:
        return reject(
            f"`{candidate}` is not one of the ten files selected for the docstring pilot, "
            "so no pull request was created. The automation may not choose new files; "
            f"widening the pilot means editing `{args.allowlist}` in a reviewed pull request.",
            selected,
        )

    on_disk = os.path.join(args.repo_root, candidate)
    if not os.path.isfile(on_disk):
        return reject(
            f"`{candidate}` is on the selected list but does not exist at `{on_disk}`. It was "
            "probably moved or renamed on the branch being edited; update the allowlist.",
            selected,
        )

    print(f"`{candidate}` is one of the ten selected pilot files.")
    write_output("target", candidate)
    return 0


def reject(reason: str, selected: list[str]) -> int:
    """Report the rejection to stdout and the Actions run summary."""
    lines = [
        "### Docstring pilot — rejected, no pull request created",
        "",
        reason,
        "",
        "Selected files:",
        "",
        *(f"- `{path}`" for path in selected),
    ]
    message = "\n".join(lines)
    print(message)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(message + "\n")
    print(f"::error::{reason}")
    return 1


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


if __name__ == "__main__":
    sys.exit(main())
