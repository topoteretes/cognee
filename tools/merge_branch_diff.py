#!/usr/bin/env python3
"""Utilities for extracting the actual branch delta from a merge commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_git_command(command: list[str], cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(cwd) if cwd is not None else None,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"Error running git command: {' '.join(command)}", file=sys.stderr)
        print(exc.stderr, file=sys.stderr)
        raise SystemExit(1) from exc


def get_merge_base(first_parent: str, second_parent: str, cwd: str | Path | None = None) -> str:
    return run_git_command(["git", "merge-base", first_parent, second_parent], cwd=cwd)


def get_branch_changed_files(
    first_parent: str, second_parent: str, cwd: str | Path | None = None
) -> list[str]:
    merge_base = get_merge_base(first_parent, second_parent, cwd=cwd)
    return [
        line
        for line in run_git_command(
            ["git", "diff", "--name-only", merge_base, second_parent],
            cwd=cwd,
        ).splitlines()
        if line.strip()
    ]


def get_branch_diff_stat(
    first_parent: str, second_parent: str, cwd: str | Path | None = None
) -> str:
    merge_base = get_merge_base(first_parent, second_parent, cwd=cwd)
    return run_git_command(["git", "diff", "--stat", merge_base, second_parent], cwd=cwd)


def get_branch_commit_subjects(
    first_parent: str, second_parent: str, cwd: str | Path | None = None
) -> list[str]:
    merge_base = get_merge_base(first_parent, second_parent, cwd=cwd)
    return [
        line
        for line in run_git_command(
            [
                "git",
                "log",
                "--no-merges",
                "--pretty=format:- %s (%h)",
                f"{merge_base}..{second_parent}",
            ],
            cwd=cwd,
        ).splitlines()
        if line.strip()
    ]
