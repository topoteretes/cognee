"""Helper script to run basic linting checks on submitted code."""

import subprocess
import sys


def run_ruff(file_path: str) -> str:
    result = subprocess.run(
        ["ruff", "check", file_path],
        capture_output=True,
        text=True,
    )
    return result.stdout or "No issues found."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: lint_check.py <file_path>")
        sys.exit(1)
    print(run_ruff(sys.argv[1]))
