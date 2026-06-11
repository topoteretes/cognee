#!/usr/bin/env python3
"""Write workflow outputs derived from a docs scope plan."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write docs scope plan workflow outputs")
    parser.add_argument("--scope-plan", required=True, type=Path)
    return parser.parse_args()


def parse_docs_needed(scope_plan: str) -> bool:
    match = re.search(
        r"^## Docs Needed\s*\n+(?P<value>.+?)(?:\n+## |\Z)",
        scope_plan,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not find '## Docs Needed' section in docs scope plan.")

    value = match.group("value").strip().splitlines()[0].strip().strip("`").lower()
    if value == "true":
        return True
    if value == "false":
        return False

    raise ValueError(
        "Expected '## Docs Needed' to contain `true` or `false`, "
        f"but got {value!r}."
    )


def main() -> None:
    args = parse_args()
    docs_needed = parse_docs_needed(args.scope_plan.read_text(encoding="utf-8"))

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"docs_needed={'true' if docs_needed else 'false'}\n")


if __name__ == "__main__":
    main()
