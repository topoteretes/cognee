#!/usr/bin/env python3
"""Write workflow outputs derived from a docs assessment result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write docs assessment workflow outputs")
    parser.add_argument("--assessment-json", required=True, type=Path)
    parser.add_argument("--branch-slug", required=True)
    parser.add_argument("--short-sha", required=True)
    parser.add_argument("--pr-number", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assessment = json.loads(args.assessment_json.read_text())

    needs_update = bool(assessment.get("needs_documentation_update"))
    docs_branch = f"automation/docs-pr-{args.pr_number}-{args.branch_slug}-{args.short_sha}"
    pr_title = f"docs: draft updates for PR #{args.pr_number}"

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"needs_update={'true' if needs_update else 'false'}\n")
        fh.write(f"docs_branch={docs_branch}\n")
        fh.write(f"pr_title={pr_title}\n")


if __name__ == "__main__":
    main()
