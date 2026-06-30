"""Shared next-step hints for the CLI."""

from __future__ import annotations


def format_next_step_hint(command: str, dataset_name: str | None = None) -> str:
    """Return a short, copy-pasteable next command for common happy paths."""
    command = (command or "").strip().lower()
    dataset = (dataset_name or "").strip()

    if command == "add":
        if dataset:
            return f"Next: `cognee cognify --datasets {dataset}`"
        return "Next: `cognee cognify`"

    if command == "cognify":
        if dataset:
            return f"Next: `cognee recall \"What did I add?\" --datasets {dataset}`"
        return "Next: `cognee recall \"What did I add?\"`"

    if command == "remember":
        if dataset:
            return f"Next: `cognee recall \"What did I add?\" --datasets {dataset}`"
        return "Next: `cognee recall \"What did I add?\"`"

    if command == "recall":
        if dataset:
            return f"Try adding data first: `cognee remember \"...\" --dataset-name {dataset}`"
        return "Try adding data first: `cognee remember \"...\"`"

    return ""
