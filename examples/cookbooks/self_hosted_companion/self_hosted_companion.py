"""Self-hosted companion: check setup, remember your notes folder, then chat.

    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --check ~/Documents/journal
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py ~/Documents/journal
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py ~/Documents/journal \
        --ask "When is my sister's birthday?"

Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script
failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from scripts.chat import chat
from scripts.ingest_notes import ingest_notes
from scripts.ui import open_ui

from cognee.shared.logging_utils import ERROR, setup_logging


def missing_setup(notes_folder: Path) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if not notes_folder.expanduser().is_dir():
        missing.append(f"Notes folder not found: {notes_folder}")
    return missing


async def run(args: argparse.Namespace) -> None:
    await ingest_notes(args.notes_folder)
    await chat(args.ask)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes_folder", type=Path, help="the folder of notes to remember")
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--ask", help="answer one message instead of an interactive chat")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    missing = missing_setup(args.notes_folder)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    setup_logging(log_level=ERROR)
    asyncio.run(run(args))
