"""Self-hosted companion: check setup, remember your notes folder, then chat.

    uv run python examples/cookbooks/self_hosted_companion/setup.py    # sample notes
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --sample
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

SAMPLE = Path(__file__).parent / "sample"
SAMPLE_QUESTION = "When is my sister's birthday, and what was I planning to get her?"


def use_sample(args: argparse.Namespace) -> None:
    """Point the companion at the sample notes that setup.py writes."""
    args.notes_folder = SAMPLE / "notes"
    args.ask = args.ask or SAMPLE_QUESTION


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if args.sample and not args.notes_folder.is_dir():
        missing.append("The sample is not written. Run setup.py first.")
    elif not args.notes_folder:
        missing.append("No notes folder given. Pass its path, or --sample.")
    elif not args.notes_folder.expanduser().is_dir():
        missing.append(f"Notes folder not found: {args.notes_folder}")
    return missing


async def run(args: argparse.Namespace) -> None:
    await ingest_notes(args.notes_folder)
    await chat(args.ask)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes_folder", type=Path, nargs="?", help="the notes to remember")
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument("--ask", help="answer one message instead of an interactive chat")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if args.sample:
        use_sample(args)

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    setup_logging(log_level=ERROR)
    asyncio.run(run(args))
