"""Self-hosted companion: check setup, remember your notes folder, then chat.

    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py    # the sample
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py --check ~/Documents/journal
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py ~/Documents/journal
    uv run python examples/cookbooks/self_hosted_companion/self_hosted_companion.py ~/Documents/journal \
        --ask "When is my sister's birthday?"

With no notes folder given it runs on the sample notes from setup.py, so it works with
only LLM_API_KEY. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script
failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.chat import chat
from scripts.clear import clear
from scripts.ingest_notes import ingest_notes
from scripts.ui import open_ui
from setup import write_sample

SAMPLE = Path(__file__).parent / "sample"
SAMPLE_QUESTION = "When is my sister's birthday, and what was I planning to get her?"


def no_sources(args: argparse.Namespace) -> bool:
    """True when no notes folder is given, so the sample stands in."""
    return not args.notes_folder


def use_sample(args: argparse.Namespace) -> None:
    """Point the companion at the sample notes that setup.py writes."""
    args.notes_folder = SAMPLE / "notes"
    args.ask = args.ask or SAMPLE_QUESTION


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if not args.sample and not args.notes_folder.expanduser().is_dir():
        missing.append(f"Notes folder not found: {args.notes_folder}")
    return missing


async def run(args: argparse.Namespace) -> None:
    if args.clear:
        await clear()
    await ingest_notes(args.notes_folder)
    await chat(args.ask)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes_folder", type=Path, nargs="?", help="the notes to remember")
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        help="forget the dataset before remembering (default: on for the sample, off otherwise)",
    )
    parser.add_argument("--ask", help="answer one message instead of an interactive chat")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if not args.sample and no_sources(args):
        print("[setup] No notes folder given, so this runs on the sample notes.")
        args.sample = True
    if args.sample:
        use_sample(args)

    if args.clear is None:  # a sample run starts from an empty dataset unless --no-clear
        args.clear = args.sample
    if args.clear:
        print("[setup] CLEAR: the cookbook's dataset is forgotten before this run.")

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args))
