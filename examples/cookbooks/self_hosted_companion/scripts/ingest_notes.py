"""Remember your notes folder: a journal, an Obsidian vault, any .md or .txt files.

Running it again remembers the folder again: new notes are added and unchanged notes are
skipped. An edited note is remembered as a new document, and its old version stays in memory.

Run alone: uv run python examples/cookbooks/self_hosted_companion/scripts/ingest_notes.py <notes folder>
"""

import argparse
import asyncio
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "companion"  # the same in every script


async def ingest_notes(notes_folder: Path) -> None:
    notes_folder = notes_folder.expanduser().resolve()
    if not notes_folder.is_dir():
        raise SystemExit(f"[ingest_notes] MISSING: no such folder {notes_folder}")
    await cognee.remember(str(notes_folder), dataset_name=DATASET, self_improvement=False)
    print(f"[ingest_notes] Remembered the notes in {notes_folder}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes_folder", type=Path, help="the folder of notes to remember")
    asyncio.run(ingest_notes(parser.parse_args().notes_folder))
