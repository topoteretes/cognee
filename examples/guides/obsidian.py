"""Obsidian connector demo: an edited note replaces its old version, a deleted note is forgotten.

``obsidian_source`` returns a ``dlt`` resource that you hand straight to ``cognee.remember``.
Each ``*.md`` note of a local vault becomes one document, with its path in the vault as id.
Every run walks the whole vault, so a re-sync extracts only the notes whose text changed, and a
note you delete or rename leaves memory on the next sync. ``cognee.remember(<folder>)`` does
neither: it stores a note once.

This script builds a small vault in a temporary folder, syncs it, edits two notes and deletes
another, syncs again and asks the same question after each sync. To sync your own vault, pass
its folder as ``vault_path`` (and ``folders=[...]`` to leave out what should not be read).
Warning: this script first wipes the whole cognee instance it runs against (``prune_data()`` and
``prune_system(metadata=True)``), not just this dataset, so run it against a throwaway instance
(a scratch ``SYSTEM_ROOT_DIRECTORY`` and ``DATA_ROOT_DIRECTORY``) and remove those two calls
before pointing it at memory you want to keep.

Privacy / opt-in
----------------
The text of every note, frontmatter included, goes to your configured LLM and embedding
providers. Nothing is read until you build a source and run it. Keep a vault in its own
dataset so it is easy to inspect and forget (``cognee.forget`` does not remove the staged
copy of the text, so the next sync builds the notes again).

Setup: set your LLM key (``LLM_API_KEY``) in ``.env`` like any other cognee example. No
account, token or extra install is needed.

Run it:

    uv run python examples/guides/obsidian.py
"""

import asyncio
import tempfile
from pathlib import Path

import cognee
from cognee.tasks.ingestion.connectors import obsidian_source

# Keep the vault in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "obsidian_vault"

# Routing kwargs for remember(). Pass the same ones on every later sync.
#   write_disposition="merge" is REQUIRED: the add pipeline defaults to "replace",
#     which would wipe the whole synced vault on the next sync.
#   max_rows_per_table=0 guarantees no per-table read cap applies (even if
#     DLT_MAX_ROWS_PER_TABLE is set), so forget-on-delete compares against the
#     *entire* synced vault.
#   incremental_loading is left at its default (True) so a re-sync only builds
#     the graph for new or changed notes instead of the whole vault again.
OBSIDIAN_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
    "self_improvement": False,
}

NOTES = {
    "People/Ada Lovelace.md": "Ada Lovelace is the lead engineer of Project Atlas.\n",
    "Projects/Atlas.md": (
        "---\n"
        "title: Project Atlas\n"
        "tags: [engineering]\n"
        "---\n"
        "Project Atlas builds the routing engine. Ada Lovelace leads it.\n"
    ),
    "Reading list.md": "Grace Hopper wrote the compiler paper that Project Atlas cites.\n",
}

QUESTION = "Who leads Project Atlas?"


async def sync_and_ask(vault: Path) -> None:
    source = obsidian_source(vault_path=vault)
    await cognee.remember(source, dataset_name=DATASET_NAME, **OBSIDIAN_REMEMBER_KWARGS)
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        query_text=QUESTION,
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print(answer[0].text)


async def main():
    # Start from a clean slate so the demo is reproducible. WARNING: this wipes the whole
    # cognee instance, every dataset, so use a throwaway instance.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    with tempfile.TemporaryDirectory() as folder:
        vault = Path(folder)
        for name, text in NOTES.items():
            (vault / name).parent.mkdir(parents=True, exist_ok=True)
            (vault / name).write_text(text)

        # ── First sync: every note is read and extracted ──────────────────
        print("\n=== First sync ===")
        await sync_and_ask(vault)

        # ── Change the vault: edit one note, delete another ───────────────
        (vault / "Projects/Atlas.md").write_text(
            "---\n"
            "title: Project Atlas\n"
            "tags: [engineering]\n"
            "---\n"
            "Project Atlas builds the routing engine. Grace Hopper leads it now.\n"
        )
        (vault / "Reading list.md").unlink()
        (vault / "People/Ada Lovelace.md").write_text("Ada Lovelace left the company.\n")

        # ── Second sync: only the changed notes are extracted again ───────
        # The edited notes replace their old documents and the deleted note
        # is forgotten, so the old answer does not come back.
        print("\n=== Second sync ===")
        await sync_and_ask(vault)


if __name__ == "__main__":
    asyncio.run(main())
