"""A stored Obsidian document is replaced only when its note's text changes.

The source runs through the real cognee.add() pipeline on a vault in ``tmp_path``. remember()
runs this same add() step first and then cognify(), which needs an LLM, so this stops at the
Data records like the Linear and Apollo tests; ``test_obsidian_forget.py`` goes through
remember() and the graph. A note's file time and a note it cannot read change nothing, and
two vaults in one dataset stay apart.
"""

import os

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import obsidian_source

DATASET = "obsidian_integration_test"
TABLE = "obsidian_test"
PLAN = "---\ntitle: Q4 plan\ntags: [work]\n---\nShip the engine.\n"
NOTES = {"Inbox.md": "Call Ada.\n", "Projects/Plan.md": PLAN}


def _vault(root, notes=NOTES):
    for rel, text in notes.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


async def _sync(vault, resource_name: str | None = TABLE, **kwargs):
    source = obsidian_source(vault_path=vault, resource_name=resource_name, **kwargs)
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "obsidian"
    }


async def _tables() -> dict[tuple[str, str], object]:
    """Every Obsidian document by (table, note id), which ``_documents`` cannot tell apart."""
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    return {
        (data.system_metadata["table_name"], data.system_metadata["external_id"]): data.id
        for data in await get_dataset_data(dataset.id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "obsidian"
    }


def _ids(documents) -> dict[str, object]:
    return {note: document.id for note, document in documents.items()}


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_tagged_documents_and_a_quiet_one_changes_nothing(
    clean_environment, tmp_path
):
    vault = _vault(tmp_path / "vault")
    await _sync(vault)
    first = await _documents()

    assert set(first) == {"obsidian:Inbox.md", "obsidian:Projects/Plan.md"}
    assert {doc.system_metadata["table_name"] for doc in first.values()} == {TABLE}
    plan = first["obsidian:Projects/Plan.md"]
    # The document is "# title" and the file as read, frontmatter included; cognee strips the
    # ends of the text when it builds a document.
    assert _stored_text(plan) == f"# Q4 plan\n\n{PLAN}".strip()
    assert _stored_text(first["obsidian:Inbox.md"]) == "# Inbox\n\nCall Ada."
    assert "url" not in plan.system_metadata
    assert [(a["kind"], a["id"]) for a in plan.system_metadata["structure"]["ancestors"]] == [
        ("folder", "Projects")
    ]
    assert "structure" not in first["obsidian:Inbox.md"].system_metadata

    quiet = await _sync(vault)
    assert quiet.cognee_sync_stats["synced"] == 2
    assert _ids(await _documents()) == _ids(first)


@pytest.mark.asyncio
async def test_an_edit_replaces_only_that_notes_document(clean_environment, tmp_path):
    vault = _vault(tmp_path / "vault")
    await _sync(vault)
    before = _ids(await _documents())

    (vault / "Inbox.md").write_text("Call Ben instead.\n")
    await _sync(vault)
    after = await _documents()

    assert after["obsidian:Inbox.md"].id != before["obsidian:Inbox.md"]
    assert "Call Ben instead." in _stored_text(after["obsidian:Inbox.md"])
    assert after["obsidian:Projects/Plan.md"].id == before["obsidian:Projects/Plan.md"]


@pytest.mark.asyncio
async def test_a_newer_file_time_with_the_same_text_replaces_no_document(
    clean_environment, tmp_path
):
    vault = _vault(tmp_path / "vault")
    await _sync(vault)
    before = _ids(await _documents())

    for rel in NOTES:
        os.utime(vault / rel, (2_000_000_000, 2_000_000_000))
    await _sync(vault)

    assert _ids(await _documents()) == before


@pytest.mark.asyncio
async def test_a_note_that_cannot_be_read_keeps_its_document_and_is_counted(
    clean_environment, tmp_path
):
    vault = _vault(tmp_path / "vault")
    await _sync(vault)
    before = _ids(await _documents())

    (vault / "Inbox.md").write_bytes(b"\xff\xfe broken")
    source = await _sync(vault)

    assert source.cognee_sync_stats["failed_read"] == 1
    assert _ids(await _documents()) == before


@pytest.mark.asyncio
async def test_an_empty_vault_syncs_and_later_notes_arrive(clean_environment, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    empty = await _sync(vault)
    assert empty.cognee_sync_stats["synced"] == 0
    assert await _documents() == {}

    _vault(vault)
    await _sync(vault)
    assert set(await _documents()) == {"obsidian:Inbox.md", "obsidian:Projects/Plan.md"}


@pytest.mark.asyncio
async def test_two_vaults_in_one_dataset_stay_apart(clean_environment, tmp_path):
    first = _vault(tmp_path / "first")
    second = _vault(tmp_path / "second", {"Inbox.md": "Another vault.\n"})
    await _sync(first, "obsidian_first")
    await _sync(second, "obsidian_second")
    before = await _tables()
    assert before.keys() == {
        ("obsidian_first", "obsidian:Inbox.md"),
        ("obsidian_first", "obsidian:Projects/Plan.md"),
        ("obsidian_second", "obsidian:Inbox.md"),
    }, "the same note id in two tables is two documents"

    (first / "Inbox.md").unlink()
    await _sync(first, "obsidian_first")

    after = await _tables()
    assert after.keys() == before.keys() - {("obsidian_first", "obsidian:Inbox.md")}
    assert (
        after[("obsidian_second", "obsidian:Inbox.md")]
        == before[("obsidian_second", "obsidian:Inbox.md")]
    )


@pytest.mark.asyncio
async def test_moving_the_vault_folder_syncs_it_again_under_the_new_default_name(
    clean_environment, tmp_path
):
    """The default resource name follows the vault path, and cleanup is scoped by table, so
    the documents synced from the old path stay until they are forgotten by hand."""
    old = _vault(tmp_path / "old")
    old_table = (await _sync(old, None)).name

    moved = old.rename(tmp_path / "new")
    new_table = (await _sync(moved, None)).name

    assert new_table != old_table
    assert {table for table, _ in await _tables()} == {old_table, new_table}
