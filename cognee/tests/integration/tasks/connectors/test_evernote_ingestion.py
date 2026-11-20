# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An Evernote account through the real ``cognee.add()`` pipeline, with the account faked.

``remember()`` runs this same add() step first and then cognify(), which needs an LLM, so this
stops at the Data records: an edit replaces only its note's document, a deleted or trashed note
leaves the dataset, a cut sync resumes without losing a document, and two accounts in one dataset
stay apart.
"""

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import evernote_source
from cognee.tasks.ingestion.connectors.evernote import CHUNK_SIZE, EvernoteRateLimitError
from cognee.tests.unit.tasks.connectors.fake_evernote import account

DATASET = "evernote_integration_test"
TABLE = "evernote_notes"


async def _sync(fake, resource_name: str = TABLE, **kwargs):
    source = evernote_source(service=fake, resource_name=resource_name, **kwargs)
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _rows() -> list[Data]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return []
    return [
        data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "evernote"
    ]


async def _ids(table: str = TABLE) -> dict[str, object]:
    """Note guid -> data id; the stored external id is the row id ``note:<guid>``."""
    return {
        str(data.system_metadata["external_id"]).removeprefix("note:"): data.id
        for data in await _rows()
        if data.system_metadata["table_name"] == table
    }


def _stored_text(document: Data) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_one_document_per_note_and_a_quiet_one_changes_nothing(
    clean_environment,
):
    fake = account(3)
    await _sync(fake)
    first = await _ids()

    assert set(first) == {"n1", "n2", "n3"}
    (document,) = [d for d in await _rows() if d.system_metadata["external_id"] == "note:n1"]
    assert "Notebook: Work" in _stored_text(document) and "Body of n1" in _stored_text(document)

    quiet = await _sync(fake)
    assert quiet.cognee_sync_stats["emitted"] == 0
    assert await _ids() == first


@pytest.mark.asyncio
async def test_an_edit_replaces_only_that_notes_document(clean_environment):
    fake = account(3)
    await _sync(fake)
    before = await _ids()

    fake.edit_note("n2", "Rewritten body")
    await _sync(fake)
    after = await _ids()

    assert after["n2"] != before["n2"]
    assert {k: v for k, v in after.items() if k != "n2"} == {
        k: v for k, v in before.items() if k != "n2"
    }


@pytest.mark.asyncio
async def test_trashed_expunged_and_deselected_notes_leave_the_dataset(clean_environment):
    fake = account(3)
    fake.add_note("h1", "Home note", notebook="nb2")
    fake.add_note("h2", "Home note two", notebook="nb2")
    await _sync(fake)

    fake.trash_note("n1")
    fake.expunge_note("n2")
    fake.expunge_notebook("nb2")
    await _sync(fake)

    assert set(await _ids()) == {"n3"}


@pytest.mark.asyncio
async def test_a_first_sync_cut_by_the_rate_limit_in_both_passes_resumes_without_losing_a_document(
    clean_environment,
):
    fake = account(CHUNK_SIZE + 20)
    fake.fail("get_filtered_sync_chunk", EvernoteRateLimitError("limited", retry_after=900), call=2)
    fake.fail("get_note", EvernoteRateLimitError("limited", retry_after=900), call=5)

    first = await _sync(fake)
    assert first.cognee_sync_stats["failed_rate_limit"] == 1 and await _ids() == {}
    second = await _sync(fake)
    assert second.cognee_sync_stats["failed_rate_limit"] == 1
    assert len(await _ids()) == 4
    await _sync(fake)

    assert set(await _ids()) == {f"n{i}" for i in range(1, CHUNK_SIZE + 21)}


@pytest.mark.asyncio
async def test_two_evernote_accounts_in_one_dataset_stay_apart(clean_environment):
    """Both accounts have a note ``n1``: each table keeps its own document."""
    one, two = account(2, user_id=1), account(1, user_id=2)
    two.edit_note("n1", "Another account")

    await _sync(one, "evernote_one")
    await _sync(two, "evernote_two")
    assert set(await _ids("evernote_one")) == {"n1", "n2"}
    assert set(await _ids("evernote_two")) == {"n1"}

    one.expunge_note("n1")
    await _sync(one, "evernote_one")
    await _sync(two, "evernote_two")

    assert set(await _ids("evernote_one")) == {"n2"}
    assert set(await _ids("evernote_two")) == {"n1"}
    with pytest.raises(Exception) as raised:
        await _sync(two, "evernote_one")
    chain, error = [], raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any("bound to another Evernote account" in str(e) for e in chain), chain
    assert set(await _ids("evernote_one")) == {"n2"}
