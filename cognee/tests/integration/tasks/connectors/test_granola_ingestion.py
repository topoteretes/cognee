"""A stored Granola document is replaced only when its rendered note changes, and forgotten
only when a deletion is proven.

The source runs through the real ``cognee.add()`` pipeline (which ``remember()`` runs first)
with Granola faked: the documents a first sync stores, a quiet re-sync, an edit, a deletion
the daily reconcile proves, a 401 after rows that the next sync resumes, the transcript option,
and that the key lands nowhere on disk after a run Granola refused.
"""

from datetime import datetime, timedelta, timezone

import pytest
import requests

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import granola as granola_module
from cognee.tasks.ingestion.connectors import granola_source
from cognee.tasks.ingestion.connectors.granola import GranolaError
from cognee.tests.unit.tasks.connectors.fake_granola import FakeGranola, note_id

DATASET = "granola_integration_test"
FOLDER = "fol_AAAAAAAAAAAAAA"
TABLE = "granola_notes_test"
KEY = "grn_SECRET-granola-key-123"


@pytest.fixture
def reconcile_every_run(monkeypatch):
    monkeypatch.setattr(granola_module, "RECONCILE_INTERVAL", timedelta(0))


def _account() -> FakeGranola:
    fake = FakeGranola()
    fake.folder(FOLDER)
    fake.clock = datetime.now(timezone.utc) - timedelta(hours=2)
    for n in (1, 2, 3):
        fake.add(n, folder=FOLDER)
    return fake


def _now(fake: FakeGranola) -> None:
    """Changes after a sync carry a stamp past its watermark, as Granola's would."""
    fake.clock = datetime.now(timezone.utc)


async def _sync(fake, **kwargs):
    kwargs.setdefault("service", fake)
    source = granola_source(scope=FOLDER, resource_name=TABLE, **kwargs)
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _documents() -> dict:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        str(data.system_metadata["external_id"]): data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "granola"
    }


def _text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_one_document_per_note_in_its_node_set(clean_environment):
    await _sync(_account())
    documents = await _documents()

    assert set(documents) == {note_id(1), note_id(2), note_id(3)}
    tables = {doc.system_metadata["table_name"].rpartition(".")[2] for doc in documents.values()}
    assert tables == {TABLE}
    assert {doc.external_metadata["node_set"][0] for doc in documents.values()} == {
        f"granola:{FOLDER}"
    }
    first = documents[note_id(1)]
    assert first.system_metadata["url"] == f"https://notes.granola.ai/d/{note_id(1)}"
    text = _text(first)
    assert text.startswith("# Meeting 1") and "Summary of meeting 1" in text
    assert "Speaker A: Line of meeting 1" in text and "PRIVATE" not in text


@pytest.mark.asyncio
async def test_a_quiet_sync_and_an_edit_replace_only_what_changed(clean_environment):
    fake = _account()
    await _sync(fake)
    before = {key: doc.id for key, doc in (await _documents()).items()}

    await _sync(fake)
    assert {key: doc.id for key, doc in (await _documents()).items()} == before

    _now(fake)
    fake.edit(note_id(2), summary_markdown="Revised: ship Friday")
    await _sync(fake)
    after = await _documents()

    assert after[note_id(1)].id == before[note_id(1)]
    assert after[note_id(2)].id != before[note_id(2)]
    assert "Revised: ship Friday" in _text(after[note_id(2)])


@pytest.mark.asyncio
async def test_a_note_the_reconcile_no_longer_lists_leaves_the_dataset(
    clean_environment, reconcile_every_run
):
    fake = _account()
    await _sync(fake)

    _now(fake)
    fake.delete(note_id(1))
    source = await _sync(fake)

    assert set(await _documents()) == {note_id(2), note_id(3)}
    assert source.cognee_sync_stats["deleted"] == 1


@pytest.mark.asyncio
async def test_an_empty_listing_keeps_every_document(clean_environment, reconcile_every_run):
    fake = _account()
    await _sync(fake)

    fake.listing_hidden.update(fake.notes)
    source = await _sync(fake)

    assert set(await _documents()) == {note_id(1), note_id(2), note_id(3)}
    assert source.cognee_sync_stats["failed_access"] == 1


@pytest.mark.asyncio
async def test_a_401_after_rows_keeps_them_and_the_next_sync_resumes(clean_environment):
    fake = _account()
    fake.fail("note", GranolaError("revoked", 401), after=1)

    source = await _sync(fake)
    assert source.cognee_sync_stats["failed_auth"] == 1
    assert set(await _documents()) == {note_id(3)}

    source = await _sync(fake)
    assert source.cognee_sync_stats["failed"] == 0
    assert set(await _documents()) == {note_id(1), note_id(2), note_id(3)}
    assert [path for path, _ in fake.calls].count(f"/v1/notes/{note_id(3)}") == 1


@pytest.mark.asyncio
async def test_include_transcript_off_stores_no_transcript(clean_environment):
    await _sync(_account(), include_transcript=False)
    for document in (await _documents()).values():
        assert "Transcript:" not in _text(document) and "Line of meeting" not in _text(document)


@pytest.mark.asyncio
async def test_the_key_lands_nowhere_on_disk_after_a_refused_run(
    clean_environment, monkeypatch, tmp_path
):
    def send(self, request, **kwargs):
        response = requests.Response()
        response.status_code = 401
        response._content = f'{{"error": "bad key {KEY}"}}'.encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    source = granola_source(scope=FOLDER, resource_name=TABLE, api_key=KEY)
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    assert source.cognee_sync_stats["failed_auth"] == 1
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes(), path
