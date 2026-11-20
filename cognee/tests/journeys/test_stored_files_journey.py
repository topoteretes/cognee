"""Journey: no stored file outlives the document that owned it.

Cognee keeps a document's bytes under the data root: the original, and the
text a loader derived from it. ``forget()`` removes the files a ``Data`` row
points at, so a file no row points at would stay on disk for good, with the
content of a document the user updated or deleted. After every step here the
data root holds the current version of the document and nothing else: an
update removes the version it replaced, and an update that does not publish
removes what it staged. A file another document still uses is kept, and so is
one that an add still in flight is about to use. A file that cannot be removed
fails the update, with an error that says what state the document was left in.
"""

from __future__ import annotations

import asyncio
import importlib
import io
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

import cognee
from cognee.api.v1.update import incremental
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions import StoredFileNotRemovedError
from cognee.tests.journeys import _support, mock_ai

# ``cognee.api.v1.update.update`` is the function when reached through the package.
update_module = importlib.import_module("cognee.api.v1.update.update")

DATASET = "journey_stored_files"
OTHER_DATASET = "journey_stored_files_other"

CONDUCTORS = {"v1": "Yusuf Haldane", "v2": "Ingrid Solvay", "v3": "Olamide Farquhar"}


def _document(version: str) -> str:
    return (
        "Title: Saltmarsh Choir\n\nThe Saltmarsh Choir rehearses in the Corn Exchange under "
        f"conductor {CONDUCTORS[version]}."
    )


def _upload(version: str, filename: str = "choir.txt") -> UploadFile:
    """The document as an API upload: stored as an original plus its derived text."""
    return UploadFile(file=io.BytesIO(_document(version).encode()), filename=filename)


def _stored(root: Path) -> dict[str, str]:
    """Every file under the data root, by path, with the version whose text it holds."""
    stored = {}
    for path in sorted((root / "data").rglob("*")):
        if path.is_file():
            text = path.read_text(errors="replace")
            stored[str(path)] = next(
                (version for version, name in CONDUCTORS.items() if name in text), "other"
            )
    return stored


def _versions(root: Path) -> set[str]:
    return set(_stored(root).values())


async def _only_row(dataset_id, user):
    [row] = await cognee.datasets.list_data(dataset_id, user)
    return row


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.parametrize("as_upload", [False, True], ids=["text", "upload"])
@pytest.mark.parametrize(
    "chunk_level_diff, status", [(True, "incremental"), (False, "full_rebuild")]
)
async def test_an_update_leaves_only_the_current_version_on_disk(
    clean_env, default_user, as_upload, chunk_level_diff, status
):
    def source(version):
        return _upload(version) if as_upload else _document(version)

    result = await cognee.remember(source("v1"), dataset_name=DATASET, self_improvement=False)
    dataset_id = _support.as_uuid(result.dataset_id)
    row = await _only_row(dataset_id, default_user)
    assert _versions(clean_env) == {"v1"}
    # An upload keeps two files, its original and the derived text; raw text is one.
    assert len(_stored(clean_env)) == (2 if as_upload else 1)

    for version in ("v2", "v3"):
        updated = await cognee.update(
            data_id=row.id,
            data=source(version),
            dataset_id=dataset_id,
            user=default_user,
            chunk_level_diff=chunk_level_diff,
        )
        assert updated["status"] == status
        assert _versions(clean_env) == {version}, (
            f"after the update to {version} the data root holds {_stored(clean_env)}"
        )
        assert len(_stored(clean_env)) == (2 if as_upload else 1)

    # The row reads back from the files that are left.
    row = await _only_row(dataset_id, default_user)
    assert CONDUCTORS["v3"].lower() in _support.stored_text(row)

    await cognee.forget(data_id=row.id, dataset_id=dataset_id, user=default_user)
    assert _stored(clean_env) == {}, "forget() left a stored file behind"


@pytest.mark.journey
@pytest.mark.asyncio
async def test_an_update_that_does_not_publish_leaves_nothing_it_staged(clean_env, default_user):
    result = await cognee.remember(_upload("v1"), dataset_name=DATASET, self_improvement=False)
    dataset_id = _support.as_uuid(result.dataset_id)
    row = await _only_row(dataset_id, default_user)
    before = _stored(clean_env)
    assert set(before.values()) == {"v1"} and len(before) == 2

    # The same content again: nothing to publish, so the staged copy goes.
    unchanged = await cognee.update(
        data_id=row.id, data=_upload("v1"), dataset_id=dataset_id, user=default_user
    )
    assert unchanged["status"] == "unchanged"
    assert _stored(clean_env) == before

    # New content whose extraction fails: the document stays on v1, and the
    # v2 content the update staged is not left on disk.
    injector = mock_ai.inject_llm_failure(RuntimeError("injected LLM outage"))
    try:
        injector.armed = True
        with pytest.raises(RuntimeError, match="injected LLM outage"):
            await cognee.update(
                data_id=row.id, data=_upload("v2"), dataset_id=dataset_id, user=default_user
            )
        assert injector.trips >= 1
    finally:
        injector.uninstall()
    assert _stored(clean_env) == before

    await cognee.forget(data_id=row.id, dataset_id=dataset_id, user=default_user)
    assert _stored(clean_env) == {}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_file_another_document_still_uses_is_kept(clean_env, default_user):
    # Identical text in two datasets is one content-addressed file with two rows.
    first = await cognee.remember(_document("v1"), dataset_name=DATASET, self_improvement=False)
    second = await cognee.remember(
        _document("v1"), dataset_name=OTHER_DATASET, self_improvement=False
    )
    dataset_id, other_id = _support.as_uuid(first.dataset_id), _support.as_uuid(second.dataset_id)
    row = await _only_row(dataset_id, default_user)
    other_row = await _only_row(other_id, default_user)
    assert row.raw_data_location == other_row.raw_data_location
    assert len(_stored(clean_env)) == 1

    updated = await cognee.update(
        data_id=row.id, data=_document("v2"), dataset_id=dataset_id, user=default_user
    )
    assert updated["status"] == "incremental"

    # The first document moved on; the second still reads its v1 file.
    assert _versions(clean_env) == {"v1", "v2"}
    assert CONDUCTORS["v1"].lower() in _support.stored_text(other_row)

    await cognee.forget(data_id=other_row.id, dataset_id=other_id, user=default_user)
    assert _versions(clean_env) == {"v2"}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_skipped_duplicate_add_leaves_the_file_an_add_in_flight_is_about_to_use(
    clean_env, default_user, monkeypatch
):
    """Two adds of the same bytes under the same name share one stored file. The one that is
    skipped as a duplicate must not remove it: the other may be between saving the file and
    committing its row, and would be left pointing at nothing."""
    ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")
    await cognee.remember(_upload("v1", filename="choir.txt"), dataset_name=DATASET)

    # Hold an add into another dataset after its file is saved and before its row is committed.
    saved, release = asyncio.Event(), asyncio.Event()
    load_text = ingest_module.data_item_to_text_file

    async def held(*args, **kwargs):
        saved.set()
        await release.wait()
        return await load_text(*args, **kwargs)

    monkeypatch.setattr(ingest_module, "data_item_to_text_file", held)
    in_flight = asyncio.create_task(
        cognee.add(
            _upload("v1", filename="choir_copy.txt"), dataset_name=OTHER_DATASET, user=default_user
        )
    )
    await asyncio.wait_for(saved.wait(), timeout=60)
    monkeypatch.setattr(ingest_module, "data_item_to_text_file", load_text)

    # The same upload into the first dataset is a duplicate there and is skipped.
    await cognee.add(
        _upload("v1", filename="choir_copy.txt"), dataset_name=DATASET, user=default_user
    )

    release.set()
    await in_flight
    other_id = await _support.dataset_id_by_name(OTHER_DATASET, default_user)
    row = await _only_row(other_id, default_user)
    assert Path(row.original_data_location.removeprefix("file://")).is_file()
    assert CONDUCTORS["v1"].lower() in _support.stored_text(row)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_file_that_cannot_be_removed_fails_the_update_and_says_what_it_left(
    clean_env, default_user, monkeypatch
):
    async def storage_down(self, location):
        raise OSError("storage is down")

    removal = "remove_data_file"  # the storage step, after the reference check

    engine_class = type(get_relational_engine())

    # --- chunk-level update: the new content is published before the cleanup ---------
    result = await cognee.remember(_document("v1"), dataset_name=DATASET, self_improvement=False)
    dataset_id = _support.as_uuid(result.dataset_id)
    row = await _only_row(dataset_id, default_user)
    old_file = row.raw_data_location

    with monkeypatch.context() as patched:
        patched.setattr(engine_class, removal, storage_down)
        with pytest.raises(StoredFileNotRemovedError) as raised:
            await cognee.update(
                data_id=row.id, data=_document("v2"), dataset_id=dataset_id, user=default_user
            )

    error = raised.value
    assert error.outcome == incremental.UPDATE_APPLIED
    assert error.locations == [old_file]
    assert "The update was applied" in str(error) and "storage is down" in str(error)
    # What the error says is true: the document holds v2, and the v1 file is still there.
    assert CONDUCTORS["v2"].lower() in _support.stored_text(
        await _only_row(dataset_id, default_user)
    )
    assert _versions(clean_env) == {"v1", "v2"}

    # --- full rebuild: the cleanup fails between storing the content and rebuilding ----
    result = await cognee.remember(
        _upload("v1"), dataset_name=OTHER_DATASET, self_improvement=False
    )
    other_id = _support.as_uuid(result.dataset_id)
    other_row = await _only_row(other_id, default_user)

    with monkeypatch.context() as patched:
        patched.setattr(engine_class, removal, storage_down)
        with pytest.raises(StoredFileNotRemovedError) as raised:
            await cognee.update(
                data_id=other_row.id,
                data=_upload("v2"),
                dataset_id=other_id,
                user=default_user,
                chunk_level_diff=False,
            )

    assert raised.value.outcome == update_module.REBUILD_INTERRUPTED
    assert "its memory has not been rebuilt" in str(raised.value)
    interrupted = await _support.snapshot_dataset(other_id, default_user)
    assert interrupted.graph_nodes == 0, "the error says the memory was not rebuilt"

    # Running the update again, as the error says, rebuilds it.
    again = await cognee.update(
        data_id=other_row.id, data=_upload("v2"), dataset_id=other_id, user=default_user
    )
    assert again["status"] == "full_rebuild"
    rebuilt = await _support.snapshot_dataset(other_id, default_user)
    assert rebuilt.graph_nodes > 0
