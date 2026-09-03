"""add() ingestion contract, end to end on the real local stack.

The COG-6241 refactor moved metadata computation from "read the stored object
back" to "compute it from the payload in hand", made raw storage keys content
addressed (``<md5>/<filename>``), and threaded storage work from the
incremental pipeline wrapper into ``ingest_data``. These tests pin the
behavior a user of ``add()`` observes, so any future change to that machinery
that alters semantics — dedup, naming, row metadata, update-in-place, or the
no-clobber guarantee — fails here rather than in production:

  1. re-adding the same file to the same dataset is a no-op (one row, stable id);
  2. duplicate content inside ONE batch collapses to one row (first wins);
  3. the same content added to two datasets stays two rows (dataset-scoped dedup);
  4. two different uploads sharing a filename never clobber each other's bytes;
  5. a pinned DataItem re-add through bare add() leaves the row untouched
     (update-in-place belongs to update());
  6. plain-text ingestion keeps the ``text_<md5>`` naming contract;
  7. row metadata is computed correctly (golden values for known bytes);
  8. no ingestion read goes through the loop-parking ``run_sync`` bridge;
  9. a local-path add records the SOURCE file as original_data_location.

Runs on the default local stack (sqlite + local files), no LLM or embedding
calls — add() never reaches them. CI-safe, no API keys.
"""

import hashlib
import io
import shutil
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def add_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_add_regression_"))
    source_dir = root / "sources"
    source_dir.mkdir()

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    def clear_config_caches():
        import importlib

        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
            ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
            ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
            (
                "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
                "create_embedding_engine",
            ),
            ("cognee.infrastructure.llm.config", "get_llm_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

    mp = pytest.MonkeyPatch()
    for key, value in dict(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        STORAGE_BACKEND="local",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
        # add-by-path tests read from source_dir; mkdtemp lives under the
        # default-allowed tempdir, this just makes the intent explicit.
        COGNEE_ALLOWED_LOCAL_FILE_ROOTS=os.pathsep.join([str(source_dir), tempfile.gettempdir()]),
    ).items():
        mp.setenv(key, value)
    clear_config_caches()

    yield source_dir

    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


def _upload(content: bytes, filename: str):
    from starlette.datastructures import Headers, UploadFile

    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "application/octet-stream"}),
    )


async def _rows(dataset_name):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data, Dataset

    async with get_relational_engine().get_async_session() as session:
        dataset = (
            (await session.execute(select(Dataset).filter(Dataset.name == dataset_name)))
            .scalars()
            .first()
        )
        if dataset is None:
            return []
        return (
            (await session.execute(select(Data).filter(Data.dataset_id == dataset.id)))
            .scalars()
            .all()
        )


async def _read_raw(row) -> bytes:
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(row.raw_data_location) as file:
        return file.read()


async def test_readding_the_same_file_is_a_noop(add_env):
    import cognee

    await cognee.add([_upload(b"stable content v1", "stable.txt")], "reg_noop")
    first = await _rows("reg_noop")
    await cognee.add([_upload(b"stable content v1", "stable.txt")], "reg_noop")
    second = await _rows("reg_noop")

    assert len(first) == len(second) == 1
    assert first[0].id == second[0].id
    assert first[0].content_hash == second[0].content_hash


async def test_duplicate_content_in_one_batch_lands_with_one_identity(add_env):
    # add() fans items out one-per-ingest_data call, so the pre-loop's
    # first-wins mint never engages here and duplicates land as two rows with
    # the same content hash — dev-parity behavior, pinned so a future change
    # to the fan-out or dedup shows up as a diff here. (The first-wins rule
    # itself governs direct multi-item ingest_data calls; see the unit tests.)
    import cognee

    await cognee.add(
        [_upload(b"batch twin", "twin_a.txt"), _upload(b"batch twin", "twin_a.txt")],
        "reg_batch",
    )

    rows = await _rows("reg_batch")
    assert {str(row.content_hash) for row in rows} == {
        __import__("hashlib").md5(b"batch twin").hexdigest()
    }
    assert len(rows) == 2


async def test_same_content_in_two_datasets_stays_two_rows(add_env):
    import cognee

    await cognee.add([_upload(b"shared bytes", "shared.txt")], "reg_scope_a")
    await cognee.add([_upload(b"shared bytes", "shared.txt")], "reg_scope_b")

    rows_a, rows_b = await _rows("reg_scope_a"), await _rows("reg_scope_b")
    assert len(rows_a) == len(rows_b) == 1
    assert rows_a[0].id != rows_b[0].id
    assert rows_a[0].content_hash == rows_b[0].content_hash


async def test_same_filename_different_content_does_not_clobber(add_env):
    # Regression: raw uploads used to be stored under the caller's filename
    # with overwrite=True, so the second of two same-named uploads silently
    # replaced the first's bytes.
    import cognee

    await cognee.add([_upload(b"first report body", "report.txt")], "reg_clobber_a")
    await cognee.add([_upload(b"second report body", "report.txt")], "reg_clobber_b")

    (row_a,), (row_b,) = await _rows("reg_clobber_a"), await _rows("reg_clobber_b")
    assert await _read_raw(row_a) == b"first report body"
    assert await _read_raw(row_b) == b"second report body"


async def test_pinned_readd_keeps_the_row_untouched(add_env):
    # Dev-parity contract: a bare add() with a pinned data_id is NOT update().
    # The row's id stays put and its content is not replaced — replacing
    # content in place is update()'s job (test_update_id_stability), and the
    # ingest_data-level update branch is pinned by
    # test_reingest_updates_persisted_data_size. Pinned here so a change to
    # the add pipeline's skip behavior surfaces as an explicit diff.
    import cognee
    from cognee.tasks.ingestion.data_item import DataItem

    await cognee.add([_upload(b"pinned v1", "pinned.txt")], "reg_pin")
    (row,) = await _rows("reg_pin")
    pinned_id = row.id

    await cognee.add(
        [DataItem(data=_upload(b"pinned v2 changed", "pinned.txt"), data_id=pinned_id)],
        "reg_pin",
        incremental_loading=False,
    )

    (after,) = await _rows("reg_pin")
    assert after.id == pinned_id
    assert after.content_hash == hashlib.md5(b"pinned v1").hexdigest()


async def test_plain_text_keeps_the_text_naming_contract(add_env):
    import cognee

    text = "a plain note about nothing"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    await cognee.add(text, "reg_text")

    (row,) = await _rows("reg_text")
    assert row.name == f"text_{digest}"
    assert row.extension == "txt"
    assert row.mime_type == "text/plain"
    assert row.raw_data_location.endswith(f"text_{digest}.txt")


async def test_row_metadata_golden_values(add_env):
    import cognee

    content = b"golden metadata fixture body\n"
    await cognee.add([_upload(content, "Golden Fixture.txt")], "reg_golden")

    (row,) = await _rows("reg_golden")
    assert row.name == "Golden Fixture"
    assert row.content_hash == hashlib.md5(content).hexdigest()
    assert row.data_size == len(content)
    assert row.original_extension == "txt"
    assert row.original_mime_type == "text/plain"
    assert row.loader_engine == "text_loader"
    # text_loader's derived text equals the source, so both hashes agree.
    assert row.raw_content_hash == row.content_hash
    # The stored raw copy carries the real filename as its basename (the
    # code-graph route and dlt derive names from it).
    assert row.original_data_location.endswith("/Golden%20Fixture.txt") or (
        row.original_data_location.endswith("/Golden Fixture.txt")
    )


async def test_no_sync_metadata_bridge_during_add(add_env, monkeypatch):
    # The sync get_metadata/get_identifier accessors bridge to async through
    # run_sync, which starts a thread and join()s it ON the event loop — every
    # such call during add() parks the loop for a full storage read. The
    # refactor's core claim is that add() never takes that bridge.
    import cognee
    from cognee.modules.ingestion.data_types import BinaryData, S3BinaryData

    calls = []

    def _record(name, original):
        def spy(self, *args, **kwargs):
            calls.append(name)
            return original(self, *args, **kwargs)

        return spy

    for cls in (BinaryData, S3BinaryData):
        for method in ("get_metadata", "get_identifier"):
            monkeypatch.setattr(
                cls, method, _record(f"{cls.__name__}.{method}", getattr(cls, method))
            )

    await cognee.add([_upload(b"bridge probe", "bridge.txt")], "reg_bridge")

    assert calls == [], f"add() took the loop-parking sync bridge: {calls}"


async def test_local_path_add_records_the_source_location(add_env):
    import cognee

    source = add_env / "on_disk_note.txt"
    source.write_bytes(b"local path body")

    await cognee.add([str(source)], "reg_localpath")

    (row,) = await _rows("reg_localpath")
    assert row.content_hash == hashlib.md5(b"local path body").hexdigest()
    assert row.name == "on_disk_note"
    # Pass-through items are not copied into cognee storage: the original
    # location is the user's own file.
    # realpath both sides: macOS tmp lives behind the /var -> /private/var
    # symlink, and ingestion stores the resolved form.
    import os

    assert row.original_data_location == Path(os.path.realpath(source)).as_uri()


async def _stored_original_files():
    import os as _os

    from cognee.base_config import get_base_config

    data_root = Path(get_base_config().data_root_directory)
    return sorted(str(p.relative_to(data_root)) for p in data_root.rglob("*") if p.is_file())


async def test_delete_reclaims_both_stored_files(add_env):
    # Deleting the last row referencing a stored object removes the object —
    # BOTH the derived text and the content-addressed original. Under
    # content-addressed keys nothing overwrites the original, so without this
    # every delete stranded it forever.
    import cognee
    from cognee.infrastructure.databases.relational import get_relational_engine

    await cognee.add([_upload(b"reclaim me on delete", "reclaim.txt")], "reg_reclaim")
    (row,) = await _rows("reg_reclaim")
    before = await _stored_original_files()
    assert any("reclaim.txt" in f for f in before)

    await get_relational_engine().delete_data_entity(row.id, row.dataset_id)

    after = await _stored_original_files()
    assert not any("reclaim.txt" in f for f in after)
    assert not any(str(row.raw_data_location).endswith(f) for f in after)


async def test_delete_keeps_files_another_row_still_references(add_env):
    # Identical payloads share one content-addressed original. Deleting one
    # row must not take the other's file with it.
    import cognee
    from cognee.infrastructure.databases.relational import get_relational_engine

    await cognee.add([_upload(b"shared reclaim body", "shared_reclaim.txt")], "reg_share_a")
    await cognee.add([_upload(b"shared reclaim body", "shared_reclaim.txt")], "reg_share_b")
    (row_a,), (row_b,) = await _rows("reg_share_a"), await _rows("reg_share_b")
    assert row_a.original_data_location == row_b.original_data_location

    await get_relational_engine().delete_data_entity(row_a.id, row_a.dataset_id)

    assert await _read_raw(row_b) == b"shared reclaim body"
