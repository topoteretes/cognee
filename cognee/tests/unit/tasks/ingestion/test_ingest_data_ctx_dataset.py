"""ingest_data reuses the pipeline's already-resolved dataset (``ctx.dataset``)
instead of resolving name/id + permissions again on every call.

The incremental pipeline calls ingest_data once per item; on the cloud pods
(NullPool over Neon) each of the three resolution lookups is a fresh TLS+SCRAM
connection, so this is ~3 connections per uploaded file. The reuse must be
conservative: only when ``ctx.dataset`` demonstrably IS the dataset the caller
selected (by id, or by name for a dataset the caller owns); anything else goes
through the full resolution + permission check as before.
"""

import importlib
import os
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data, Dataset
from cognee.modules.ingestion import StoredFile

# The package re-exports the FUNCTION under the module's name, so
# `import ... as ingest_module` would bind the function and break patch.object.
ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()


class _FakeFile:
    name = "/tmp/doc.txt"

    def read(self, *_):
        return b""

    def seek(self, *_):
        return 0

    def tell(self):
        return 0


class _FakeOpen:
    def __init__(self, *_a, **_k):
        pass

    async def __aenter__(self):
        return _FakeFile()

    async def __aexit__(self, *_):
        return False


def _metadata():
    return {
        "name": "doc",
        "file_path": "/tmp/doc.txt",
        "mime_type": "text/plain",
        "extension": "txt",
        "content_hash": "hash-1",
        "file_size": 11,
    }


async def _make_engine():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()
    return engine, tmp.name


def _install_mocks(stack, engine):
    meta = _metadata()

    async def _aget_metadata():
        return meta

    async def _aget_identifier():
        return meta["content_hash"]

    classified = SimpleNamespace(
        get_metadata=lambda: meta,
        get_identifier=lambda: meta["content_hash"],
        aget_metadata=_aget_metadata,
        aget_identifier=_aget_identifier,
    )
    stack.enter_context(patch.object(ingest_module, "get_relational_engine", lambda: engine))
    stack.enter_context(
        patch.object(
            ingest_module,
            "save_data_item_to_storage_detailed",
            AsyncMock(return_value=StoredFile(file_path="/tmp/doc.txt")),
        )
    )
    stack.enter_context(patch.object(ingest_module, "get_data_file_path", lambda p: p))
    stack.enter_context(patch.object(ingest_module, "open_data_file", _FakeOpen))
    stack.enter_context(
        patch.object(
            ingest_module,
            "data_item_to_text_file",
            AsyncMock(return_value=("/tmp/doc.txt", SimpleNamespace(loader_name="text_loader"))),
        )
    )
    stack.enter_context(patch.object(ingest_module.ingestion, "classify", lambda _f: classified))

    resolution = {
        "by_id": AsyncMock(side_effect=AssertionError("resolved dataset by id")),
        "by_name": AsyncMock(side_effect=AssertionError("resolved dataset by name")),
        "create": AsyncMock(side_effect=AssertionError("load_or_create_datasets called")),
    }
    stack.enter_context(
        patch.object(ingest_module, "get_specific_user_permission_datasets", resolution["by_id"])
    )
    stack.enter_context(
        patch.object(ingest_module, "get_authorized_existing_datasets", resolution["by_name"])
    )
    stack.enter_context(
        patch.object(ingest_module, "load_or_create_datasets", resolution["create"])
    )
    return resolution


@pytest.mark.asyncio
async def test_matching_ctx_dataset_skips_resolution_and_writes_the_row():
    engine, db_path = await _make_engine()
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine)
            rows = await ingest_module.ingest_data(
                data="hello world",
                dataset_name="ds",
                user=USER,
                dataset_id=DATASET_ID,
                ctx=SimpleNamespace(dataset=dataset, user=USER),
            )
        assert len(rows) == 1
        async with engine.get_async_session() as session:
            stored = await session.get(Data, rows[0].id)
            assert stored is not None
            assert stored.dataset_id == DATASET_ID
            assert stored.content_hash == "hash-1"
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_matching_ctx_dataset_by_owned_name_skips_resolution():
    engine, db_path = await _make_engine()
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id, tenant_id=None)
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine)
            rows = await ingest_module.ingest_data(
                data="hello world",
                dataset_name="ds",
                user=USER,
                ctx=SimpleNamespace(dataset=dataset, user=USER),
            )
        assert len(rows) == 1 and rows[0].dataset_id == DATASET_ID
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_mismatched_ctx_dataset_falls_back_to_full_resolution():
    """A custom pipeline may point ingest_data at a different dataset than the
    run's; that path must still resolve + permission-check as before."""
    engine, db_path = await _make_engine()
    other = Dataset(id=uuid4(), name="other", owner_id=USER.id)
    target = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    try:
        with ExitStack() as stack:
            resolution = _install_mocks(stack, engine)
            resolution["by_id"].side_effect = None
            resolution["by_id"].return_value = [target]
            rows = await ingest_module.ingest_data(
                data="hello world",
                dataset_name="ds",
                user=USER,
                dataset_id=DATASET_ID,
                ctx=SimpleNamespace(dataset=other, user=USER),
            )
        resolution["by_id"].assert_awaited_once()
        assert rows[0].dataset_id == DATASET_ID
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_no_ctx_resolves_as_before():
    engine, db_path = await _make_engine()
    target = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    try:
        with ExitStack() as stack:
            resolution = _install_mocks(stack, engine)
            resolution["by_name"].side_effect = None
            resolution["by_name"].return_value = []
            resolution["create"].side_effect = None
            resolution["create"].return_value = target
            rows = await ingest_module.ingest_data(data="hello world", dataset_name="ds", user=USER)
        resolution["by_name"].assert_awaited_once()
        resolution["create"].assert_awaited_once()
        assert rows[0].dataset_id == DATASET_ID
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
