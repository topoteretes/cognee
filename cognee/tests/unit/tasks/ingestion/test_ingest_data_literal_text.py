"""``DataItem.literal_text`` must stop URL/path sniffing at the storage boundary.

A document synced from an external source (e.g. an untitled Notion page) can
have content that is just a URL or a path that happens to exist on the host.
``ingest_data`` hands the whole ``DataItem`` to
``save_data_item_to_storage_detailed``, which stores a ``literal_text`` item
verbatim (``test_save_data_item_literal_text.py``). These tests drive real
``ingest_data`` against a real sqlite engine and assert that the flag reaches
the storage function instead of being unwrapped away.
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
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.modules.data.models import Data, Dataset
from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.data_item import DataItem

ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()


def _metadata(content_hash="hash-1"):
    return {
        "name": "doc",
        "file_path": "/tmp/doc.txt",
        "mime_type": "text/plain",
        "extension": "txt",
        "content_hash": content_hash,
        "file_size": 11,
    }


async def _make_engine():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")
    await engine.create_database()
    return engine, db_path


async def _run_ingest(data_item, item_save_mock):
    engine, db_path = await _make_engine()
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    ctx = SimpleNamespace(dataset=dataset, user=USER, extras={})
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ingest_module, "get_relational_engine", lambda: engine)
            )
            stack.enter_context(
                patch.object(ingest_module, "save_data_item_to_storage_detailed", item_save_mock)
            )
            stack.enter_context(patch.object(ingest_module, "get_data_file_path", lambda p: p))
            stack.enter_context(
                patch.object(
                    ingest_module,
                    "data_item_to_text_file",
                    AsyncMock(
                        return_value=(
                            LoaderResult(file_path="/tmp/doc.txt", file_metadata=_metadata()),
                            SimpleNamespace(loader_name="text_loader"),
                        )
                    ),
                )
            )
            rows = await ingest_module.ingest_data(
                data=data_item,
                dataset_name="ds",
                user=USER,
                dataset_id=DATASET_ID,
                ctx=ctx,
            )
        async with engine.get_async_session() as session:
            stored = await session.get(Data, rows[0].id)
        return stored
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


def _stored():
    return AsyncMock(return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["https://example.com/x", "s3://some-bucket/some-key", "file:///etc/passwd"],
)
async def test_literal_text_item_reaches_storage_with_its_flag(content):
    item = DataItem(data=content, literal_text=True)
    item_save_mock = _stored()

    await _run_ingest(item, item_save_mock)

    item_save_mock.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_literal_text_existing_path_reaches_storage_with_its_flag(tmp_path):
    existing_file = tmp_path / "secret.txt"
    existing_file.write_text("do not leak this", encoding="utf-8")
    item = DataItem(data=str(existing_file), literal_text=True)
    item_save_mock = _stored()

    await _run_ingest(item, item_save_mock)

    item_save_mock.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_item_without_literal_text_takes_the_same_storage_entry_point():
    item = DataItem(data="# My Page\n\nbody text")
    item_save_mock = _stored()

    await _run_ingest(item, item_save_mock)

    item_save_mock.assert_awaited_once_with(item)
