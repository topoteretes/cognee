"""``DataItem.literal_text`` must stop URL/path sniffing at the storage boundary.

A document synced from an external source (e.g. an untitled Notion page) can
have content that is just a URL or a path that happens to exist on the host.
``save_data_item_to_storage_detailed`` would fetch such a URL or read such a
path instead of storing the string; ``literal_text=True`` routes the item
straight to ``save_data_to_file_detailed`` instead, which always stores its
argument as text. These tests drive real ``ingest_data`` against a real
sqlite engine and assert on which storage function actually ran.
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


async def _run_ingest(data_item, item_save_mock, text_save_mock):
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
            stack.enter_context(
                patch.object(ingest_module, "save_data_to_file_detailed", text_save_mock)
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


@pytest.mark.asyncio
async def test_literal_text_url_content_is_stored_as_text_not_fetched():
    content = "https://example.com/x"
    item_save_mock = AsyncMock(side_effect=AssertionError("URL was fetched"))
    text_save_mock = AsyncMock(
        return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata())
    )

    await _run_ingest(DataItem(data=content, literal_text=True), item_save_mock, text_save_mock)

    item_save_mock.assert_not_called()
    text_save_mock.assert_awaited_once_with(content)


@pytest.mark.asyncio
async def test_literal_text_existing_path_content_is_stored_as_text_not_read(tmp_path):
    existing_file = tmp_path / "secret.txt"
    existing_file.write_text("do not leak this", encoding="utf-8")
    content = str(existing_file)

    item_save_mock = AsyncMock(side_effect=AssertionError("local file was read"))
    text_save_mock = AsyncMock(
        return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata())
    )

    await _run_ingest(DataItem(data=content, literal_text=True), item_save_mock, text_save_mock)

    item_save_mock.assert_not_called()
    text_save_mock.assert_awaited_once_with(content)


@pytest.mark.asyncio
async def test_literal_text_s3_content_is_stored_as_text_not_treated_as_s3_path():
    content = "s3://some-bucket/some-key"
    item_save_mock = AsyncMock(side_effect=AssertionError("s3 path was handed through"))
    text_save_mock = AsyncMock(
        return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata())
    )

    await _run_ingest(DataItem(data=content, literal_text=True), item_save_mock, text_save_mock)

    item_save_mock.assert_not_called()
    text_save_mock.assert_awaited_once_with(content)


@pytest.mark.asyncio
async def test_literal_text_file_uri_content_is_stored_as_text_not_read():
    content = "file:///etc/passwd"
    item_save_mock = AsyncMock(side_effect=AssertionError("file:// URI was resolved"))
    text_save_mock = AsyncMock(
        return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata())
    )

    await _run_ingest(DataItem(data=content, literal_text=True), item_save_mock, text_save_mock)

    item_save_mock.assert_not_called()
    text_save_mock.assert_awaited_once_with(content)


@pytest.mark.asyncio
async def test_titled_row_without_literal_text_takes_the_normal_storage_path():
    # A titled row's rendered text ("# Title\n\n...") never looks like a URL or
    # path, so it already fell through to save_data_to_file_detailed before
    # this fix. Default literal_text=False must keep routing through the
    # ordinary save_data_item_to_storage_detailed entry point unchanged.
    content = "# My Page\n\nbody text"
    item_save_mock = AsyncMock(
        return_value=StoredFile(file_path="/tmp/doc.txt", metadata=_metadata())
    )
    text_save_mock = AsyncMock(side_effect=AssertionError("literal text path was used"))

    await _run_ingest(DataItem(data=content), item_save_mock, text_save_mock)

    text_save_mock.assert_not_called()
    item_save_mock.assert_awaited_once_with(content)
