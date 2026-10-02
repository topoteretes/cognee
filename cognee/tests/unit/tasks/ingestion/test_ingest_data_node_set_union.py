"""ingest_data must actually write the call+item node_set union, not just
compute it: this exercises the real store_data_to_dataset path (the same
throwaway-SQLite harness as test_ingest_route_stamp_on_change.py), so a
regression that computes _union_node_sets correctly but forgets to wire it
into Data.node_set / external_metadata still gets caught.
"""

import importlib
import json
import os
import tempfile
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data, Dataset
from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.data_item import DataItem

ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()


def _metadata(content_hash):
    return {
        "name": "doc.txt",
        "file_path": "/tmp/doc.txt",
        "extension": "txt",
        "mime_type": "text/plain",
        "content_hash": content_hash,
        "file_size": 42,
    }


@asynccontextmanager
async def _fake_open_data_file(_path):
    yield object()


async def _fresh_engine():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")
    await engine.create_database()
    return engine, db_path


def _install_mocks(stack, engine, meta):
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
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    stack.enter_context(patch.object(ingest_module, "get_relational_engine", lambda: engine))
    stack.enter_context(
        patch.object(
            ingest_module,
            "save_data_item_to_storage_detailed",
            AsyncMock(return_value=StoredFile(file_path="/tmp/doc.txt")),
        )
    )
    stack.enter_context(patch.object(ingest_module, "get_data_file_path", lambda p: p))
    stack.enter_context(patch.object(ingest_module, "open_data_file", _fake_open_data_file))
    stack.enter_context(
        patch.object(
            ingest_module,
            "data_item_to_text_file",
            AsyncMock(return_value=("/tmp/doc.txt", SimpleNamespace(loader_name="text_loader"))),
        )
    )
    stack.enter_context(patch.object(ingest_module.ingestion, "classify", lambda _f: classified))
    stack.enter_context(patch.object(ingest_module, "identify_many", AsyncMock(return_value={})))
    stack.enter_context(
        patch.object(ingest_module, "get_authorized_existing_datasets", AsyncMock(return_value=[]))
    )
    stack.enter_context(
        patch.object(ingest_module, "load_or_create_datasets", AsyncMock(return_value=dataset))
    )


def _node_set(row):
    return json.loads(row.node_set) if row.node_set else None


@pytest.mark.asyncio
async def test_call_and_item_node_set_are_unioned_into_the_stored_row():
    engine, db_path = await _fresh_engine()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, _metadata("h1"))
            item = DataItem(data="plain text", node_set=["notion:ws:root"])
            await ingest_module.ingest_data(
                data=item, dataset_name="ds", user=USER, node_set=["call-level"]
            )

        async with engine.get_async_session() as session:
            rows = (await session.execute(Data.__table__.select())).fetchall()
            assert len(rows) == 1
            row = await session.get(Data, rows[0].id)
            assert _node_set(row) == ["call-level", "notion:ws:root"]
            assert row.external_metadata["node_set"] == ["call-level", "notion:ws:root"]
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_item_only_node_set_reaches_the_stored_row():
    engine, db_path = await _fresh_engine()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, _metadata("h2"))
            item = DataItem(data="plain text", node_set=["notion:ws:root"])
            await ingest_module.ingest_data(data=item, dataset_name="ds", user=USER)

        async with engine.get_async_session() as session:
            rows = (await session.execute(Data.__table__.select())).fetchall()
            row = await session.get(Data, rows[0].id)
            assert _node_set(row) == ["notion:ws:root"]
            assert row.external_metadata["node_set"] == ["notion:ws:root"]
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_two_items_in_one_call_keep_their_own_node_sets():
    engine, db_path = await _fresh_engine()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, _metadata("h3"))
            # Both items share the mocked content hash, so give them distinct ids.
            items = [
                DataItem(data="page one", data_id=uuid4(), node_set=["notion:ws:a"]),
                DataItem(data="page two", data_id=uuid4(), node_set=["notion:ws:b"]),
            ]
            await ingest_module.ingest_data(
                data=items, dataset_name="ds", user=USER, node_set=["call-level"]
            )

        async with engine.get_async_session() as session:
            rows = (await session.execute(Data.__table__.select())).fetchall()
            by_id = {row.id: await session.get(Data, row.id) for row in rows}
        assert {tuple(_node_set(row)) for row in by_id.values()} == {
            ("call-level", "notion:ws:a"),
            ("call-level", "notion:ws:b"),
        }
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_items_without_a_node_set_behave_as_before():
    engine, db_path = await _fresh_engine()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, _metadata("h4"))
            await ingest_module.ingest_data(
                data=DataItem(data="plain text"), dataset_name="ds", user=USER, node_set=["only"]
            )

        async with engine.get_async_session() as session:
            rows = (await session.execute(Data.__table__.select())).fetchall()
            row = await session.get(Data, rows[0].id)
            assert _node_set(row) == ["only"]
            assert row.external_metadata["node_set"] == ["only"]
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
