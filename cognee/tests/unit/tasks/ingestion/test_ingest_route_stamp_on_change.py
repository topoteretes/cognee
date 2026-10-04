"""The route stamp follows the content when an existing row is refreshed.

``Data.system_metadata`` is the cognify route stamp of the content (a DLT
manifest, a code file). update()'s full rebuild refreshes the row in place,
so a replacement of another kind must not inherit the old stamp: text
replacing a DLT manifest kept ``source: dlt_source`` and was then read as a
manifest by every later cognify of the dataset. The update branch of
``ingest_data`` now takes the stamp from the new item when the content
changed, including none; unchanged content keeps what it has.

Same harness as test_ingest_update_data_size: the real update branch against
a throwaway SQLite engine, every other collaborator stubbed.
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

ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()
DATA_ID = uuid4()
DLT_STAMP = {"source": "dlt_source", "source_name": "people", "row_count": 2}


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


async def _engine_with_stamped_row():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")
    await engine.create_database()
    async with engine.get_async_session() as session:
        session.add(
            Data(
                id=DATA_ID,
                name="doc.txt",
                content_hash="old-hash",
                data_size=1,
                dataset_id=DATASET_ID,
                system_metadata=DLT_STAMP,
            )
        )
        await session.commit()
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
    stack.enter_context(
        patch.object(
            ingest_module,
            "identify_many",
            AsyncMock(return_value={meta["content_hash"]: DATA_ID}),
        )
    )
    stack.enter_context(
        patch.object(ingest_module, "get_authorized_existing_datasets", AsyncMock(return_value=[]))
    )
    stack.enter_context(
        patch.object(ingest_module, "load_or_create_datasets", AsyncMock(return_value=dataset))
    )


def _stamp(row):
    metadata = row.system_metadata
    return json.loads(metadata) if isinstance(metadata, str) else metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("new_hash", "expected_stamp"),
    [
        ("new-hash", None),  # text replacing a manifest: the DLT stamp goes
        ("old-hash", DLT_STAMP),  # identical content re-added: nothing changes
    ],
    ids=["content-changed-clears-stamp", "content-unchanged-keeps-stamp"],
)
async def test_route_stamp_follows_the_content(new_hash, expected_stamp):
    engine, db_path = await _engine_with_stamped_row()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, _metadata(new_hash))
            await ingest_module.ingest_data(data="plain text", dataset_name="ds", user=USER)

        async with engine.get_async_session() as session:
            refreshed = await session.get(Data, DATA_ID)
            assert _stamp(refreshed) == expected_stamp, _stamp(refreshed)
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
