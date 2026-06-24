"""Regression test for the re-ingestion `data_size` update path.

When an already-ingested file is added again with a different size, the update
branch of ``store_data_to_dataset`` (in ``cognee/tasks/ingestion/ingest_data.py``)
must write the new size to the ``Data.data_size`` column. A prior version wrote
to ``data_point.file_size`` — an attribute the ``Data`` model does not declare —
so SQLAlchemy's unit-of-work never flushed it and the persisted ``data_size``
stayed stale.

This test drives the real ``ingest_data`` update branch against an in-memory
SQLite relational engine and asserts the persisted ``data_size`` reflects the
new value. It fails before the fix and passes after it.
"""

import importlib
import os
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data, Dataset

# Import the module object explicitly: `cognee.tasks.ingestion.__init__` re-exports
# `ingest_data` (the function), which shadows the submodule of the same name, so
# `import ... as ingest_module` would bind the function and break patch.object.
ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()
DATA_ID = uuid4()

OLD_SIZE = 100
NEW_SIZE = 200


def _metadata():
    """Metadata returned by the loader for the re-ingested file (new size)."""
    return {
        "name": "doc.txt",
        "file_path": "/tmp/doc.txt",
        "extension": "txt",
        "mime_type": "text/plain",
        "content_hash": "new-hash",
        "file_size": NEW_SIZE,
    }


@asynccontextmanager
async def _fake_open_data_file(_path):
    yield object()


async def _make_engine():
    """Create a throwaway SQLite-backed relational engine and seed one existing
    Data row whose data_size is OLD_SIZE."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()

    async with engine.get_async_session() as session:
        session.add(
            Data(
                id=DATA_ID,
                name="doc.txt",
                content_hash="old-hash",
                data_size=OLD_SIZE,
            )
        )
        await session.commit()

    return engine, tmp.name


def _install_mocks(stack, engine):
    """Patch ingest_data's collaborators so the update branch runs without real
    storage, loaders, or dataset/permission machinery — only the relational
    engine is real."""
    meta = _metadata()
    classified = SimpleNamespace(get_metadata=lambda: meta)
    # A real (transient) mapped Dataset: store_data_to_dataset does `dataset in
    # session` / session.add / session.merge, which require a mapped instance.
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)

    stack.enter_context(patch.object(ingest_module, "get_relational_engine", lambda: engine))
    stack.enter_context(
        patch.object(
            ingest_module, "save_data_item_to_storage", AsyncMock(return_value="/tmp/doc.txt")
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
        patch.object(ingest_module.ingestion, "identify", AsyncMock(return_value=DATA_ID))
    )
    # Dataset resolution: return the seeded dataset and report DATA_ID as already
    # belonging to it, so the existing-record UPDATE branch (not CREATE) runs.
    stack.enter_context(
        patch.object(ingest_module, "get_authorized_existing_datasets", AsyncMock(return_value=[]))
    )
    stack.enter_context(
        patch.object(ingest_module, "load_or_create_datasets", AsyncMock(return_value=dataset))
    )
    stack.enter_context(
        patch.object(
            ingest_module,
            "get_dataset_data",
            AsyncMock(return_value=[SimpleNamespace(id=DATA_ID)]),
        )
    )


@pytest.mark.asyncio
async def test_reingest_updates_persisted_data_size():
    from contextlib import ExitStack

    engine, db_path = await _make_engine()
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine)
            await ingest_module.ingest_data(
                data="hello world",
                dataset_name="ds",
                user=USER,
            )

        async with engine.get_async_session() as session:
            refreshed = await session.get(Data, DATA_ID)
            assert refreshed.data_size == NEW_SIZE, (
                f"persisted data_size should be {NEW_SIZE} after re-ingestion, "
                f"got {refreshed.data_size}"
            )
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
