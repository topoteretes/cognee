"""``ingest_data`` must not repeat storage work the pipeline wrapper already did.

The incremental wrapper saves each item and hashes its bytes to resolve dedup
identity, publishing the result on ``ctx.extras``. Before the handoff, this
task paid the same upload (a duplicate PUT of the whole payload on S3) and the
same read-back per item. These tests drive the real ``ingest_data`` against a
real sqlite engine and assert on the storage calls themselves, so a regression
that quietly re-introduces the duplicate work fails here — not only in an S3
call-count benchmark.
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
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.modules.ingestion import StoredFile
from cognee.modules.pipelines.operations.run_tasks_data_item import INGEST_PRECOMPUTED_SOURCE

ingest_module = importlib.import_module("cognee.tasks.ingestion.ingest_data")

USER = SimpleNamespace(id=uuid4(), tenant_id=None)
DATASET_ID = uuid4()


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


def _install_mocks(stack, engine, save_mock, open_mock):
    stack.enter_context(patch.object(ingest_module, "get_relational_engine", lambda: engine))
    stack.enter_context(
        patch.object(ingest_module, "save_data_item_to_storage_detailed", save_mock)
    )
    stack.enter_context(patch.object(ingest_module, "get_data_file_path", lambda p: p))
    stack.enter_context(patch.object(ingest_module, "open_data_file", open_mock))
    stack.enter_context(
        patch.object(
            ingest_module,
            "data_item_to_text_file",
            AsyncMock(
                return_value=(
                    # Loaders describe the text they wrote (file_metadata), so
                    # ingest_data must not read the derived file back either.
                    LoaderResult(file_path="/tmp/doc.txt", file_metadata=_metadata()),
                    SimpleNamespace(loader_name="text_loader"),
                )
            ),
        )
    )


async def _run_ingest(data_item, ctx, save_mock, open_mock):
    engine, db_path = await _make_engine()
    dataset = Dataset(id=DATASET_ID, name="ds", owner_id=USER.id)
    ctx.dataset = dataset
    ctx.user = USER
    try:
        with ExitStack() as stack:
            _install_mocks(stack, engine, save_mock, open_mock)
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
async def test_carried_item_is_not_saved_or_read_again():
    data_item = object.__new__(object)  # identity is what the handoff keys on
    ctx = SimpleNamespace(
        extras={
            INGEST_PRECOMPUTED_SOURCE: {
                "data_item_id": id(data_item),
                "file_path": "/tmp/doc.txt",
                "metadata": _metadata(),
            }
        }
    )
    save_mock = AsyncMock(side_effect=AssertionError("item was uploaded a second time"))
    open_mock = lambda *_a, **_k: (_ for _ in ()).throw(  # noqa: E731
        AssertionError("item was read back from storage")
    )

    stored = await _run_ingest(data_item, ctx, save_mock, open_mock)

    assert stored is not None
    assert stored.content_hash == "hash-1"
    assert stored.data_size == 11


@pytest.mark.asyncio
async def test_path_item_reuses_wrapper_metadata_after_identity_changes():
    # The in-chain resolve_data_directories re-creates string paths, so the
    # id() published by the wrapper no longer matches — but the stored path
    # does, and the metadata must be reused rather than the file re-read.
    wrapper_saw = "/tmp/doc.txt"  # the string the wrapper inspected
    # Built at runtime: a compile-time concat would be constant-folded into the
    # same interned object, silently reintroducing the id() match.
    task_receives = "".join(["/tmp/doc", ".txt"])
    assert wrapper_saw == task_receives and wrapper_saw is not task_receives

    ctx = SimpleNamespace(
        extras={
            INGEST_PRECOMPUTED_SOURCE: {
                "data_item_id": id(wrapper_saw),
                "file_path": "/tmp/doc.txt",
                "metadata": _metadata(),
            }
        }
    )
    # The pass-through save is I/O-free and returns the path with no metadata.
    save_mock = AsyncMock(return_value=StoredFile(file_path="/tmp/doc.txt", metadata=None))
    open_mock = lambda *_a, **_k: (_ for _ in ()).throw(  # noqa: E731
        AssertionError("metadata was recomputed by re-reading the file")
    )

    stored = await _run_ingest(task_receives, ctx, save_mock, open_mock)

    assert stored is not None
    assert stored.content_hash == "hash-1"
    save_mock.assert_awaited_once()
