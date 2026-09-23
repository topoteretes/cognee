"""End-to-end wiring tests for operation recording (SDK-399).

Unlike test_record_operation.py (which exercises the recorder directly),
these call the real public entry points, so a regression that removes or
bypasses the ``record_operation`` wrapper inside an operation fails here
even though the recorder itself still works.

Runs against a real temporary SQLite database — no LLM, no network.
"""

import importlib

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.pipelines.models.PipelineRun import PipelineRun

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="ops_wiring_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _fetch_rows(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun).order_by(PipelineRun.created_at))
        return result.scalars().all()


@pytest.mark.asyncio
async def test_search_failure_records_row_through_public_entry_point(ops_engine):
    """A failing search() leaves a durable failed record, and the error propagates."""
    from cognee.api.v1.search.search import search

    with pytest.raises(CogneeValidationError):
        await search(query_text="anything", node_name_filter_operator="NEITHER")

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    row = rows[0]

    assert row.operation_name == "search"
    assert row.outcome == "failed"
    assert row.error_class == "CogneeValidationError"
    assert row.started_at is not None
    assert row.ended_at is not None
    # Operation-level record, not a pipeline run.
    assert row.status is None
    assert row.pipeline_name is None
    assert row.run_info is None


@pytest.mark.asyncio
async def test_prune_data_records_succeeded_row_through_public_entry_point(
    ops_engine, monkeypatch, tmp_path
):
    """A successful non-pipeline operation leaves a single succeeded record."""
    prune_data_mod = importlib.import_module("cognee.modules.data.deletion.prune_data")

    class _NoopStorage:
        async def remove_all(self):
            return None

    async def _noop_close():
        return None

    monkeypatch.setattr(
        prune_data_mod,
        "get_storage_config",
        lambda: {"data_root_directory": str(tmp_path / "data_root")},
    )
    monkeypatch.setattr(prune_data_mod, "get_file_storage", lambda _root: _NoopStorage())
    monkeypatch.setattr(prune_data_mod, "close_cache_engine", _noop_close)

    await prune_data_mod.prune_data()

    rows = await _fetch_rows(ops_engine)
    assert len(rows) == 1
    row = rows[0]

    assert row.operation_name == "prune_data"
    assert row.outcome == "succeeded"
    assert row.error_class is None
    assert row.started_at is not None
    assert row.ended_at is not None
    assert row.started_at <= row.ended_at
    assert row.status is None
