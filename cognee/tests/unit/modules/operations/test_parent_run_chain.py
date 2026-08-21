"""Parent edges mirror the token chain (pipeline-in-pipeline fix).

The session bridge runs a cognify pipeline INSIDE a memify pipeline. Tokens
chain cognify → memify → improve, so the parent_operation_id edges must
follow the same path — otherwise memify and cognify appear as siblings under
improve, each carrying the same chained totals, and summing a row's children
double-counts (observed live: improve 1120/3931 with two "sibling" children
of 1119/3930 each).

Runs against a real temporary SQLite database — no LLM, no network.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.operations import parent_run_scope, record_operation
from cognee.modules.pipelines.models.PipelineRun import PipelineRun

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")
complete_mod = importlib.import_module(
    "cognee.modules.pipelines.operations.log_pipeline_run_complete"
)
error_mod = importlib.import_module("cognee.modules.pipelines.operations.log_pipeline_run_error")


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="parent_chain_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    for module in (record_operation_mod, complete_mod, error_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _rows_by_run_id(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun))
        return {row.pipeline_run_id: row for row in result.scalars().all()}


@pytest.mark.asyncio
async def test_pipeline_in_pipeline_parents_to_enclosing_pipeline(ops_engine):
    """The session-bridge shape: operation → outer pipeline → inner pipeline."""
    memify_run_id = uuid4()
    cognify_run_id = uuid4()

    async with record_operation("improve") as improve_ctx:
        # run_tasks pushes parent_run_scope(pipeline_run_id) around task
        # execution; the terminal writers run inside that scope.
        with parent_run_scope(memify_run_id):
            # Inner pipeline started by a memify task (the "cognify session" task).
            with parent_run_scope(cognify_run_id):
                await error_mod.log_pipeline_run_error(
                    cognify_run_id, uuid4(), "cognify_pipeline", uuid4(), [], ValueError("x")
                )
            await complete_mod.log_pipeline_run_complete(
                memify_run_id, uuid4(), "memify_pipeline", uuid4(), []
            )

    rows = await _rows_by_run_id(ops_engine)

    # cognify parents to memify (NOT to improve): no more sibling double-count.
    assert rows[cognify_run_id].parent_operation_id == memify_run_id
    # memify parents to the improve operation (its own scope is excluded).
    assert rows[memify_run_id].parent_operation_id == improve_ctx.operation_id
    # improve is the root of this tree.
    assert rows[improve_ctx.operation_id].parent_operation_id is None


@pytest.mark.asyncio
async def test_operation_inside_pipeline_parents_to_that_pipeline(ops_engine):
    """A recorded operation called mid-pipeline (e.g. search in a custom task)."""
    pipeline_run_id = uuid4()

    with parent_run_scope(pipeline_run_id):
        async with record_operation("search") as ctx:
            pass

    rows = await _rows_by_run_id(ops_engine)
    assert rows[ctx.operation_id].parent_operation_id == pipeline_run_id


@pytest.mark.asyncio
async def test_nested_operations_still_chain(ops_engine):
    """Operation-in-operation linkage is unchanged by the new mechanism."""
    async with record_operation("remember") as outer:
        async with record_operation("improve") as inner:
            assert inner.parent_operation_id == outer.operation_id

    rows = await _rows_by_run_id(ops_engine)
    assert rows[inner.operation_id].parent_operation_id == outer.operation_id
    assert rows[outer.operation_id].parent_operation_id is None
