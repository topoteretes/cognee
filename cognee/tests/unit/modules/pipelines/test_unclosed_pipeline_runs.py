"""The unclosed-run lookup startup recovery selects from (SDK-577).

Recovery cannot use the latest-run-per-dataset lookups next door: those answer
a status question, where only the newest row matters, and one crash can leave
several runs of the same dataset and pipeline open at once. Covered here
against a real (tmp sqlite) engine:
- every open run comes back, including several of the same (dataset, pipeline)
- a run with a terminal row of its own is not open, whichever terminal it is
- a run is returned once even when it carries two STARTED rows
- operation records, which carry no pipeline_name, stay out
- dataset scoping, including the empty-list short circuit
"""

import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

# The methods package rebinds the submodule name to the function it exports,
# so the module itself has to come from importlib for monkeypatch to reach its
# globals (same gotcha as the graph-counts and record_operation tests).
unclosed_module = importlib.import_module(
    "cognee.modules.pipelines.methods.get_unclosed_pipeline_runs"
)
get_unclosed_pipeline_runs = unclosed_module.get_unclosed_pipeline_runs


@pytest_asyncio.fixture
async def runs_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="unclosed_runs_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    monkeypatch.setattr(unclosed_module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _insert(engine, *rows):
    async with engine.get_async_session() as session:
        for row in rows:
            session.add(row)
        await session.commit()


def _row(dataset_id, pipeline_name, status, run_id, minutes_ago=0):
    return PipelineRun(
        pipeline_run_id=run_id,
        pipeline_name=pipeline_name,
        pipeline_id=uuid4(),
        status=status,
        dataset_id=dataset_id,
        run_info={},
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


@pytest.mark.asyncio
async def test_every_open_run_of_the_same_pipeline_comes_back(runs_engine):
    """A background batch writes all its STARTED rows up front, so one crash
    abandons several runs of one (dataset, pipeline). Ranking by recency would
    hide all but the newest and leave the rest open forever."""
    dataset_id = uuid4()
    first, second, third = uuid4(), uuid4(), uuid4()
    await _insert(
        runs_engine,
        _row(
            dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, first, 30
        ),
        _row(
            dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, second, 20
        ),
        _row(
            dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, third, 10
        ),
    )

    runs = await get_unclosed_pipeline_runs()

    # Oldest first: recovery closes them in the order they were abandoned.
    assert [run.pipeline_run_id for run in runs] == [first, second, third]


@pytest.mark.asyncio
async def test_a_run_with_a_terminal_row_is_closed(runs_engine):
    dataset_id = uuid4()
    completed, errored, open_run = uuid4(), uuid4(), uuid4()
    await _insert(
        runs_engine,
        _row(
            dataset_id, "add_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, completed, 30
        ),
        _row(
            dataset_id,
            "add_pipeline",
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            completed,
            29,
        ),
        _row(
            dataset_id,
            "cognify_pipeline",
            PipelineRunStatus.DATASET_PROCESSING_STARTED,
            errored,
            20,
        ),
        _row(
            dataset_id,
            "cognify_pipeline",
            PipelineRunStatus.DATASET_PROCESSING_ERRORED,
            errored,
            19,
        ),
        _row(
            dataset_id,
            "memify_pipeline",
            PipelineRunStatus.DATASET_PROCESSING_STARTED,
            open_run,
            10,
        ),
    )

    runs = await get_unclosed_pipeline_runs()

    assert [run.pipeline_run_id for run in runs] == [open_run]


@pytest.mark.asyncio
async def test_a_run_with_two_started_rows_is_returned_once(runs_engine):
    """log_pipeline_run_progress can insert a second STARTED row for a run, and
    a caller closing it wants to see it once, not twice."""
    dataset_id = uuid4()
    run_id = uuid4()
    await _insert(
        runs_engine,
        _row(
            dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, run_id, 30
        ),
        _row(
            dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, run_id, 20
        ),
    )

    runs = await get_unclosed_pipeline_runs()

    assert [run.pipeline_run_id for run in runs] == [run_id]


@pytest.mark.asyncio
async def test_operation_records_are_not_runs(runs_engine):
    """record_operation writes one NULL-status, NULL-pipeline row per search /
    recall / forget. Those are activity evidence, not pipeline runs."""
    dataset_id = uuid4()
    open_run = uuid4()
    await _insert(
        runs_engine,
        _row(
            dataset_id, "add_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, open_run, 10
        ),
        PipelineRun(
            pipeline_run_id=uuid4(),
            pipeline_name=None,
            pipeline_id=None,
            status=None,
            dataset_id=dataset_id,
            operation_name="search",
            outcome="succeeded",
        ),
    )

    runs = await get_unclosed_pipeline_runs()

    assert [run.pipeline_run_id for run in runs] == [open_run]


@pytest.mark.asyncio
async def test_dataset_scoping(runs_engine):
    wanted_dataset, other_dataset = uuid4(), uuid4()
    wanted, other = uuid4(), uuid4()
    await _insert(
        runs_engine,
        _row(
            wanted_dataset, "add_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, wanted, 10
        ),
        _row(
            other_dataset, "add_pipeline", PipelineRunStatus.DATASET_PROCESSING_STARTED, other, 10
        ),
    )

    scoped = await get_unclosed_pipeline_runs([wanted_dataset])
    assert [run.pipeline_run_id for run in scoped] == [wanted]

    # No dataset filter covers every dataset, the way startup recovery calls it.
    unscoped = await get_unclosed_pipeline_runs()
    assert {run.pipeline_run_id for run in unscoped} == {wanted, other}

    # An empty list scopes to nothing, rather than falling back to everything.
    assert await get_unclosed_pipeline_runs([]) == []
