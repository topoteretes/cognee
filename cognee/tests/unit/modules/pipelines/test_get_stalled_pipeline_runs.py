"""Tests for the stalled pipeline run report.

The report has to make exactly one distinction correctly: a run that is slow
but advancing is healthy, a run whose heartbeat has frozen is not. Age is not
usable as a proxy for either, so these cases all use runs that started days
ago and differ only in when they last reported progress.
"""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

# methods/__init__ re-exports the function under the module's own name, so a
# plain `import ... as mod` binds the function and any patch on it is silently
# lost. Go through importlib to get the module itself.
stalled_module = importlib.import_module(
    "cognee.modules.pipelines.methods.get_stalled_pipeline_runs"
)


@asynccontextmanager
async def sqlite_engine(tmp_path):
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'stalled_test.db'}")
    await adapter.create_database()
    try:
        yield adapter
    finally:
        await adapter.engine.dispose()


def _run(run_id, dataset_id, status, created_at, heartbeat=None, name="cognify_pipeline"):
    return PipelineRun(
        pipeline_run_id=run_id,
        pipeline_name=name,
        pipeline_id=uuid4(),
        dataset_id=dataset_id,
        status=status,
        created_at=created_at,
        last_heartbeat_at=heartbeat,
        run_info={},
    )


@pytest.mark.asyncio
async def test_report_separates_advancing_runs_from_frozen_ones(monkeypatch, tmp_path):
    async with sqlite_engine(tmp_path) as engine:
        monkeypatch.setattr(stalled_module, "get_relational_engine", lambda: engine)

        now = datetime.now(timezone.utc)
        long_ago = now - timedelta(days=3)

        advancing_id, wedged_id, finished_id, legacy_id = (uuid4() for _ in range(4))
        advancing_ds, wedged_ds, finished_ds, legacy_ds = (uuid4() for _ in range(4))

        started = PipelineRunStatus.DATASET_PROCESSING_STARTED

        async with engine.get_async_session() as session:
            # Three days old and still reporting progress: healthy.
            session.add(
                _run(advancing_id, advancing_ds, started, long_ago, now - timedelta(seconds=30))
            )
            # Three days old, silent for six hours: stalled.
            session.add(_run(wedged_id, wedged_ds, started, long_ago, now - timedelta(hours=6)))
            # Old STARTED row superseded by a terminal row: not in flight at all.
            session.add(_run(finished_id, finished_ds, started, long_ago))
            session.add(
                _run(
                    finished_id,
                    finished_ds,
                    PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                    now - timedelta(hours=2),
                )
            )
            # No heartbeat at all: falls back to created_at, which is stale.
            session.add(_run(legacy_id, legacy_ds, started, long_ago))
            await session.commit()

        stalled = await stalled_module.get_stalled_pipeline_runs(idle_seconds=3600)
        reported = {run.pipeline_run_id for run in stalled}

        assert wedged_id in reported
        assert legacy_id in reported
        assert advancing_id not in reported, "a run still completing tasks is not stalled"
        assert finished_id not in reported, "a completed run is not in flight"
        assert len(reported) == 2


@pytest.mark.asyncio
async def test_report_honours_filters_and_window(monkeypatch, tmp_path):
    async with sqlite_engine(tmp_path) as engine:
        monkeypatch.setattr(stalled_module, "get_relational_engine", lambda: engine)

        now = datetime.now(timezone.utc)
        long_ago = now - timedelta(days=3)
        started = PipelineRunStatus.DATASET_PROCESSING_STARTED

        cognify_id, other_id = uuid4(), uuid4()
        cognify_ds, other_ds = uuid4(), uuid4()

        async with engine.get_async_session() as session:
            session.add(_run(cognify_id, cognify_ds, started, long_ago))
            session.add(_run(other_id, other_ds, started, long_ago, name="other_pipeline"))
            await session.commit()

        by_dataset = await stalled_module.get_stalled_pipeline_runs(
            idle_seconds=3600, dataset_ids=[cognify_ds]
        )
        assert {run.pipeline_run_id for run in by_dataset} == {cognify_id}

        by_name = await stalled_module.get_stalled_pipeline_runs(
            idle_seconds=3600, pipeline_name="other_pipeline"
        )
        assert {run.pipeline_run_id for run in by_name} == {other_id}

        # A window wider than the runs' silence reports nothing.
        assert await stalled_module.get_stalled_pipeline_runs(idle_seconds=10 * 24 * 3600) == []
