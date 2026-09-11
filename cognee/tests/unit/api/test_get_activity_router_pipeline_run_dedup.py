"""GET /activity/pipeline-runs must not read a finished run's STARTED row
as ABANDONED (Major review finding on PR #4983, follow-up to SDK-591).

log_pipeline_run_start/_complete/_error each INSERT a new PipelineRun row
sharing one pipeline_run_id; none ever UPDATEs. This endpoint applies no
dedup, so a finished run's STARTED row and its terminal row both come back
as separate results. get_effective_pipeline_status decides staleness per
row it is handed, so without telling it about the sibling, a run that
started 2 hours ago and completed 10 seconds later reported its STARTED row
as ABANDONED on every request after the 30-minute default threshold -- i.e.
on essentially every successful run in the table, not an edge case.

These tests run against the real relational engine rather than the fake
session in test_get_activity_router_pipeline_runs.py, because the fix lives
in the SQL: a correlated EXISTS finds a terminal sibling row regardless of
which page it lands on. A fake session that just replays canned rows cannot
exercise that, and the pagination test below exists specifically to catch a
"simplification" that swaps the EXISTS for a scan over the fetched page.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start

router_module = importlib.import_module("cognee.api.v1.activity.routers.get_activity_router")


def _client(user_id) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_activity_router(), prefix="/activity")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None, email="me@example.com"
    )
    return TestClient(app)


def _stub_visibility(monkeypatch, *, user_id):
    """Scope visibility to this test's own user_id, so leftover rows from
    other tests sharing the session-wide SQLite DB never show up here."""

    async def fake_visible_user_ids(_user_id):
        return [user_id]

    async def fake_permitted_dataset_ids(_user_id):
        return []

    monkeypatch.setattr(router_module, "get_visible_user_ids", fake_visible_user_ids)
    monkeypatch.setattr(router_module, "get_permitted_dataset_ids", fake_permitted_dataset_ids)


async def _backdate(pipeline_run_id, status, created_at):
    """Rewrite a row's created_at after the fact, the same way
    test_effective_status_reporting.py manufactures a stale row."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = (
            await session.execute(
                select(PipelineRun).filter(
                    PipelineRun.pipeline_run_id == pipeline_run_id,
                    PipelineRun.status == status,
                )
            )
        ).scalar_one()
        row.created_at = created_at
        await session.commit()


def _rows_for(body, pipeline_run_id):
    return [row for row in body if row["pipeline_run_id"] == str(pipeline_run_id)]


@pytest.mark.asyncio
async def test_completed_sibling_suppresses_abandoned_on_started_row(monkeypatch):
    """The exact reported shape: a run that started 2 hours ago and
    completed 10 seconds later must not show its STARTED row as ABANDONED."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "1800")
    await create_db_and_tables()

    user_id = uuid4()
    dataset_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None)
    started = await log_pipeline_run_start(uuid4(), "cognify_pipeline", dataset_id, None, user=user)
    await log_pipeline_run_complete(
        started.pipeline_run_id,
        started.pipeline_id,
        "cognify_pipeline",
        dataset_id,
        None,
        user=user,
    )

    started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    completed_at = started_at + timedelta(seconds=10)
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_STARTED, started_at
    )
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_COMPLETED, completed_at
    )

    _stub_visibility(monkeypatch, user_id=user_id)
    body = _client(user_id).get("/activity/pipeline-runs").json()

    rows = _rows_for(body, started.pipeline_run_id)
    statuses = {row["status"] for row in rows}
    assert statuses == {"DATASET_PROCESSING_STARTED", "DATASET_PROCESSING_COMPLETED"}
    assert "ABANDONED" not in statuses


@pytest.mark.asyncio
async def test_errored_sibling_suppresses_abandoned_on_started_row(monkeypatch):
    """Same as the completed case, for the other terminal status."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "1800")
    await create_db_and_tables()

    user_id = uuid4()
    dataset_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None)
    started = await log_pipeline_run_start(uuid4(), "cognify_pipeline", dataset_id, None, user=user)
    await log_pipeline_run_error(
        started.pipeline_run_id,
        started.pipeline_id,
        "cognify_pipeline",
        dataset_id,
        None,
        RuntimeError("boom"),
        user=user,
    )

    started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    errored_at = started_at + timedelta(seconds=10)
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_STARTED, started_at
    )
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_ERRORED, errored_at
    )

    _stub_visibility(monkeypatch, user_id=user_id)
    body = _client(user_id).get("/activity/pipeline-runs").json()

    rows = _rows_for(body, started.pipeline_run_id)
    statuses = {row["status"] for row in rows}
    assert statuses == {"DATASET_PROCESSING_STARTED", "DATASET_PROCESSING_ERRORED"}
    assert "ABANDONED" not in statuses


@pytest.mark.asyncio
async def test_genuinely_stale_run_with_no_terminal_row_still_reads_abandoned(monkeypatch):
    """A run with no terminal sibling at all is the original SDK-591 case --
    a crashed worker -- and must keep reading ABANDONED."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "60")
    await create_db_and_tables()

    user_id = uuid4()
    dataset_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None)
    started = await log_pipeline_run_start(uuid4(), "cognify_pipeline", dataset_id, None, user=user)
    await _backdate(
        started.pipeline_run_id,
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        datetime.now(timezone.utc) - timedelta(seconds=120),
    )

    _stub_visibility(monkeypatch, user_id=user_id)
    body = _client(user_id).get("/activity/pipeline-runs").json()

    rows = _rows_for(body, started.pipeline_run_id)
    assert [row["status"] for row in rows] == ["ABANDONED"]


@pytest.mark.asyncio
async def test_terminal_sibling_outside_the_requested_page_still_suppresses_abandoned(monkeypatch):
    """Regression guard for "simplify the EXISTS into a scan over the
    fetched rows": the COMPLETED row sorts ahead of the STARTED row (it is
    slightly newer), so requesting the page that contains only the STARTED
    row must still see the sibling and skip ABANDONED."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "1800")
    await create_db_and_tables()

    user_id = uuid4()
    dataset_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None)
    started = await log_pipeline_run_start(uuid4(), "cognify_pipeline", dataset_id, None, user=user)
    await log_pipeline_run_complete(
        started.pipeline_run_id,
        started.pipeline_id,
        "cognify_pipeline",
        dataset_id,
        None,
        user=user,
    )

    started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    completed_at = started_at + timedelta(seconds=10)
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_STARTED, started_at
    )
    await _backdate(
        started.pipeline_run_id, PipelineRunStatus.DATASET_PROCESSING_COMPLETED, completed_at
    )

    _stub_visibility(monkeypatch, user_id=user_id)
    # Ordered created_at desc: [COMPLETED, STARTED]. offset=1/limit=1 lands
    # on the STARTED row alone -- its terminal sibling is on the page before
    # this one, not in the result set the router actually returns.
    body = _client(user_id).get("/activity/pipeline-runs", params={"offset": 1, "limit": 1}).json()

    assert len(body) == 1
    assert body[0]["pipeline_run_id"] == str(started.pipeline_run_id)
    assert body[0]["status"] == "DATASET_PROCESSING_STARTED"
