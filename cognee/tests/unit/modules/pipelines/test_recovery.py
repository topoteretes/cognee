"""Startup recovery of pipeline runs abandoned by a crashed process (SDK-577).

Every test runs against its own tmp SQLite engine. The recovery sweep is
global by design, so a test that let it reach the ambient database would close
runs the developer cares about and execute the real cognify rollback against
their graph, while still reporting pass.

Covered here:
- a stale add run ends up terminal, with the dead run's own identity on the
  closing row, an error class a consumer can act on, and origin "background"
- one crash can abandon several runs of one pipeline, and all of them close
- a closed run is not reopened on the next boot
- attribution falls back to the dataset owner for rows that carry no user
- the summarized run_info is carried through rather than summarized again
- a stale cognify run is rolled back inside its dataset's database context,
  and a pipeline with no rollback policy never enters it
- a failing rollback leaves the run open for the next startup to retry, and a
  failing dataset lookup does not stop the sweep
- a missing dataset, and a run younger than the staleness threshold, are
  skipped
- the staleness threshold survives a misconfigured env var
"""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.users.models import User

recovery_module = importlib.import_module("cognee.modules.pipelines.recovery")
unclosed_module = importlib.import_module(
    "cognee.modules.pipelines.methods.get_unclosed_pipeline_runs"
)
log_error_module = importlib.import_module(
    "cognee.modules.pipelines.operations.log_pipeline_run_error"
)
run_by_dataset_module = importlib.import_module(
    "cognee.modules.pipelines.methods.get_pipeline_run_by_dataset"
)
get_pipeline_status = importlib.import_module(
    "cognee.modules.pipelines.operations.get_pipeline_status"
).get_pipeline_status

STARTED = PipelineRunStatus.DATASET_PROCESSING_STARTED
ERRORED = PipelineRunStatus.DATASET_PROCESSING_ERRORED


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest_asyncio.fixture
async def recovery_db(tmp_path, monkeypatch):
    """A SQLite engine of this test's own, plus recorded collaborators.

    Everything recovery reaches for an engine is pointed here, so the sweep
    stays global (which is what recovery does) without leaving the test.
    """
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recovery_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    # The whole metadata, not a subset: recovery reads a User, which is joined
    # onto principals, and the terminal write reaches for a few more.
    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    for module in (recovery_module, unclosed_module, log_error_module, run_by_dataset_module):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    calls = SimpleNamespace(rollbacks=[], dataset_contexts=[])

    async def _rollback_handler(**kwargs):
        calls.rollbacks.append(kwargs)

    @asynccontextmanager
    async def _dataset_context(dataset_id=None, owner_id=None, *_args, **_kwargs):
        calls.dataset_contexts.append((dataset_id, owner_id))
        yield

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _dataset_context)

    yield SimpleNamespace(engine=engine, calls=calls)

    await engine.engine.dispose()


async def _insert(engine, *rows):
    async with engine.get_async_session() as session:
        for row in rows:
            session.add(row)
        await session.commit()


async def _rows(engine, run_id=None, status=None):
    query = select(PipelineRun).order_by(PipelineRun.created_at, PipelineRun.id)
    if run_id is not None:
        query = query.filter(PipelineRun.pipeline_run_id == run_id)
    if status is not None:
        query = query.filter(PipelineRun.status == status)
    async with engine.get_async_session() as session:
        return (await session.execute(query)).scalars().all()


def _dataset(owner_id=None):
    return Dataset(id=uuid4(), name=f"ds_{uuid4().hex[:8]}", owner_id=owner_id or uuid4())


def _started_run(dataset_id, pipeline_name, hours_ago=2, user_id=None, run_info=None):
    started_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return PipelineRun(
        pipeline_run_id=uuid4(),
        pipeline_name=pipeline_name,
        pipeline_id=uuid4(),
        status=STARTED,
        dataset_id=dataset_id,
        run_info=run_info if run_info is not None else {},
        user_id=user_id,
        created_at=started_at,
        started_at=started_at,
    )


@pytest.mark.asyncio
async def test_a_stale_add_run_ends_up_terminal(recovery_db):
    """An add killed mid-flight keeps its dataset on STARTED forever, because
    only the next add to that dataset resets it. After recovery the dataset
    reports a terminal status, which is the only kind a poller stops on."""
    dataset = _dataset()
    stale_add = _started_run(dataset.id, "add_pipeline")
    await _insert(recovery_db.engine, dataset, stale_add)

    before = await get_pipeline_status([dataset.id], "add_pipeline")
    assert before[str(dataset.id)] == STARTED

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    after = await get_pipeline_status([dataset.id], "add_pipeline")
    assert after[str(dataset.id)] == ERRORED

    # add writes no graph, so nothing is unwound and the dataset's own
    # databases are never entered (entering would provision them).
    assert recovery_db.calls.rollbacks == []
    assert recovery_db.calls.dataset_contexts == []

    closed = await _rows(recovery_db.engine, run_id=stale_add.pipeline_run_id, status=ERRORED)
    assert len(closed) == 1
    # The closing row belongs to the run that died.
    assert closed[0].pipeline_id == stale_add.pipeline_id
    assert closed[0].started_at is not None
    assert closed[0].error_class == AbandonedPipelineRunError.__name__
    assert closed[0].outcome == "failed"
    # Nothing about this row came from a caller.
    assert closed[0].origin == "background"


@pytest.mark.asyncio
async def test_every_abandoned_run_of_one_pipeline_is_closed(recovery_db):
    """A background batch writes all its STARTED rows up front, so one crash
    leaves several runs of one (dataset, pipeline) open. Recovering only the
    newest would leave the rest open forever, and their partial graphs with
    them."""
    dataset = _dataset()
    runs = [_started_run(dataset.id, "cognify_pipeline", hours_ago=hours) for hours in (4, 3, 2)]
    await _insert(recovery_db.engine, dataset, *runs)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, status=ERRORED)
    assert {row.pipeline_run_id for row in closed} == {run.pipeline_run_id for run in runs}
    # Each one was unwound before it was closed.
    assert [call["pipeline_run_id"] for call in recovery_db.calls.rollbacks] == [
        run.pipeline_run_id for run in runs
    ]


@pytest.mark.asyncio
async def test_a_closed_run_is_not_reopened_on_the_next_boot(recovery_db):
    dataset = _dataset()
    stale_run = _started_run(dataset.id, "cognify_pipeline")
    await _insert(recovery_db.engine, dataset, stale_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()
    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, status=ERRORED)
    assert len(closed) == 1
    assert len(recovery_db.calls.rollbacks) == 1


@pytest.mark.asyncio
async def test_attribution_falls_back_to_the_dataset_owner(recovery_db):
    """Rows written before the user_id column existed, and by writers that pass
    no user, carry none. Those are exactly the rows still sitting open, so the
    closing row would lose attribution the old status reset used to stamp."""
    owner = User(id=uuid4(), email=f"{uuid4().hex[:8]}@example.com", hashed_password="x")
    owner.tenant_id = uuid4()
    dataset = _dataset(owner_id=owner.id)
    stale_run = _started_run(dataset.id, "add_pipeline", user_id=None)
    await _insert(recovery_db.engine, owner, dataset, stale_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, run_id=stale_run.pipeline_run_id, status=ERRORED)
    assert closed[0].user_id == owner.id
    assert closed[0].tenant_id == owner.tenant_id


@pytest.mark.asyncio
async def test_run_info_data_is_carried_through_untouched(recovery_db):
    """The STARTED row's run_info is already summarized. Summarizing it again
    stringifies the list of data ids and re-truncates a truncated preview with
    a wrong character count."""
    dataset = _dataset()
    data_ids = [str(uuid4()), str(uuid4())]
    stale_run = _started_run(dataset.id, "add_pipeline", run_info={"data": data_ids})
    await _insert(recovery_db.engine, dataset, stale_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, run_id=stale_run.pipeline_run_id, status=ERRORED)
    assert closed[0].run_info["data"] == data_ids


@pytest.mark.asyncio
async def test_a_stale_cognify_run_is_rolled_back_in_its_dataset_context(recovery_db):
    dataset = _dataset()
    stale_run = _started_run(dataset.id, "cognify_pipeline")
    await _insert(recovery_db.engine, dataset, stale_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert len(recovery_db.calls.rollbacks) == 1
    assert recovery_db.calls.rollbacks[0]["pipeline_run_id"] == stale_run.pipeline_run_id
    assert recovery_db.calls.rollbacks[0]["dataset"].id == dataset.id
    # The rollback touches the dataset's own graph and vector databases.
    assert recovery_db.calls.dataset_contexts == [(dataset.id, dataset.owner_id)]


@pytest.mark.asyncio
async def test_a_failing_rollback_leaves_the_run_open_and_continues(recovery_db, monkeypatch):
    """A rollback that fails must not close the run: the partial graph is still
    there, so the next startup has to try again. And it must not abandon the
    other open runs."""
    dataset = _dataset()
    cognify_run = _started_run(dataset.id, "cognify_pipeline", hours_ago=3)
    add_run = _started_run(dataset.id, "add_pipeline", hours_ago=2)
    await _insert(recovery_db.engine, dataset, cognify_run, add_run)

    async def _failing_rollback(**_kwargs):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _failing_rollback)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, status=ERRORED)
    assert [row.pipeline_run_id for row in closed] == [add_run.pipeline_run_id]


@pytest.mark.asyncio
async def test_a_dataset_lookup_failure_does_not_stop_the_sweep(recovery_db, monkeypatch):
    """The lookup runs inside the per-run guard, so a transient relational
    error costs that one run, not the whole sweep, and not the API startup the
    sweep is blocking."""
    dataset = _dataset()
    first = _started_run(dataset.id, "add_pipeline", hours_ago=3)
    second = _started_run(dataset.id, "memify_pipeline", hours_ago=2)
    await _insert(recovery_db.engine, dataset, first, second)

    real_loader = recovery_module._load_dataset_and_user
    seen = []

    async def _flaky_loader(pipeline_run):
        seen.append(pipeline_run.pipeline_run_id)
        if len(seen) == 1:
            raise RuntimeError("database is locked")
        return await real_loader(pipeline_run)

    monkeypatch.setattr(recovery_module, "_load_dataset_and_user", _flaky_loader)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    closed = await _rows(recovery_db.engine, status=ERRORED)
    assert [row.pipeline_run_id for row in closed] == [second.pipeline_run_id]


@pytest.mark.asyncio
async def test_a_missing_dataset_is_skipped(recovery_db):
    stale_run = _started_run(uuid4(), "cognify_pipeline")
    await _insert(recovery_db.engine, stale_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert await _rows(recovery_db.engine, status=ERRORED) == []
    assert recovery_db.calls.rollbacks == []


@pytest.mark.asyncio
async def test_a_recent_run_is_left_alone(recovery_db):
    """A STARTED run younger than the staleness threshold is left alone so a
    live run on another worker is not closed out from under it."""
    dataset = _dataset()
    recent_run = _started_run(dataset.id, "cognify_pipeline", hours_ago=0)
    await _insert(recovery_db.engine, dataset, recent_run)

    await recovery_module.recover_stale_pipeline_runs_on_startup()

    assert await _rows(recovery_db.engine, status=ERRORED) == []
    assert recovery_db.calls.rollbacks == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 3600),
        ("7200", 7200),
        # A typo must not stop the server: this module is imported from the
        # API lifespan, where a raising int() would.
        ("30m", 3600),
        # Nor may it turn recovery into a sweep that closes runs started
        # seconds ago.
        ("0", 60),
        ("-1", 60),
    ],
)
def test_the_staleness_threshold_survives_a_bad_env_var(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", raising=False)
    else:
        monkeypatch.setenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", raw)

    assert recovery_module._read_stale_run_min_age() == expected
