"""Improve watermarks are read from succeeded, stamped improve rows.

Only an improve row whose ``run_info`` carries the stage's stamp counts, the
stamp lookup filters in SQL on SQLite and Postgres, and failed, noop and other
datasets' rows are never watermarks.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.improve import watermarks
from cognee.modules.pipelines.models import PipelineRunStatus


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalar_one(self):
        return self._value

    def first(self):
        return self._value[0] if self._value else None


class _Session:
    def __init__(self, answers):
        self._answers = list(answers)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        answer = self._answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return _Result(answer)


def _install_engine(monkeypatch, answers):
    session = _Session(answers)

    @asynccontextmanager
    async def get_async_session():
        yield session

    engine = SimpleNamespace(get_async_session=get_async_session)
    relational = __import__("cognee.infrastructure.databases.relational", fromlist=["x"])
    monkeypatch.setattr(relational, "get_relational_engine", lambda: engine)
    return session


@pytest.mark.parametrize("key", [watermarks.REVIEW_WATERMARK_KEY])
@pytest.mark.asyncio
async def test_stamp_lookup_filters_in_sql_for_sqlite_and_postgres(monkeypatch, key):
    from sqlalchemy.dialects import postgresql, sqlite

    session = _install_engine(monkeypatch, [[]])
    assert await watermarks.last_improve_watermark(uuid4(), key) is None
    statement = session.statements[0]
    sqlite_sql = str(
        statement.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True})
    )
    postgres_sql = str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )
    assert "JSON_EXTRACT" in sqlite_sql and key in sqlite_sql
    assert "->>" in postgres_sql and key in postgres_sql
    assert "IS NOT NULL" in sqlite_sql and "IS NOT NULL" in postgres_sql
    assert "LIMIT 1" in sqlite_sql and "LIMIT 1" in postgres_sql


@pytest.mark.parametrize("key", [watermarks.REVIEW_WATERMARK_KEY])
@pytest.mark.asyncio
async def test_stamp_survives_long_history_and_excludes_failed_rows(monkeypatch, tmp_path, key):
    import importlib
    from datetime import timedelta

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from cognee.modules.pipelines.models import PipelineRun

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/watermarks.db")
    session_factory = async_sessionmaker(engine)
    relational = importlib.import_module("cognee.infrastructure.databases.relational")
    monkeypatch.setattr(
        relational,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    dataset_id, operation_id, current_id = uuid4(), uuid4(), uuid4()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def row(offset, **values):
        return PipelineRun(
            pipeline_run_id=values.pop("pipeline_run_id", uuid4()),
            dataset_id=values.pop("dataset_id", dataset_id),
            operation_name="improve",
            outcome=values.pop("outcome", "succeeded"),
            status=None,
            ended_at=start + timedelta(seconds=offset),
            **values,
        )

    try:
        async with engine.begin() as connection:
            await connection.run_sync(PipelineRun.__table__.create)
        async with session_factory() as session:
            session.add(
                row(
                    0,
                    pipeline_run_id=operation_id,
                    run_info=watermarks.watermark_stamp(key, "completed", start),
                )
            )
            session.add_all(row(i) for i in range(1, 102))
            for offset, fields in enumerate(
                [
                    {"outcome": "failed"},
                    {"outcome": "noop"},
                    {"dataset_id": uuid4()},
                    {"pipeline_run_id": current_id},
                ],
                102,
            ):
                session.add(
                    row(
                        offset,
                        run_info=watermarks.watermark_stamp(key, "completed", start),
                        **fields,
                    )
                )
            await session.commit()
        watermark = await watermarks.last_improve_watermark(
            dataset_id, key, exclude_operation_id=current_id
        )
        assert (watermark.operation_id, watermark.started_at) == (operation_id, start)
        async with session_factory() as session:
            session.add(row(200, run_info={key: {"started_at": "not-a-date"}}))
            await session.commit()
        # An invalid newest stamp cannot borrow an older success to skip work.
        assert (
            await watermarks.last_improve_watermark(
                dataset_id, key, exclude_operation_id=current_id
            )
            is None
        )
    finally:
        await engine.dispose()


# --- the global context index's stricter check -------------------------------

STAMPED_RUN = uuid4()
OWN_BUILD = uuid4()  # a pipeline run the stamped run started
STAMP_START = datetime(2025, 1, 1, tzinfo=timezone.utc)


@asynccontextmanager
async def _pipeline_runs_on_sqlite(monkeypatch, tmp_path, dataset_id, rows):
    """A real pipeline_runs table: the run that stamped ``dataset_id``, then ``rows``.

    Every row is created now, after the stamp -- the stamped run's own row too,
    since a run's row is written when the run ends.
    """
    import importlib

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from cognee.modules.pipelines.models import PipelineRun

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/runs.db")
    session_factory = async_sessionmaker(engine)
    relational = importlib.import_module("cognee.infrastructure.databases.relational")
    monkeypatch.setattr(
        relational,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    stamped = {
        "pipeline_run_id": STAMPED_RUN,
        "operation_name": "improve",
        "outcome": "succeeded",
        "ended_at": STAMP_START,
        "run_info": watermarks.watermark_stamp(
            watermarks.GLOBAL_CONTEXT_WATERMARK_KEY, "completed", STAMP_START
        ),
    }
    # The build's terminal row names its parent; its start row (a case below) does not.
    build = {
        "pipeline_run_id": OWN_BUILD,
        "operation_name": "memify_pipeline",
        "status": PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        "parent_operation_id": STAMPED_RUN,
    }
    try:
        async with engine.begin() as connection:
            await connection.run_sync(PipelineRun.__table__.create)
        async with session_factory() as session:
            session.add_all(
                PipelineRun(**{"pipeline_run_id": uuid4(), "dataset_id": dataset_id, **row})
                for row in [stamped, build, *rows]
            )
            await session.commit()
        yield
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("row", "changed"),
    [
        ({"operation_name": "search"}, False),
        ({"operation_name": "memify_pipeline", "pipeline_run_id": OWN_BUILD}, False),
        (
            {
                "operation_name": "cognify_pipeline",
                "status": PipelineRunStatus.DATASET_PROCESSING_STARTED,
            },
            True,
        ),
        (
            {
                "operation_name": "cognify_pipeline",
                "status": PipelineRunStatus.DATASET_PROCESSING_ERRORED,
            },
            True,
        ),
        ({"operation_name": "forget", "outcome": "succeeded"}, True),
        ({"operation_name": "add_pipeline"}, True),
        ({"operation_name": "improve", "outcome": "succeeded"}, True),
        ({"operation_name": "improve", "outcome": "noop"}, False),
    ],
    ids=[
        "read",
        "the stamped run's own build",
        "another run's start row",
        "failed write",
        "deletion",
        "add",
        "another improve",
        "noop improve",
    ],
)
@pytest.mark.asyncio
async def test_every_run_but_a_read_a_noop_or_the_stamped_one_changes_the_dataset(
    monkeypatch, tmp_path, row, changed
):
    """A failed write may have written part of its data, a deletion runs no
    pipeline, and add() can drop a source's stale documents. A noop improve ran
    no stage."""
    dataset_id = uuid4()

    async with _pipeline_runs_on_sqlite(monkeypatch, tmp_path, dataset_id, [row]):
        assert (
            await watermarks.has_dataset_changed_since_last_improve(
                dataset_id, watermarks.GLOBAL_CONTEXT_WATERMARK_KEY
            )
            is changed
        )


@pytest.mark.asyncio
async def test_another_datasets_run_counts_only_when_the_graph_is_shared(monkeypatch, tmp_path):
    dataset_id = uuid4()
    other_write = {"dataset_id": uuid4(), "operation_name": "cognify_pipeline"}
    key = watermarks.GLOBAL_CONTEXT_WATERMARK_KEY

    async with _pipeline_runs_on_sqlite(monkeypatch, tmp_path, dataset_id, [other_write]):
        assert await watermarks.has_dataset_changed_since_last_improve(dataset_id, key) is False
        assert (
            await watermarks.has_dataset_changed_since_last_improve(
                dataset_id, key, any_dataset=True
            )
            is True
        )
