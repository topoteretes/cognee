"""Stage 8's changed-check reads pipeline_runs and is conservative when it cannot decide.

The watermark is stage-8-scoped: only an improve row stamped by a completed
enrichment (``run_info["triplet_enrichment"]``) gates a later run, and writes
are compared against the stamped stage START time with the stamped run's own
child pipelines excluded.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.improve import graph_changes


def _stamp(started_at=None, status="completed"):
    return graph_changes.enrichment_watermark_stamp(
        status, started_at or datetime.now(timezone.utc)
    )


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


@pytest.mark.asyncio
async def test_no_prior_improve_means_changed(monkeypatch):
    session = _install_engine(monkeypatch, [[]])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True
    assert len(session.statements) == 1  # no second query needed


@pytest.mark.asyncio
async def test_unstamped_improve_rows_are_not_watermarks(monkeypatch):
    """A succeeded run whose stage 8 was skipped (triplet_embedding off, disabled
    by config) carries no stamp and must not gate enrichment off."""
    session = _install_engine(monkeypatch, [[(uuid4(), None), (uuid4(), {"other": 1})]])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_stamped_improve_and_no_writes_means_unchanged(monkeypatch):
    _install_engine(monkeypatch, [[(uuid4(), _stamp())], 0])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is False


@pytest.mark.asyncio
async def test_stamped_improve_and_writes_means_changed(monkeypatch):
    _install_engine(monkeypatch, [[(uuid4(), _stamp())], 3])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True


@pytest.mark.asyncio
async def test_newest_stamped_row_wins_over_older_ones(monkeypatch):
    """The SQL query filters unstamped rows and selects only the newest stamp."""
    session = _install_engine(
        monkeypatch,
        [[(uuid4(), _stamp())], 0],
    )
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is False
    assert len(session.statements) == 2


@pytest.mark.asyncio
async def test_writes_are_compared_against_the_stage_start(monkeypatch):
    """The stamp carries the stage START time so a cognify landing between
    stage 8 and the row close is still visible to the next run's gate."""
    session = _install_engine(monkeypatch, [[(uuid4(), _stamp())], 1])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True
    writes_sql = str(session.statements[1])
    assert "created_at >" in writes_sql


@pytest.mark.asyncio
async def test_stamped_runs_own_children_are_excluded(monkeypatch):
    """Stage 8's own memify row and stage 9's pipeline start after the stamp;
    they are the enrichment, not writes it missed."""
    session = _install_engine(monkeypatch, [[(uuid4(), _stamp())], 0])
    await graph_changes.has_graph_changed_since_last_improve(uuid4())
    writes_sql = str(session.statements[1])
    assert "parent_operation_id" in writes_sql


@pytest.mark.asyncio
async def test_query_failure_is_conservative(monkeypatch):
    _install_engine(monkeypatch, [RuntimeError("db down")])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True


@pytest.mark.asyncio
async def test_malformed_stamp_is_not_a_watermark(monkeypatch):
    session = _install_engine(
        monkeypatch,
        [[(uuid4(), {"triplet_enrichment": {"started_at": "not-a-date"}})]],
    )
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_own_operation_row_is_excluded_from_the_watermark(monkeypatch):
    """The calling run's own record must never serve as its own watermark."""
    session = _install_engine(monkeypatch, [[]])
    own_id = uuid4()

    assert (
        await graph_changes.has_graph_changed_since_last_improve(
            uuid4(), exclude_operation_id=own_id
        )
        is True
    )

    watermark_sql = str(session.statements[0])
    assert "pipeline_run_id !=" in watermark_sql

    session = _install_engine(monkeypatch, [[]])
    await graph_changes.has_graph_changed_since_last_improve(uuid4())
    assert "pipeline_run_id !=" not in str(session.statements[0])


@pytest.mark.asyncio
async def test_watermark_counts_only_succeeded_improve_rows(monkeypatch):
    """ "failed" and "noop" rows (lost lock claim, all-skipped run) are not watermarks."""
    session = _install_engine(monkeypatch, [[]])

    await graph_changes.has_graph_changed_since_last_improve(uuid4())

    watermark_sql = str(session.statements[0])
    assert "outcome =" in watermark_sql


def test_write_pipelines_cover_every_graph_writer_but_not_add():
    names = set(graph_changes.WRITE_PIPELINE_NAMES)
    assert {
        "cognify_pipeline",
        "code_graph_pipeline",
        "memify_pipeline",
        "custom_pipeline",
        "presort_graph_pipeline",
    } <= names
    assert "add_pipeline" not in names


@pytest.mark.parametrize(
    "key", [graph_changes.ENRICHMENT_WATERMARK_KEY, graph_changes.REVIEW_WATERMARK_KEY]
)
@pytest.mark.asyncio
async def test_stamp_lookup_filters_in_sql_for_sqlite_and_postgres(monkeypatch, key):
    from sqlalchemy.dialects import postgresql, sqlite

    session = _install_engine(monkeypatch, [[]])
    assert await graph_changes.last_improve_watermark(uuid4(), key) is None
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


@pytest.mark.parametrize(
    "key", [graph_changes.ENRICHMENT_WATERMARK_KEY, graph_changes.REVIEW_WATERMARK_KEY]
)
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
                    run_info=graph_changes.watermark_stamp(key, "completed", start),
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
                        run_info=graph_changes.watermark_stamp(key, "completed", start),
                        **fields,
                    )
                )
            await session.commit()
        watermark = await graph_changes.last_improve_watermark(
            dataset_id, key, exclude_operation_id=current_id
        )
        assert (watermark.operation_id, watermark.started_at) == (operation_id, start)
        async with session_factory() as session:
            session.add(row(200, run_info={key: {"started_at": "not-a-date"}}))
            await session.commit()
        # An invalid newest stamp cannot borrow an older success to skip work.
        assert (
            await graph_changes.last_improve_watermark(
                dataset_id, key, exclude_operation_id=current_id
            )
            is None
        )
    finally:
        await engine.dispose()
