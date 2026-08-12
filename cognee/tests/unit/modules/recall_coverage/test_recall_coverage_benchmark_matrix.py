"""Guards on the datasets x agents matrix — spec section 5 route 12.

Two SQL-level invariants that the router cannot check because it only sees the
result:

* **Cells are restricted to ``source = "curated"`` AND the curated question's
  ``scope = "shared"``.** The benchmark set is the only reason two agents' numbers
  compare at all: identical prompts, so the difference is the memory rather than
  the questions. An agent-scoped curated row is one person's list for one tool, and
  an observed row is not a benchmark at all — either leaking in would make the
  "comparable across agents" number not comparable.
* **Each label contributes its latest *complete* run.** A pending run has no
  numbers and a failed one has numbers nobody should read, so joining either would
  show a score that no finished run ever produced.

``AVG`` and the scored count both ignore NULL scores, which is the same rule
:mod:`cognee.modules.recall_coverage.aggregate` applies in Python — a row we could
not judge is absent from the mean rather than counted as a zero.

Real SQLite over ``tmp_path``, so the ``GROUP BY`` and the join are the ones that
will run in production. No LLM, no embedding engine, no network.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import update

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.aggregate import (
    CoverageRow,
    row_impact,
    run_counters,
    summarize,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.models import (
    RecallCoverageCuratedQuestion,
    RecallCoverageQuestion,
    RecallCoverageRun,
)
from cognee.modules.recall_coverage.types import CoverageParams, CuratedScope, QuestionSource

repository = importlib.import_module("cognee.modules.recall_coverage.repository")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

DATASET_A = uuid4()
DATASET_B = uuid4()


def _config() -> RecallCoverageConfig:
    return RecallCoverageConfig(_env_file=None)


def _params(**overrides) -> CoverageParams:
    return CoverageParams.from_config(_config(), **overrides)


def _user(user_id=None) -> SimpleNamespace:
    return SimpleNamespace(id=user_id or uuid4(), tenant_id=None)


@pytest_asyncio.fixture
async def matrix_engine(tmp_path, monkeypatch):
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recall_coverage_matrix_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[
                RecallCoverageRun.__table__,
                RecallCoverageQuestion.__table__,
                RecallCoverageCuratedQuestion.__table__,
            ],
        )

    monkeypatch.setattr(repository, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


def _row(
    *,
    user_id,
    text,
    dataset_id,
    judge_score,
    source=QuestionSource.OBSERVED.value,
    curated_question_id=None,
) -> CoverageRow:
    return CoverageRow(
        question_text=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name="dataset-a" if dataset_id == DATASET_A else "dataset-b",
        question_group_id=uuid4(),
        source=source,
        was_asked=source == QuestionSource.OBSERVED.value,
        curated_question_id=curated_question_id,
        topic_id=None,
        topic_label="Other",
        answer=None,
        judge_score=judge_score,
        judge_answered=None if judge_score is None else judge_score > 0,
        retrieval_context=None,
        error=None,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        occurrence_count=1,
        impact=row_impact(1, judge_score, 5),
        is_shared_curated=source == QuestionSource.CURATED.value,
    )


async def _complete(owner_id, label, rows, *, created_at=None):
    params = _params()
    run = await repository.create_run(owner_id, label, params=params)
    await repository.persist_run_results(
        run.id,
        rows,
        summarize(rows, params=params),
        run_counters(
            rows,
            recall_row_count=len(rows),
            distinct_ask_count=len(rows),
            collapsed_retry_count=0,
            taxonomy_version=0,
        ),
    )
    if created_at is not None:
        # Pinned rather than trusted: two runs inserted in one test can share a
        # created_at to the microsecond, and "latest" must be decided by the test.
        engine = repository.get_relational_engine()
        async with engine.get_async_session() as session:
            await session.execute(
                update(RecallCoverageRun)
                .where(RecallCoverageRun.id == run.id)
                .values(created_at=created_at)
            )
            await session.commit()
    return run


@pytest.mark.asyncio
async def test_only_shared_curated_rows_reach_the_matrix(matrix_engine):
    caller = _user()

    shared = await repository.create_curated_question(
        caller, "What is our escalation path?", CuratedScope.SHARED.value, None, config=_config()
    )
    agent_scoped = await repository.create_curated_question(
        caller, "Where are the runbooks?", CuratedScope.AGENT.value, "codex", config=_config()
    )

    rows = [
        # Two shared curated rows in dataset A, one in dataset B.
        _row(
            user_id=caller.id,
            text="What is our escalation path?",
            dataset_id=DATASET_A,
            judge_score=4,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        ),
        _row(
            user_id=caller.id,
            text="What is our escalation path? (2)",
            dataset_id=DATASET_A,
            judge_score=2,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        ),
        _row(
            user_id=caller.id,
            text="What is our escalation path? (3)",
            dataset_id=DATASET_B,
            judge_score=5,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        ),
        # An agent-scoped curated row: not a benchmark.
        _row(
            user_id=caller.id,
            text="Where are the runbooks?",
            dataset_id=DATASET_A,
            judge_score=0,
            source=QuestionSource.CURATED.value,
            curated_question_id=agent_scoped.id,
        ),
        # And plain observed traffic, which is not a benchmark either.
        _row(
            user_id=caller.id,
            text="How do I rotate credentials?",
            dataset_id=DATASET_A,
            judge_score=0,
        ),
    ]

    run = await _complete(caller.id, "codex", rows)

    cells = await repository.benchmark_cells({"codex": run.id})

    by_dataset = {cell.dataset_id: cell for cell in cells}
    assert set(by_dataset) == {DATASET_A, DATASET_B}

    # Only the two shared rows in dataset A: mean of 4 and 2, and the 0-scoring
    # agent-scoped and observed rows are absent.
    assert by_dataset[DATASET_A].question_count == 2
    assert by_dataset[DATASET_A].scored_question_count == 2
    assert by_dataset[DATASET_A].avg_score == 3.0
    assert by_dataset[DATASET_A].dataset_name == "dataset-a"
    assert by_dataset[DATASET_A].agent_label == "codex"

    assert by_dataset[DATASET_B].question_count == 1
    assert by_dataset[DATASET_B].avg_score == 5.0


@pytest.mark.asyncio
async def test_an_unjudged_row_is_counted_but_not_averaged(matrix_engine):
    caller = _user()
    shared = await repository.create_curated_question(
        caller, "What is our escalation path?", CuratedScope.SHARED.value, None, config=_config()
    )

    rows = [
        _row(
            user_id=caller.id,
            text="q1",
            dataset_id=DATASET_A,
            judge_score=4,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        ),
        _row(
            user_id=caller.id,
            text="q2",
            dataset_id=DATASET_A,
            judge_score=None,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        ),
    ]
    run = await _complete(caller.id, "codex", rows)

    cell = (await repository.benchmark_cells({"codex": run.id}))[0]

    assert cell.question_count == 2
    assert cell.scored_question_count == 1
    # 4, not 2: an unjudged row is absent from the mean, never a zero in it.
    assert cell.avg_score == 4.0


@pytest.mark.asyncio
async def test_two_agents_are_two_rows_of_the_same_matrix(matrix_engine):
    caller = _user()
    shared = await repository.create_curated_question(
        caller, "What is our escalation path?", CuratedScope.SHARED.value, None, config=_config()
    )

    def _benchmark_row(score, text):
        return _row(
            user_id=caller.id,
            text=text,
            dataset_id=DATASET_A,
            judge_score=score,
            source=QuestionSource.CURATED.value,
            curated_question_id=shared.id,
        )

    codex = await _complete(caller.id, "codex", [_benchmark_row(4, "q1")])
    claude = await _complete(caller.id, "claude-code", [_benchmark_row(2, "q1")])

    cells = await repository.benchmark_cells({"codex": codex.id, "claude-code": claude.id})

    assert [(cell.agent_label, cell.avg_score) for cell in cells] == [
        ("claude-code", 2.0),
        ("codex", 4.0),
    ]


@pytest.mark.asyncio
async def test_no_runs_means_no_query_and_no_cells(matrix_engine):
    assert await repository.benchmark_cells({}) == []


# --- latest complete runs -----------------------------------------------------


@pytest.mark.asyncio
async def test_latest_complete_runs_picks_the_newest_finished_run_per_label(matrix_engine):
    caller = _user()

    older = await _complete(caller.id, "codex", [], created_at=BASE_TIME - timedelta(days=1))
    newer = await _complete(caller.id, "codex", [], created_at=BASE_TIME)
    other = await _complete(caller.id, "claude-code", [], created_at=BASE_TIME)

    latest = await repository.latest_complete_runs([caller.id])

    assert latest["codex"].id == newer.id
    assert latest["claude-code"].id == other.id
    assert older.id not in {run.id for run in latest.values()}


@pytest.mark.asyncio
async def test_pending_and_failed_runs_are_never_the_latest(matrix_engine):
    caller = _user()

    complete = await _complete(caller.id, "codex", [], created_at=BASE_TIME - timedelta(days=1))

    pending = await repository.create_run(caller.id, "codex", params=_params())
    failed = await repository.create_run(caller.id, "codex", params=_params())
    await repository.fail_run(failed.id, "boom")

    latest = await repository.latest_complete_runs([caller.id])

    assert latest["codex"].id == complete.id
    assert pending.id not in {run.id for run in latest.values()}
    assert failed.id not in {run.id for run in latest.values()}


@pytest.mark.asyncio
async def test_latest_complete_runs_is_owner_scoped_and_label_filterable(matrix_engine):
    caller = _user()
    stranger = _user()

    await _complete(caller.id, "codex", [])
    await _complete(caller.id, "claude-code", [])
    await _complete(stranger.id, "codex", [])

    assert set(await repository.latest_complete_runs([caller.id])) == {"codex", "claude-code"}
    assert set(await repository.latest_complete_runs([caller.id], ["codex"])) == {"codex"}
    assert set(await repository.latest_complete_runs([uuid4()])) == set()
