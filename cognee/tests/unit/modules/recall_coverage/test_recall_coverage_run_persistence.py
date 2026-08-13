"""Guards on run and question-row persistence — spec section 2 phase 4 step 14.

Four invariants:

* **The summary is frozen.** It is written once, at completion, and stays a
  literal JSON snapshot. Recomputing breakdowns at read time would let a deleted
  topic — or an owner losing dataset access — silently reshape a historical run
  and destroy the trend that stable topic ids exist to carry, so this file asserts
  the stored dict still names a topic after that topic is gone.
* **Rows and summary land together.** One transaction, so a reader never sees a
  table full of scores under a ``memory_score: null`` that would read as
  "memory answered nothing".
* **Owner scope is part of every id-keyed read**, and a miss is 404, never 403.
* **The default report order is a read-side property.** There is no ordering
  column, ``id`` is a ``uuid4`` and every row in a run shares a ``created_at``, so
  ``load_run_questions`` re-applies the order rather than trusting the table.

SQLite over ``tmp_path`` holding only the two tables under test; no LLM, no
embedding engine, no network.
"""

import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.aggregate import (
    CoverageRow,
    SuggestedTopic,
    run_counters,
    summarize,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import CoverageRunNotFoundError
from cognee.modules.recall_coverage.models import (
    RecallCoverageQuestion,
    RecallCoverageRun,
)
from cognee.modules.recall_coverage.types import CoverageParams, QuestionSource, RunStatus

repository = importlib.import_module("cognee.modules.recall_coverage.repository")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

AGENT_LABEL = "claude-code"


def _params(**overrides) -> CoverageParams:
    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _row(
    *,
    user_id,
    dataset_id=None,
    dataset_name=None,
    topic_id=None,
    coverage_score=3,
    relevance=2,
    source: str = QuestionSource.OBSERVED.value,
    agent_label="claude-code",
    text: str = "Where are the runbooks?",
    error=None,
) -> CoverageRow:
    return CoverageRow(
        question=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        agent_label=agent_label if relevance > 0 else None,
        source=source,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        topic_id=topic_id,
        topic="Runbooks" if topic_id else "Uncategorized",
        answer="An answer." if coverage_score else None,
        coverage_score=coverage_score,
        retrieval_context="Some context." if coverage_score else None,
        error=error,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        relevance=relevance,
    )


@pytest_asyncio.fixture
async def run_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the run and question tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recall_coverage_runs_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[RecallCoverageRun.__table__, RecallCoverageQuestion.__table__],
        )

    monkeypatch.setattr(repository, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


# --- the run row lifecycle ---------------------------------------------------


@pytest.mark.asyncio
async def test_create_run_inserts_a_pending_row_carrying_its_params(run_engine):
    owner_id = uuid4()
    params = _params(max_questions=42)

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)

    assert run.status == RunStatus.PENDING.value
    assert run.is_in_flight
    assert run.agent_label == AGENT_LABEL
    assert run.owner_id == owner_id
    assert run.recall_count == 0
    assert run.question_count == 0
    assert run.finished_at is None
    # Persisted in JSON mode, so the snapshot survives a round trip unchanged.
    assert run.params["max_questions"] == 42
    assert run.params["query_types"]


@pytest.mark.asyncio
async def test_mark_run_running_then_persist_completes_the_run(run_engine):
    owner_id = uuid4()
    params = _params(min_scored_questions_per_topic=1)
    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)

    running = await repository.mark_run_running(run.id)
    assert running.status == RunStatus.RUNNING.value
    assert running.is_in_flight

    rows = [_row(user_id=uuid4(), topic_id=uuid4(), coverage_score=4)]
    summary = summarize(rows, params=params)
    counters = run_counters(rows, recall_count=9)

    completed = await repository.persist_run_results(run.id, rows, summary, counters)

    assert completed.status == RunStatus.COMPLETE.value
    assert not completed.is_in_flight
    assert completed.finished_at is not None
    assert completed.recall_count == 9
    assert completed.question_count == 1
    assert completed.summary["memory_score"] == 4.0


@pytest.mark.asyncio
async def test_an_empty_window_completes_with_a_null_score_and_no_questions(run_engine):
    """ "Nothing asked yet" is a complete run, not a failure and not a score of zero."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    await repository.mark_run_running(run.id)

    summary = summarize([], params=params)
    counters = run_counters([], recall_count=0)

    completed = await repository.persist_run_results(run.id, [], summary, counters)

    assert completed.status == RunStatus.COMPLETE.value
    assert completed.summary["memory_score"] is None
    assert completed.question_count == 0
    assert await repository.load_run_questions(run.id) == []


@pytest.mark.asyncio
async def test_fail_run_records_why_and_leaves_the_counters_alone(run_engine):
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)

    failed = await repository.fail_run(run.id, "the embedding engine returned zero vectors")

    assert failed.status == RunStatus.FAILED.value
    assert failed.finished_at is not None
    assert "zero vectors" in failed.summary["error"]
    assert failed.question_count == 0


@pytest.mark.asyncio
async def test_persisting_into_a_missing_run_raises_not_found(run_engine):
    params = _params()
    summary = summarize([], params=params)
    counters = run_counters([], recall_count=0)

    with pytest.raises(CoverageRunNotFoundError):
        await repository.persist_run_results(uuid4(), [], summary, counters)

    with pytest.raises(CoverageRunNotFoundError):
        await repository.mark_run_running(uuid4())


# --- the frozen summary ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_stored_summary_names_the_topic_verbatim(run_engine):
    """Why freezing works: nothing needs to still exist for the report to render.

    The stored JSON carries the topic id *and its label* as literals, so deleting
    the topic later cannot rewrite this run — and the read endpoint resolves each
    question row's topic name out of here rather than out of a live join, which is
    what makes that true.
    """
    owner_id = uuid4()
    user_id = uuid4()
    dataset_id = uuid4()
    topic_id = uuid4()
    params = _params(min_scored_questions_per_topic=1)

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    rows = [
        _row(
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_name="infra-docs",
            topic_id=topic_id,
            coverage_score=score,
            text=f"q{score}",
        )
        for score in (4, 2)
    ]
    summary = summarize(rows, params=params)
    counters = run_counters(rows, recall_count=7)

    await repository.persist_run_results(run.id, rows, summary, counters)

    # The topic, the dataset and the user are all named inside the stored JSON, so
    # nothing needs to still exist for the report to render.
    stored = (await repository.get_run(run.id, [owner_id])).summary

    assert stored["memory_score"] == 3.0
    assert stored["topics"] == [
        {
            "topic_id": str(topic_id),
            "topic": "Runbooks",
            "question_count": 2,
            "memory_score": 3.0,
        }
    ]
    # This run proposed nothing, and that is an empty list rather than a null.
    assert stored["suggested_topics"] == []
    # The dataset and the asker travel on the question rows, not in the summary:
    # the report is one flat table the UI groups, so there is no per-dataset block
    # to keep in step with it.
    assert set(stored) == {"memory_score", "topics", "suggested_topics"}
    stored_rows = await repository.load_run_questions(run.id)
    assert {row.dataset_name for row in stored_rows} == {"infra-docs"}
    assert {row.user_id for row in stored_rows} == {user_id}


@pytest.mark.asyncio
async def test_the_question_rows_and_the_summary_land_in_one_transaction(run_engine):
    owner_id = uuid4()
    params = _params(min_scored_questions_per_topic=1)
    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)

    rows = [_row(user_id=uuid4(), coverage_score=index, text=f"q{index}") for index in range(3)]
    summary = summarize(rows, params=params)
    counters = run_counters(rows, recall_count=3)

    await repository.persist_run_results(run.id, rows, summary, counters)

    async with run_engine.get_async_session() as session:
        stored_rows = (
            (
                await session.execute(
                    select(RecallCoverageQuestion).where(RecallCoverageQuestion.run_id == run.id)
                )
            )
            .scalars()
            .all()
        )
        stored_run = (
            await session.execute(select(RecallCoverageRun).where(RecallCoverageRun.id == run.id))
        ).scalar_one()

    assert len(stored_rows) == 3
    assert stored_run.status == RunStatus.COMPLETE.value
    assert stored_run.summary["topics"][0]["question_count"] == 3


# --- question rows -----------------------------------------------------------


@pytest.mark.asyncio
async def test_persisted_question_rows_round_trip_every_judged_field(run_engine):
    owner_id = uuid4()
    user_id = uuid4()
    dataset_id = uuid4()
    topic_id = uuid4()
    params = _params()

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    row = _row(
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name="infra-docs",
        topic_id=topic_id,
        coverage_score=1,
        relevance=4,
    )
    summary = summarize([row], params=params)
    counters = run_counters([row], recall_count=4)

    await repository.persist_run_results(run.id, [row], summary, counters)
    stored = (await repository.load_run_questions(run.id))[0]

    assert stored.run_id == run.id
    assert stored.question == row.question
    assert stored.user_id == user_id
    assert stored.dataset_id == dataset_id
    assert stored.dataset_name == "infra-docs"
    assert stored.source == QuestionSource.OBSERVED.value
    assert stored.topic_id == topic_id
    assert stored.coverage_score == 1
    assert stored.retrieval_context == "Some context."
    assert stored.relevance == 4
    assert stored.error is None
    assert stored.first_asked_at is not None
    # Persisted rather than re-derived at read time: the label came from the
    # prefix map as configured when the run executed, so deriving it later would
    # relabel history the moment a deployment edits that map.
    assert stored.agent_label == "claude-code"


@pytest.mark.asyncio
async def test_the_topic_label_is_not_persisted_on_the_question_row(run_engine):
    """The label lives on the topic row, and "Uncategorized" is not a row at all.

    Storing it per question would let a run's copy drift from the thing it
    describes; the report reads it out of the frozen summary instead.
    """
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    row = _row(user_id=uuid4(), topic_id=uuid4())

    await repository.persist_run_results(
        run.id, [row], summarize([row], params=params), run_counters([row], recall_count=1)
    )

    assert "topic" not in {column.name for column in RecallCoverageQuestion.__table__.columns}
    assert not hasattr((await repository.load_run_questions(run.id))[0], "topic")


@pytest.mark.asyncio
async def test_a_user_defined_row_nobody_asked_persists_a_null_agent(run_engine):
    """Nobody asked it, so no agent can be responsible for it."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    row = _row(user_id=uuid4(), source=QuestionSource.USER_DEFINED.value, relevance=0)

    await repository.persist_run_results(
        run.id, [row], summarize([row], params=params), run_counters([row], recall_count=0)
    )
    stored = (await repository.load_run_questions(run.id))[0]

    assert stored.agent_label is None
    assert stored.relevance == 0
    assert stored.is_curated


@pytest.mark.asyncio
async def test_an_unjudged_row_persists_a_null_score_and_its_error(run_engine):
    """ "We could not ask" is a NULL score, never a zero — a zero is a real verdict."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    row = _row(user_id=uuid4(), coverage_score=None, error="replay failed", relevance=9)

    summary = summarize([row], params=params)
    counters = run_counters([row], recall_count=9)
    await repository.persist_run_results(run.id, [row], summary, counters)

    stored = (await repository.load_run_questions(run.id))[0]

    assert stored.coverage_score is None
    assert stored.error == "replay failed"
    assert stored.relevance == 9


@pytest.mark.asyncio
async def test_load_run_questions_pins_user_defined_rows_above_observed_rows(run_engine):
    """The read side re-applies the default order; the table has no ordering column."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)

    rows = [
        _row(user_id=uuid4(), coverage_score=5, relevance=1, text="best observed"),
        _row(user_id=uuid4(), coverage_score=1, relevance=8, text="worst observed"),
        _row(
            user_id=uuid4(),
            source=QuestionSource.USER_DEFINED.value,
            coverage_score=0,
            relevance=0,
            text="user defined",
        ),
        _row(user_id=uuid4(), coverage_score=None, error="boom", relevance=99, text="unjudged"),
    ]
    summary = summarize(rows, params=params)
    counters = run_counters(rows, recall_count=9)
    await repository.persist_run_results(run.id, rows, summary, counters)

    ordered = [record.question for record in await repository.load_run_questions(run.id)]

    # The read side and the write side are duck-typed on the same key, so a drift
    # between them cannot hide: user-defined first, then by demand.
    assert ordered == ["user defined", "unjudged", "worst observed", "best observed"]


@pytest.mark.asyncio
async def test_load_run_questions_is_scoped_to_one_run(run_engine):
    params = _params()
    owner_id = uuid4()
    first = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    second = await repository.create_run(owner_id, "codex", params=params)

    for run, text in ((first, "claude question"), (second, "codex question")):
        rows = [_row(user_id=uuid4(), text=text)]
        await repository.persist_run_results(
            run.id,
            rows,
            summarize(rows, params=params),
            run_counters(rows, recall_count=1),
        )

    assert [row.question for row in await repository.load_run_questions(first.id)] == [
        "claude question"
    ]


# --- reads and scoping -------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_404s_for_another_owner_rather_than_403(run_engine):
    """A 403 would confirm that a run with this id exists under someone else."""
    owner_id = uuid4()
    other_owner = uuid4()
    run = await repository.create_run(owner_id, AGENT_LABEL, params=_params())

    assert (await repository.get_run(run.id, [owner_id])).id == run.id

    with pytest.raises(CoverageRunNotFoundError):
        await repository.get_run(run.id, [other_owner])

    with pytest.raises(CoverageRunNotFoundError):
        await repository.get_run(uuid4(), [owner_id])


@pytest.mark.asyncio
async def test_list_runs_is_newest_first_owner_scoped_and_limitable(run_engine):
    owner_id = uuid4()
    other_owner = uuid4()

    for _ in range(3):
        await repository.create_run(owner_id, AGENT_LABEL, params=_params())
    await repository.create_run(owner_id, "codex", params=_params())
    await repository.create_run(other_owner, AGENT_LABEL, params=_params())

    mine = await repository.list_runs([owner_id])
    assert len(mine) == 4
    timestamps = [run.created_at for run in mine]
    assert timestamps == sorted(timestamps, reverse=True)

    labelled = await repository.list_runs([owner_id], AGENT_LABEL)
    assert len(labelled) == 3
    assert {run.agent_label for run in labelled} == {AGENT_LABEL}

    assert len(await repository.list_runs([owner_id], limit=2)) == 2
    # limit=None means no LIMIT, not a limit of nothing.
    assert len(await repository.list_runs([owner_id], limit=None)) == 4


@pytest.mark.asyncio
async def test_runs_in_flight_sees_pending_and_running_but_not_settled_runs(run_engine):
    owner_id = uuid4()
    params = _params()

    pending = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    assert [run.id for run in await repository.runs_in_flight(owner_id, AGENT_LABEL)] == [
        pending.id
    ]

    await repository.mark_run_running(pending.id)
    assert [run.id for run in await repository.runs_in_flight(owner_id, AGENT_LABEL)] == [
        pending.id
    ]

    await repository.persist_run_results(
        pending.id,
        [],
        summarize([], params=params),
        run_counters([], recall_count=0),
    )
    assert await repository.runs_in_flight(owner_id, AGENT_LABEL) == []

    failed = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    await repository.fail_run(failed.id, "boom")
    assert await repository.runs_in_flight(owner_id, AGENT_LABEL) == []


@pytest.mark.asyncio
async def test_runs_in_flight_is_per_label_and_per_owner(run_engine):
    """One caller's Codex run must not block their Claude Code run, nor a teammate's."""
    owner_id = uuid4()
    other_owner = uuid4()

    await repository.create_run(owner_id, "codex", params=_params())
    await repository.create_run(other_owner, AGENT_LABEL, params=_params())

    assert await repository.runs_in_flight(owner_id, AGENT_LABEL) == []


@pytest.mark.asyncio
async def test_a_run_killed_in_flight_stops_blocking_after_the_staleness_bound(run_engine):
    """Status is not liveness: the task lives in one process, the row outlives it.

    A pod rescheduled mid-run leaves ``running`` on the row for ever, and there is
    no cancel or delete route — so without an age bound the 409 guard would refuse
    every later run for that ``(owner, agent_label)`` until somebody ran SQL.
    """
    owner_id = uuid4()

    killed = await repository.create_run(owner_id, AGENT_LABEL, params=_params())
    await repository.mark_run_running(killed.id)

    # Backdate the row the way a rescheduled pod would leave it.
    async with run_engine.get_async_session() as session:
        row = (
            await session.execute(
                select(RecallCoverageRun).where(RecallCoverageRun.id == killed.id)
            )
        ).scalar_one()
        row.created_at = datetime.now(timezone.utc) - timedelta(hours=6)
        await session.commit()

    # Unbounded, it still blocks; bounded, it does not.
    assert [run.id for run in await repository.runs_in_flight(owner_id, AGENT_LABEL)] == [killed.id]
    assert (await repository.runs_in_flight(owner_id, AGENT_LABEL, stale_after_seconds=3600)) == []

    # A run that started a moment ago still blocks under the same bound.
    fresh = await repository.create_run(owner_id, AGENT_LABEL, params=_params())
    assert [
        run.id
        for run in await repository.runs_in_flight(owner_id, AGENT_LABEL, stale_after_seconds=3600)
    ] == [fresh.id]


# --- counters ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_two_counters_are_the_window_and_the_rows_it_judged(run_engine):
    """A window truncated by ``window_row_cap`` is only visible in the gap between them.

    Everything else the collapse counted — distinct asks, swallowed retries, how
    many topics or datasets took part — is derivable from the rows or is not
    reported at all, so the run row carries these two and nothing more.
    """
    owner_id = uuid4()
    anna, ben = uuid4(), uuid4()
    params = _params()

    rows = [
        _row(user_id=anna, dataset_id=uuid4(), topic_id=uuid4(), text="a"),
        _row(user_id=ben, topic_id=None, text="b"),
        _row(user_id=ben, source=QuestionSource.USER_DEFINED.value, relevance=0, text="c"),
    ]

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    stored = await repository.persist_run_results(
        run.id, rows, summarize(rows, params=params), run_counters(rows, recall_count=11)
    )

    assert stored.recall_count == 11
    # Every row the run judged, including the one nobody asked.
    assert stored.question_count == 3


@pytest.mark.asyncio
async def test_the_runs_own_suggestions_are_frozen_onto_it(run_engine):
    """A suggestion is a per-run output: the queue moves, a historical run must not."""
    owner_id = uuid4()
    suggestion_id = uuid4()
    params = _params(min_scored_questions_per_topic=1)
    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)

    rows = [_row(user_id=uuid4(), topic_id=None, text="unplaceable")]
    summary = summarize(
        rows,
        params=params,
        suggested_topics=[
            SuggestedTopic(suggestion_id=suggestion_id, label="Deploy rollbacks", question_count=7)
        ],
    )

    await repository.persist_run_results(run.id, rows, summary, run_counters(rows, recall_count=1))
    stored = (await repository.get_run(run.id, [owner_id])).summary

    assert stored["suggested_topics"] == [
        {
            "suggestion_id": str(suggestion_id),
            "label": "Deploy rollbacks",
            "question_count": 7,
        }
    ]
