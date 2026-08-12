"""Guards on run and question-row persistence — spec section 2 phase 4 step 14.

Four invariants:

* **The summary is frozen.** It is written once, at completion, and stays a
  literal JSON snapshot. Recomputing breakdowns at read time would let a deleted
  topic — or an owner losing dataset access — silently reshape a historical run
  and destroy the trend that stable topic ids exist to carry, so this file asserts
  the stored dict still names a topic after that topic is gone.
* **Rows and summary land together.** One transaction, so a reader never sees a
  table full of scores under an ``overall_score: null`` that would read as
  "memory answered nothing".
* **Owner scope is part of every id-keyed read**, and a miss is 404, never 403.
* **The default report order is a read-side property.** There is no ordering
  column, ``id`` is a ``uuid4`` and every row in a run shares a ``created_at``, so
  ``load_run_questions`` re-applies the order rather than trusting the table.

SQLite over ``tmp_path`` holding only the two tables under test; no LLM, no
embedding engine, no network.
"""

import importlib
from datetime import datetime, timezone
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
    row_impact,
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
    judge_score=3,
    occurrence_count=2,
    source: str = QuestionSource.OBSERVED.value,
    text: str = "Where are the runbooks?",
    error=None,
) -> CoverageRow:
    return CoverageRow(
        question_text=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        question_group_id=uuid4(),
        source=source,
        was_asked=source == QuestionSource.OBSERVED.value,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        topic_id=topic_id,
        topic_label="Runbooks" if topic_id else "Other",
        answer="An answer." if judge_score else None,
        judge_score=judge_score,
        judge_answered=None if judge_score is None else judge_score > 0,
        retrieval_context="Some context." if judge_score else None,
        error=error,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        occurrence_count=occurrence_count,
        impact=row_impact(occurrence_count, judge_score, 5),
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

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params, taxonomy_version=3)

    assert run.status == RunStatus.PENDING.value
    assert run.is_in_flight
    assert run.agent_label == AGENT_LABEL
    assert run.owner_id == owner_id
    assert run.taxonomy_version == 3
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

    rows = [_row(user_id=uuid4(), topic_id=uuid4(), judge_score=4)]
    summary = summarize(rows, params=params, distinct_ask_count=5)
    counters = run_counters(
        rows,
        recall_row_count=9,
        distinct_ask_count=5,
        collapsed_retry_count=4,
        taxonomy_version=2,
    )

    completed = await repository.persist_run_results(run.id, rows, summary, counters)

    assert completed.status == RunStatus.COMPLETE.value
    assert not completed.is_in_flight
    assert completed.finished_at is not None
    assert completed.recall_row_count == 9
    assert completed.distinct_ask_count == 5
    assert completed.collapsed_retry_count == 4
    assert completed.question_row_count == 1
    assert completed.taxonomy_version == 2
    assert completed.summary["overall_score"] == 4.0


@pytest.mark.asyncio
async def test_an_empty_window_completes_with_a_null_score_and_no_questions(run_engine):
    """The expected result for every label but ``all`` until ``session_id`` ships."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    await repository.mark_run_running(run.id)

    summary = summarize([], params=params)
    counters = run_counters(
        [], recall_row_count=0, distinct_ask_count=0, collapsed_retry_count=0, taxonomy_version=0
    )

    completed = await repository.persist_run_results(run.id, [], summary, counters)

    assert completed.status == RunStatus.COMPLETE.value
    assert completed.summary["overall_score"] is None
    assert completed.question_row_count == 0
    assert await repository.load_run_questions(run.id) == []


@pytest.mark.asyncio
async def test_fail_run_records_why_and_leaves_the_counters_alone(run_engine):
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)

    failed = await repository.fail_run(run.id, "the embedding engine returned zero vectors")

    assert failed.status == RunStatus.FAILED.value
    assert failed.finished_at is not None
    assert "zero vectors" in failed.summary["error"]
    assert failed.question_row_count == 0


@pytest.mark.asyncio
async def test_persisting_into_a_missing_run_raises_not_found(run_engine):
    params = _params()
    summary = summarize([], params=params)
    counters = run_counters(
        [], recall_row_count=0, distinct_ask_count=0, collapsed_retry_count=0, taxonomy_version=0
    )

    with pytest.raises(CoverageRunNotFoundError):
        await repository.persist_run_results(uuid4(), [], summary, counters)

    with pytest.raises(CoverageRunNotFoundError):
        await repository.mark_run_running(uuid4())


# --- the frozen summary ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_stored_summary_names_the_topic_dataset_and_user_verbatim(run_engine):
    """Why freezing works: nothing needs to still exist for the report to render.

    The stored JSON carries the topic id and label, the dataset id and name and
    the asker's id as literals, so deleting the topic or losing access to the
    dataset later cannot rewrite this run. (The delete itself is commit 8's; this
    asserts the property that makes it harmless.)
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
            judge_score=score,
            text=f"q{score}",
        )
        for score in (4, 2)
    ]
    summary = summarize(rows, params=params, distinct_ask_count=7)
    counters = run_counters(
        rows,
        recall_row_count=7,
        distinct_ask_count=7,
        collapsed_retry_count=0,
        taxonomy_version=1,
    )

    await repository.persist_run_results(run.id, rows, summary, counters)

    # The topic, the dataset and the user are all named inside the stored JSON, so
    # nothing needs to still exist for the report to render.
    stored = (await repository.get_run(run.id, [owner_id])).summary

    assert stored["overall_score"] == 3.0
    assert stored["topics"] == [
        {
            "topic_id": str(topic_id),
            "label": "Runbooks",
            "question_count": 2,
            "scored_question_count": 2,
            "avg_score": 3.0,
        }
    ]
    assert stored["datasets"][0]["dataset_id"] == str(dataset_id)
    assert stored["datasets"][0]["dataset_name"] == "infra-docs"
    assert stored["users"][0]["user_id"] == str(user_id)
    assert stored["unscoped_ask_share"] == 0.0


@pytest.mark.asyncio
async def test_the_question_rows_and_the_summary_land_in_one_transaction(run_engine):
    owner_id = uuid4()
    params = _params(min_scored_questions_per_topic=1)
    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)

    rows = [_row(user_id=uuid4(), judge_score=index, text=f"q{index}") for index in range(3)]
    summary = summarize(rows, params=params)
    counters = run_counters(
        rows, recall_row_count=3, distinct_ask_count=3, collapsed_retry_count=0, taxonomy_version=0
    )

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
    assert stored_run.summary["scored_question_count"] == 3


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
        judge_score=1,
        occurrence_count=4,
    )
    summary = summarize([row], params=params)
    counters = run_counters(
        [row], recall_row_count=4, distinct_ask_count=4, collapsed_retry_count=0, taxonomy_version=0
    )

    await repository.persist_run_results(run.id, [row], summary, counters)
    stored = (await repository.load_run_questions(run.id))[0]

    assert stored.run_id == run.id
    assert stored.question_text == row.question_text
    assert stored.user_id == user_id
    assert stored.dataset_id == dataset_id
    assert stored.dataset_name == "infra-docs"
    assert stored.question_group_id == row.question_group_id
    assert stored.source == QuestionSource.OBSERVED.value
    assert stored.was_asked is True
    assert stored.topic_id == topic_id
    assert stored.judge_score == 1
    assert stored.judge_answered is True
    assert stored.retrieval_context == "Some context."
    assert stored.occurrence_count == 4
    # 4 asks x (5 - 1) missing score.
    assert stored.impact == 16.0
    assert stored.error is None
    assert stored.first_asked_at is not None


@pytest.mark.asyncio
async def test_an_unjudged_row_persists_null_scores_and_a_null_impact(run_engine):
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)
    row = _row(user_id=uuid4(), judge_score=None, error="replay failed", occurrence_count=9)

    summary = summarize([row], params=params)
    counters = run_counters(
        [row], recall_row_count=9, distinct_ask_count=9, collapsed_retry_count=0, taxonomy_version=0
    )
    await repository.persist_run_results(run.id, [row], summary, counters)

    stored = (await repository.load_run_questions(run.id))[0]

    assert stored.judge_score is None
    assert stored.judge_answered is None
    assert stored.impact is None
    assert stored.error == "replay failed"


@pytest.mark.asyncio
async def test_load_run_questions_pins_curated_rows_above_observed_rows(run_engine):
    """The read side re-applies the default order; the table has no ordering column."""
    params = _params()
    run = await repository.create_run(uuid4(), AGENT_LABEL, params=params)

    rows = [
        _row(user_id=uuid4(), judge_score=5, occurrence_count=1, text="best observed"),
        _row(user_id=uuid4(), judge_score=1, occurrence_count=8, text="worst observed"),
        _row(
            user_id=uuid4(),
            source=QuestionSource.CURATED.value,
            judge_score=0,
            occurrence_count=0,
            text="curated",
        ),
        _row(user_id=uuid4(), judge_score=None, error="boom", occurrence_count=99, text="unjudged"),
    ]
    summary = summarize(rows, params=params)
    counters = run_counters(
        rows, recall_row_count=9, distinct_ask_count=9, collapsed_retry_count=0, taxonomy_version=0
    )
    await repository.persist_run_results(run.id, rows, summary, counters)

    ordered = [record.question_text for record in await repository.load_run_questions(run.id)]

    assert ordered == ["curated", "worst observed", "best observed", "unjudged"]


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
            run_counters(
                rows,
                recall_row_count=1,
                distinct_ask_count=1,
                collapsed_retry_count=0,
                taxonomy_version=0,
            ),
        )

    assert [row.question_text for row in await repository.load_run_questions(first.id)] == [
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
        run_counters(
            [],
            recall_row_count=0,
            distinct_ask_count=0,
            collapsed_retry_count=0,
            taxonomy_version=0,
        ),
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


# --- counters ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_counters_count_participants_not_taxonomy_size(run_engine):
    owner_id = uuid4()
    anna, ben = uuid4(), uuid4()
    dataset_id = uuid4()
    topic_id = uuid4()
    params = _params()

    rows = [
        _row(user_id=anna, dataset_id=dataset_id, topic_id=topic_id, text="a"),
        _row(user_id=ben, dataset_id=dataset_id, topic_id=None, text="b"),
        _row(user_id=ben, dataset_id=None, source=QuestionSource.CURATED.value, text="c"),
    ]
    counters = run_counters(
        rows,
        recall_row_count=11,
        distinct_ask_count=6,
        collapsed_retry_count=5,
        taxonomy_version=4,
    )

    run = await repository.create_run(owner_id, AGENT_LABEL, params=params)
    stored = await repository.persist_run_results(
        run.id, rows, summarize(rows, params=params), counters
    )

    assert stored.question_row_count == 3
    assert stored.curated_question_count == 1
    # One topic received a row; the sink is not a topic and the curated row's
    # dataset is NULL, so neither is counted.
    assert stored.topic_count == 1
    assert stored.dataset_count == 1
    assert stored.user_count == 2
    assert stored.recall_row_count == 11
    assert stored.distinct_ask_count == 6
    assert stored.collapsed_retry_count == 5
    assert stored.taxonomy_version == 4
