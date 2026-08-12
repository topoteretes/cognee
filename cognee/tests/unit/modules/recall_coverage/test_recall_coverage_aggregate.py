"""Guards on recall-coverage aggregation — spec section 2 phase 4 step 13.

Every assertion here is about *which rows* a number is a mean over, because that
is the only way this report can be wrong without looking wrong:

* **``null`` is not ``0.0``.** ``0.0`` says memory answered nothing; ``null`` says
  the run measured nothing. Two tests pin the difference, including the case
  where every score genuinely is 0 and ``overall_score`` must therefore *be*
  ``0.0``.
* **Curated rows feed nothing but ``benchmark_score_pct``**, and only the shared
  ones — otherwise a user could raise the headline score by adding questions
  their memory happens to answer.
* **The breakdowns reconcile.** Topics-plus-sink, datasets and users are three
  ``GROUP BY``s over one row set; their counts and their count-weighted means
  must agree, and with equally sized topics they agree with ``overall_score``
  exactly.
* **A NULL score is not a zero**, and neither is its ``impact``.

No LLM, no embedding engine, no database: the phase-3 outputs are constructed
directly, and every parameter comes from ``RecallCoverageConfig(_env_file=None)``
so a developer's ``.env`` cannot move a threshold under the assertions.
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from cognee.modules.recall_coverage.aggregate import (
    SINK_CLUSTER_ALERT,
    SINK_SHARE_ALERT,
    CoverageRow,
    build_rows,
    dataset_breakdown,
    default_row_order,
    reconciles,
    row_impact,
    summarize,
    topic_breakdown,
    user_breakdown,
)
from cognee.modules.recall_coverage.assign import TopicAssignment
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.judge import JudgedRow
from cognee.modules.recall_coverage.replay import ReplayedRow
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_ID,
    SINK_TOPIC_LABEL,
    CoverageParams,
    QuestionSource,
)

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

ANNA = uuid4()
BEN = uuid4()

INFRA = uuid4()
BILLING = uuid4()

TOPIC_ONE = uuid4()
TOPIC_TWO = uuid4()


def _params(**overrides) -> CoverageParams:
    """A parameter snapshot that ignores the developer's ``.env``."""
    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _row(
    *,
    user_id: UUID = ANNA,
    dataset_id=INFRA,
    dataset_name="infra-docs",
    topic_id=TOPIC_ONE,
    topic_label="Runbooks",
    judge_score=3,
    occurrence_count=1,
    source: str = QuestionSource.OBSERVED.value,
    is_shared_curated: bool = False,
    text: str = "Where are the runbooks?",
    error=None,
    judge_score_max: int = 5,
) -> CoverageRow:
    """One aggregated row, with the impact the formula would have given it."""
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
        topic_label=SINK_TOPIC_LABEL if topic_id is None else topic_label,
        answer=None if judge_score in (None, 0) else "An answer.",
        judge_score=judge_score,
        judge_answered=None if judge_score is None else judge_score > 0,
        retrieval_context=None if judge_score in (None, 0) else "Some context.",
        error=error,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        occurrence_count=occurrence_count,
        impact=row_impact(occurrence_count, judge_score, judge_score_max),
        is_shared_curated=is_shared_curated,
    )


def _balanced_rows() -> list[CoverageRow]:
    """Six observed rows over two users, two datasets and two equally sized topics.

    Scores sum to 15 over 6 rows, so the row-level mean is 2.5; both topics hold
    three scored rows, so the unweighted mean of the topic averages is 2.5 too.
    That coincidence is what lets the reconciliation test compare the breakdowns
    against ``overall_score`` numerically rather than only structurally.
    """
    return [
        _row(user_id=ANNA, dataset_id=INFRA, topic_id=TOPIC_ONE, judge_score=5, text="a"),
        _row(user_id=ANNA, dataset_id=INFRA, topic_id=TOPIC_ONE, judge_score=3, text="b"),
        _row(
            user_id=ANNA,
            dataset_id=BILLING,
            dataset_name="billing",
            topic_id=TOPIC_TWO,
            topic_label="Billing",
            judge_score=1,
            text="c",
        ),
        _row(
            user_id=BEN,
            dataset_id=INFRA,
            topic_id=TOPIC_TWO,
            topic_label="Billing",
            judge_score=2,
            text="d",
        ),
        _row(user_id=BEN, dataset_id=BILLING, dataset_name="billing", judge_score=4, text="e"),
        _row(
            user_id=BEN,
            dataset_id=BILLING,
            dataset_name="billing",
            topic_id=TOPIC_TWO,
            topic_label="Billing",
            judge_score=0,
            text="f",
        ),
    ]


def _weighted_mean(cells) -> float:
    """The count-weighted mean of a breakdown's averages: its row-level mean."""
    total = sum(cell.scored_question_count for cell in cells)
    weighted = sum(
        cell.avg_score * cell.scored_question_count for cell in cells if cell.avg_score is not None
    )
    return weighted / total


# --- impact ------------------------------------------------------------------


def test_impact_is_occurrence_count_times_the_missing_score():
    assert row_impact(4, 1, 5) == 16.0
    assert row_impact(1, 5, 5) == 0.0
    # A purely curated row nobody has asked yet loses nothing, however it scored.
    assert row_impact(0, 0, 5) == 0.0


def test_impact_is_null_when_the_score_is_null():
    """A row we could not judge is not a row with nothing at stake."""
    assert row_impact(9, None, 5) is None


# --- overall_score: null vs 0.0 ---------------------------------------------


def test_overall_score_is_null_when_no_topic_qualifies():
    """Nothing measured is not the same claim as nothing answered."""
    params = _params(min_scored_questions_per_topic=3)
    rows = [_row(judge_score=5, text=f"q{index}") for index in range(2)]

    summary = summarize(rows, params=params)

    assert summary.overall_score is None
    assert summary.topics[0].avg_score is None
    assert summary.topics[0].question_count == 2


def test_overall_score_is_zero_when_a_qualifying_topic_scored_zero():
    """The other half of the same distinction: 0.0 must remain reachable."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [_row(judge_score=0, text=f"q{index}") for index in range(3)]

    summary = summarize(rows, params=params)

    assert summary.overall_score == 0.0


def test_overall_score_is_null_on_an_empty_run():
    """An empty window completes; it does not report a score of zero."""
    summary = summarize([], params=_params())

    assert summary.overall_score is None
    assert summary.benchmark_score_pct is None
    assert summary.unscoped_ask_share is None
    assert summary.topics == []
    assert summary.sink.share is None
    assert reconciles(summary)


# --- overall_score: exclusions -----------------------------------------------


def test_a_topic_below_the_minimum_is_excluded_from_overall_score():
    """One question must not speak for a whole topic."""
    params = _params(min_scored_questions_per_topic=3)
    rows = [
        _row(topic_id=TOPIC_ONE, judge_score=4, text="a"),
        _row(topic_id=TOPIC_ONE, judge_score=4, text="b"),
        _row(topic_id=TOPIC_ONE, judge_score=4, text="c"),
        _row(topic_id=TOPIC_TWO, topic_label="Billing", judge_score=0, text="d"),
        _row(topic_id=TOPIC_TWO, topic_label="Billing", judge_score=0, text="e"),
    ]

    summary = summarize(rows, params=params)
    by_id = {topic.topic_id: topic for topic in summary.topics}

    assert by_id[TOPIC_TWO].avg_score is None
    assert by_id[TOPIC_TWO].question_count == 2
    # Only the qualifying topic's average survives into the headline.
    assert summary.overall_score == 4.0


def test_the_sink_is_excluded_from_overall_score_but_reported_on_its_own():
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(topic_id=TOPIC_ONE, judge_score=4, text="a"),
        _row(topic_id=None, judge_score=0, text="b"),
        _row(topic_id=None, judge_score=0, text="c"),
    ]

    summary = summarize(rows, params=params)

    assert [topic.topic_id for topic in summary.topics] == [TOPIC_ONE]
    assert summary.overall_score == 4.0
    assert summary.sink.question_count == 2
    assert summary.sink.avg_score == 0.0
    assert summary.sink.share == pytest.approx(2 / 3)
    assert summary.sink.to_dict()["topic_id"] == SINK_TOPIC_ID


def test_an_unscored_row_is_absent_from_every_mean_but_present_in_the_counts():
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(judge_score=4, text="a"),
        _row(judge_score=None, error="replay failed", text="b"),
    ]

    summary = summarize(rows, params=params)

    assert summary.topics[0].question_count == 2
    assert summary.topics[0].scored_question_count == 1
    assert summary.topics[0].avg_score == 4.0
    assert summary.overall_score == 4.0
    assert summary.datasets[0].question_count == 2
    assert summary.datasets[0].scored_question_count == 1


# --- reconciliation ----------------------------------------------------------


def test_dataset_and_user_breakdowns_reconcile_with_overall_score():
    """The three breakdowns are three groupings of one row set.

    Structurally this always holds (:func:`reconciles`): the counts sum to the
    same total and the count-weighted means agree. Numerically it also equals
    ``overall_score`` here because the two topics hold the same number of scored
    rows, so averaging topics and averaging rows coincide — see ``_balanced_rows``.
    """
    params = _params(min_scored_questions_per_topic=1)
    rows = _balanced_rows()

    summary = summarize(rows, params=params)

    assert reconciles(summary)
    assert summary.observed_question_count == 6
    assert summary.scored_question_count == 6
    assert sum(dataset.question_count for dataset in summary.datasets) == 6
    assert sum(user.question_count for user in summary.users) == 6
    assert sum(topic.question_count for topic in summary.topics) == 6
    assert summary.sink.question_count == 0

    assert summary.overall_score == pytest.approx(2.5)
    assert _weighted_mean(summary.datasets) == pytest.approx(2.5)
    assert _weighted_mean(summary.users) == pytest.approx(2.5)
    assert _weighted_mean(summary.topics) == pytest.approx(2.5)


def test_the_breakdowns_still_reconcile_with_a_sink_and_unscored_rows():
    params = _params(min_scored_questions_per_topic=1)
    rows = _balanced_rows() + [
        _row(topic_id=None, judge_score=0, text="g"),
        _row(topic_id=None, judge_score=None, error="boom", text="h"),
        _row(dataset_id=None, dataset_name=None, judge_score=2, text="i"),
    ]

    summary = summarize(rows, params=params)

    assert reconciles(summary)
    assert summary.observed_question_count == 9
    assert summary.scored_question_count == 8
    # The unscoped row groups under dataset_id None rather than being dropped.
    assert None in {dataset.dataset_id for dataset in summary.datasets}


def test_curated_rows_are_absent_from_every_breakdown():
    """Curated rows are in the table and in no ``GROUP BY``."""
    params = _params(min_scored_questions_per_topic=1)
    rows = _balanced_rows() + [
        _row(source=QuestionSource.CURATED.value, judge_score=5, occurrence_count=0, text="cur"),
    ]

    summary = summarize(rows, params=params)

    assert reconciles(summary)
    assert summary.observed_question_count == 6
    assert summary.curated_question_count == 1
    # Adding a perfectly answered curated question must not move the headline.
    assert summary.overall_score == pytest.approx(2.5)
    assert sum(topic.question_count for topic in summary.topics) == 6
    assert sum(user.question_count for user in summary.users) == 6


# --- benchmark_score_pct -----------------------------------------------------


def test_shared_curated_rows_feed_benchmark_score_pct_only():
    params = _params(min_scored_questions_per_topic=1, judge_score_max=5)
    rows = [
        _row(judge_score=4, text="observed"),
        _row(
            source=QuestionSource.CURATED.value,
            is_shared_curated=True,
            judge_score=2,
            occurrence_count=0,
            text="shared one",
        ),
        _row(
            source=QuestionSource.CURATED.value,
            is_shared_curated=True,
            judge_score=3,
            occurrence_count=0,
            text="shared two",
        ),
    ]

    summary = summarize(rows, params=params)

    # Mean of 2 and 3 is 2.5, i.e. 50% of a five-point scale.
    assert summary.benchmark_score_pct == pytest.approx(50.0)
    # The observed row alone is the headline; the benchmark rows are not in it.
    assert summary.overall_score == 4.0


def test_agent_scoped_curated_rows_feed_no_aggregate_at_all():
    """One person's list for one tool is not a cross-agent benchmark."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(judge_score=4, text="observed"),
        _row(
            source=QuestionSource.CURATED.value,
            is_shared_curated=False,
            judge_score=0,
            occurrence_count=0,
            text="agent scoped",
        ),
    ]

    summary = summarize(rows, params=params)

    assert summary.benchmark_score_pct is None
    assert summary.overall_score == 4.0


def test_benchmark_score_pct_is_null_when_no_shared_row_was_scored():
    params = _params()
    rows = [
        _row(
            source=QuestionSource.CURATED.value,
            is_shared_curated=True,
            judge_score=None,
            error="boom",
            occurrence_count=0,
        )
    ]

    assert summarize(rows, params=params).benchmark_score_pct is None


# --- unscoped_ask_share ------------------------------------------------------


def test_unscoped_ask_share_divides_unattributed_rows_by_the_windows_asks():
    params = _params()
    rows = [
        _row(dataset_id=None, dataset_name=None, judge_score=1, text="a"),
        _row(dataset_id=None, dataset_name=None, judge_score=1, text="b"),
        _row(judge_score=1, text="c"),
    ]

    summary = summarize(rows, params=params, distinct_ask_count=50)

    assert summary.unscoped_ask_share == pytest.approx(2 / 50)


def test_unscoped_ask_share_is_null_when_the_window_held_no_asks():
    """Zero would read as "everything is attributed"; nothing was."""
    assert summarize([], params=_params(), distinct_ask_count=0).unscoped_ask_share is None


def test_a_curated_row_without_a_dataset_does_not_count_as_an_unscoped_ask():
    """The denominator counts asks, so the numerator must count asked rows."""
    params = _params()
    rows = [
        _row(
            source=QuestionSource.CURATED.value,
            dataset_id=None,
            dataset_name=None,
            judge_score=1,
            occurrence_count=0,
        )
    ]

    assert summarize(rows, params=params, distinct_ask_count=10).unscoped_ask_share == 0.0


# --- alerts ------------------------------------------------------------------


def test_a_sink_share_above_the_threshold_raises_one_alert():
    params = _params(sink_share_alert=0.30, sink_cluster_alert_size=10)
    rows = [_row(topic_id=None, judge_score=0, text=f"q{index}") for index in range(4)] + [
        _row(judge_score=3, text="assigned")
    ]

    summary = summarize(rows, params=params)

    assert summary.sink.share == pytest.approx(0.8)
    assert [alert.code for alert in summary.sink.alerts] == [SINK_SHARE_ALERT]


def test_a_dense_sink_cluster_raises_an_alert_even_with_a_small_sink():
    """A tight unmatched theme is a missing topic regardless of the overall share."""
    params = _params(sink_share_alert=0.90, sink_cluster_alert_size=10)
    rows = [_row(topic_id=None, judge_score=0, text="sink")] + [
        _row(judge_score=3, text=f"q{index}") for index in range(9)
    ]

    summary = summarize(rows, params=params, sink_cluster_sizes=[11, 2])

    assert [alert.code for alert in summary.sink.alerts] == [SINK_CLUSTER_ALERT]


def test_a_healthy_run_raises_no_alerts():
    params = _params(sink_share_alert=0.30, sink_cluster_alert_size=10)
    rows = [_row(judge_score=4, text=f"q{index}") for index in range(5)]

    summary = summarize(rows, params=params, sink_cluster_sizes=[3])

    assert summary.sink.alerts == []


# --- the frozen summary ------------------------------------------------------


def test_the_summary_freezes_all_three_breakdowns_as_json():
    """What lands in ``runs.summary``: the report, not the ingredients.

    Frozen because a deleted topic or a lost dataset permission would otherwise
    reshape a historical run and destroy the trend stable topic ids exist to
    carry.
    """
    params = _params(min_scored_questions_per_topic=1)
    summary = summarize(_balanced_rows(), params=params, distinct_ask_count=20)

    frozen = json.loads(json.dumps(summary.to_dict()))

    assert set(frozen) == {
        "overall_score",
        "benchmark_score_pct",
        "unscoped_ask_share",
        "observed_question_count",
        "scored_question_count",
        "curated_question_count",
        "topics",
        "datasets",
        "users",
        "sink",
    }
    assert frozen["overall_score"] == pytest.approx(2.5)
    assert len(frozen["topics"]) == 2
    assert len(frozen["datasets"]) == 2
    assert len(frozen["users"]) == 2
    assert frozen["sink"]["alerts"] == []

    # Ids are strings in the frozen JSON, and the sink keeps its wire literal.
    assert frozen["topics"][0]["topic_id"] in {str(TOPIC_ONE), str(TOPIC_TWO)}
    assert frozen["sink"]["topic_id"] == SINK_TOPIC_ID


def test_the_frozen_summary_keeps_snake_case_keys():
    """The agreed wire contract, and the reason no response model here is an OutDTO."""
    summary = summarize(_balanced_rows(), params=_params(min_scored_questions_per_topic=1))
    frozen = summary.to_dict()

    assert "overall_score" in frozen
    assert "overallScore" not in frozen
    assert "question_count" in frozen["topics"][0]
    assert "questionCount" not in frozen["topics"][0]


# --- ordering ----------------------------------------------------------------


def test_curated_rows_are_pinned_above_observed_rows_then_ranked_by_impact():
    rows = [
        _row(judge_score=1, occurrence_count=8, text="worst observed"),
        _row(judge_score=5, occurrence_count=1, text="best observed"),
        _row(
            source=QuestionSource.CURATED.value,
            judge_score=0,
            occurrence_count=0,
            text="curated",
        ),
        _row(judge_score=None, error="boom", occurrence_count=99, text="unjudged"),
    ]

    ordered = [row.question_text for row in default_row_order(rows)]

    assert ordered == ["curated", "worst observed", "best observed", "unjudged"]


# --- build_rows --------------------------------------------------------------


def _deduped(text: str, *, dataset_id=INFRA, source=QuestionSource.OBSERVED.value, occurrences=2):
    return DedupedQuestion(
        text=text,
        user_id=ANNA,
        dataset_id=dataset_id,
        source=source,
        was_asked=source == QuestionSource.OBSERVED.value,
        occurrence_count=occurrences,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        canonical_index=0,
        ask_indices=[0],
        query_ids=[],
        question_group_id=uuid4(),
    )


def test_build_rows_zips_the_real_phase_three_outputs():
    """Also the guard that the concrete phase-3 classes satisfy this module's protocols."""
    question = _deduped("Where are the runbooks?")
    assignment = TopicAssignment(topic_id=TOPIC_ONE, topic_label="Runbooks", similarity=0.8)
    replayed = ReplayedRow(
        retrieval_context="Some context.", dataset_name="infra-docs", payload_count=1
    )
    judged = JudgedRow(judge_score=2, judge_answered=True, answer="An answer.")

    rows = build_rows(
        [question], [assignment], [replayed], [judged], judge_score_max=5, shared_curated_ids=[]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.question_text == "Where are the runbooks?"
    assert row.topic_id == TOPIC_ONE
    assert row.topic_label == "Runbooks"
    assert row.retrieval_context == "Some context."
    assert row.dataset_name == "infra-docs"
    assert row.judge_score == 2
    assert row.impact == 6.0
    assert row.is_observed
    assert not row.is_sink


def test_build_rows_marks_only_shared_curated_ids_as_benchmark_rows():
    shared = _deduped("shared", source=QuestionSource.CURATED.value, occurrences=0)
    private = _deduped("private", source=QuestionSource.CURATED.value, occurrences=0)

    sink = TopicAssignment(topic_id=None, topic_label=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)
    judged = JudgedRow(judge_score=3, judge_answered=True, answer="a")

    rows = build_rows(
        [shared, private],
        [sink, sink],
        [replayed, replayed],
        [judged, judged],
        judge_score_max=5,
        shared_curated_ids=[shared.curated_question_id],
    )

    assert [row.is_shared_curated for row in rows] == [True, False]


def test_build_rows_prefers_the_resolved_dataset_name_over_the_replays():
    """A row whose replay failed still has a dataset worth naming."""
    question = _deduped("q")
    sink = TopicAssignment(topic_id=None, topic_label=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context=None, dataset_name=None, payload_count=0, error="boom")
    judged = JudgedRow(judge_score=None, judge_answered=None, answer=None, error="boom")

    rows = build_rows(
        [question],
        [sink],
        [replayed],
        [judged],
        judge_score_max=5,
        dataset_names={INFRA: "infra-docs"},
    )

    assert rows[0].dataset_name == "infra-docs"
    assert rows[0].judge_score is None
    assert rows[0].impact is None
    assert rows[0].error == "boom"


def test_build_rows_refuses_misaligned_inputs():
    """A short zip would score one question's context against another question."""
    question = _deduped("q")
    sink = TopicAssignment(topic_id=None, topic_label=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)
    judged = JudgedRow(judge_score=1, judge_answered=True, answer="a")

    with pytest.raises(ValueError, match="index-aligned"):
        build_rows(
            [question, question],
            [sink],
            [replayed, replayed],
            [judged, judged],
            judge_score_max=5,
        )


# --- breakdown details -------------------------------------------------------


def test_breakdowns_are_ordered_by_question_count_descending():
    rows = [_row(user_id=ANNA, text=f"a{index}", judge_score=3) for index in range(3)] + [
        _row(user_id=BEN, text="b0", judge_score=3)
    ]

    assert [user.user_id for user in user_breakdown(rows)] == [ANNA, BEN]


def test_a_dataset_cell_keeps_the_first_non_null_name_it_saw():
    rows = [
        _row(dataset_id=INFRA, dataset_name=None, judge_score=1, text="a"),
        _row(dataset_id=INFRA, dataset_name="infra-docs", judge_score=1, text="b"),
    ]

    assert dataset_breakdown(rows)[0].dataset_name == "infra-docs"


def test_topic_breakdown_never_contains_the_sink():
    rows = [_row(topic_id=None, judge_score=0, text=f"q{index}") for index in range(3)]

    assert topic_breakdown(rows, min_scored_questions_per_topic=1) == []
