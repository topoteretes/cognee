"""Guards on recall-coverage aggregation — phase 4.

Every assertion here is about *which rows* a number is a mean over, because that
is the only way this report can be wrong without looking wrong:

* **``null`` is not ``0.0``.** ``0.0`` says memory answered nothing; ``null`` says
  the run measured nothing. Two tests pin the difference, including the case
  where every score genuinely is 0 and ``memory_score`` must therefore *be*
  ``0.0``.
* **Demand is ``relevance > 0``, never ``source``.** A user-defined question
  nobody asked feeds no average — otherwise a user could raise the headline by
  adding questions memory happens to answer — while one that merged into real
  traffic stays in every mean, or naming a badly-answered question would delete it
  from the report.
* **``memory_score`` averages topics, not rows**, and excludes both
  ``Uncategorized`` and any topic below the scored-row minimum.
* **The sink is a member row of ``topics[]``**, sorted last, with
  ``topic_id: null`` and the label ``Uncategorized``.
* **The summary is frozen**, and its key set is the wire contract.

No LLM, no embedding engine, no database: the phase-3 outputs are constructed
directly, and every parameter comes from ``RecallCoverageConfig(_env_file=None)``
so a developer's ``.env`` cannot move a threshold under the assertions.
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from cognee.modules.recall_coverage.aggregate import (
    CoverageRow,
    SuggestedTopic,
    build_rows,
    default_row_order,
    memory_score,
    run_counters,
    summarize,
    topic_breakdown,
)
from cognee.modules.recall_coverage.assign import TopicAssignment
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.judge import JudgedRow
from cognee.modules.recall_coverage.replay import ReplayedRow
from cognee.modules.recall_coverage.types import (
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

SUGGESTION_ONE = uuid4()
SUGGESTION_TWO = uuid4()


def _params(**overrides) -> CoverageParams:
    """A parameter snapshot that ignores the developer's ``.env``."""
    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _row(
    *,
    user_id: UUID = ANNA,
    dataset_id=INFRA,
    dataset_name="infra-docs",
    topic_id=TOPIC_ONE,
    topic="Runbooks",
    coverage_score=3,
    relevance=1,
    source: str = QuestionSource.OBSERVED.value,
    agent_label="claude-code",
    text: str = "Where are the runbooks?",
    error=None,
) -> CoverageRow:
    """One aggregated row.

    ``relevance`` defaults to 1, i.e. asked once. Pass ``relevance=0`` for the
    user-defined question nobody has asked, and ``source=USER_DEFINED`` with a
    non-zero ``relevance`` for the third case dedup produces — a written-down
    question that merged into real traffic.
    """
    return CoverageRow(
        question=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        agent_label=agent_label if relevance > 0 else None,
        source=source,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        topic_id=topic_id,
        topic=SINK_TOPIC_LABEL if topic_id is None else topic,
        answer=None if coverage_score in (None, 0) else "An answer.",
        coverage_score=coverage_score,
        retrieval_context=None if coverage_score in (None, 0) else "Some context.",
        error=error,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        relevance=relevance,
    )


def _balanced_rows() -> list[CoverageRow]:
    """Six observed rows over two users, two datasets and two equally sized topics.

    Scores sum to 15 over 6 rows, so the row-level mean is 2.5; both topics hold
    three scored rows, so the unweighted mean of the two topic averages is 2.5
    too. That coincidence is what lets a test compare "average the topics" against
    "average the rows" numerically instead of only structurally.
    """
    return [
        _row(user_id=ANNA, dataset_id=INFRA, topic_id=TOPIC_ONE, coverage_score=5, text="a"),
        _row(user_id=ANNA, dataset_id=INFRA, topic_id=TOPIC_ONE, coverage_score=3, text="b"),
        _row(
            user_id=ANNA,
            dataset_id=BILLING,
            dataset_name="billing",
            topic_id=TOPIC_TWO,
            topic="Billing",
            coverage_score=1,
            text="c",
        ),
        _row(
            user_id=BEN,
            dataset_id=INFRA,
            topic_id=TOPIC_TWO,
            topic="Billing",
            coverage_score=2,
            text="d",
        ),
        _row(user_id=BEN, dataset_id=BILLING, dataset_name="billing", coverage_score=4, text="e"),
        _row(
            user_id=BEN,
            dataset_id=BILLING,
            dataset_name="billing",
            topic_id=TOPIC_TWO,
            topic="Billing",
            coverage_score=0,
            text="f",
        ),
    ]


# --- memory_score: null vs 0.0 ----------------------------------------------


def test_memory_score_is_null_when_no_topic_qualifies():
    """Nothing measured is not the same claim as nothing answered."""
    params = _params(min_scored_questions_per_topic=3)
    rows = [_row(coverage_score=5, text=f"q{index}") for index in range(2)]

    summary = summarize(rows, params=params)

    assert summary.memory_score is None
    assert summary.topics[0].memory_score is None
    # The rows still count: the topic was asked about, we just will not put a
    # number on it yet.
    assert summary.topics[0].question_count == 2


def test_memory_score_is_zero_when_a_qualifying_topic_scored_zero():
    """The other half of the same distinction: 0.0 must remain reachable."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [_row(coverage_score=0, text=f"q{index}") for index in range(3)]

    summary = summarize(rows, params=params)

    assert summary.memory_score == 0.0


def test_memory_score_is_null_on_an_empty_run():
    """An empty window completes; it does not report a score of zero."""
    summary = summarize([], params=_params())

    assert summary.memory_score is None
    assert summary.topics == []
    assert summary.suggested_topics == []


# --- memory_score: exclusions ------------------------------------------------


def test_a_topic_below_the_minimum_is_excluded_from_memory_score():
    """One question must not speak for a whole topic."""
    params = _params(min_scored_questions_per_topic=3)
    rows = [
        _row(topic_id=TOPIC_ONE, coverage_score=4, text="a"),
        _row(topic_id=TOPIC_ONE, coverage_score=4, text="b"),
        _row(topic_id=TOPIC_ONE, coverage_score=4, text="c"),
        _row(topic_id=TOPIC_TWO, topic="Billing", coverage_score=0, text="d"),
        _row(topic_id=TOPIC_TWO, topic="Billing", coverage_score=0, text="e"),
    ]

    summary = summarize(rows, params=params)
    by_id = {topic.topic_id: topic for topic in summary.topics}

    assert by_id[TOPIC_TWO].memory_score is None
    assert by_id[TOPIC_TWO].question_count == 2
    # Only the qualifying topic's average survives into the headline.
    assert summary.memory_score == 4.0


def test_memory_score_averages_topics_not_rows():
    """The one deliberate asymmetry: a chatty topic must not become the headline."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [_row(topic_id=TOPIC_ONE, coverage_score=0, text=f"q{index}") for index in range(9)] + [
        _row(topic_id=TOPIC_TWO, topic="Billing", coverage_score=10, text="rare")
    ]

    summary = summarize(rows, params=params)

    # Averaging rows would give 1.0; averaging the two topics gives 5.0.
    assert summary.memory_score == 5.0


def test_the_sink_is_a_member_of_topics_and_excluded_from_memory_score():
    """``topic_id: null`` / ``Uncategorized`` is one line of the same table.

    Excluded from the headline because it is the absence of a topic: averaging it
    in would let an unplaceable question lower the score of a taxonomy that never
    claimed to cover it. Its own row still reports its average — that is the
    "your taxonomy is missing something" signal.
    """
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(topic_id=TOPIC_ONE, coverage_score=4, text="a"),
        _row(topic_id=None, coverage_score=0, text="b"),
        _row(topic_id=None, coverage_score=0, text="c"),
    ]

    summary = summarize(rows, params=params)
    sink = next(topic for topic in summary.topics if topic.topic_id is None)

    assert summary.memory_score == 4.0
    assert sink.topic == SINK_TOPIC_LABEL
    assert sink.question_count == 2
    assert sink.memory_score == 0.0


def test_the_sink_is_sorted_last_however_big_it_is():
    """Reading the sink as the biggest topic is exactly the wrong impression."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [_row(topic_id=None, coverage_score=0, text=f"s{index}") for index in range(5)] + [
        _row(topic_id=TOPIC_ONE, coverage_score=4, text="a")
    ]

    topics = summarize(rows, params=params).topics

    assert [topic.topic_id for topic in topics] == [TOPIC_ONE, None]


def test_real_topics_are_ordered_by_question_count_descending():
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(topic_id=TOPIC_ONE, coverage_score=3, text="a"),
        _row(topic_id=TOPIC_TWO, topic="Billing", coverage_score=3, text="b"),
        _row(topic_id=TOPIC_TWO, topic="Billing", coverage_score=3, text="c"),
    ]

    topics = summarize(rows, params=params).topics

    assert [topic.topic_id for topic in topics] == [TOPIC_TWO, TOPIC_ONE]


def test_an_unscored_row_is_absent_from_every_mean_but_present_in_the_count():
    """ "We could not ask" is not evidence about memory, and is not a zero."""
    params = _params(min_scored_questions_per_topic=1)
    rows = [
        _row(coverage_score=4, text="a"),
        _row(coverage_score=None, error="replay failed", text="b"),
    ]

    summary = summarize(rows, params=params)

    assert summary.topics[0].question_count == 2
    assert summary.topics[0].memory_score == 4.0
    assert summary.memory_score == 4.0


# --- demand is relevance, not source ----------------------------------------


def test_a_never_asked_user_defined_row_feeds_no_average():
    """Otherwise a user could raise the headline by adding answerable questions."""
    params = _params(min_scored_questions_per_topic=1)
    rows = _balanced_rows() + [
        _row(
            source=QuestionSource.USER_DEFINED.value,
            coverage_score=10,
            relevance=0,
            text="never asked",
        ),
    ]

    summary = summarize(rows, params=params)

    # Adding a perfectly answered question nobody asked must not move the number.
    assert summary.memory_score == pytest.approx(2.5)
    assert sum(topic.question_count for topic in summary.topics) == 6


def test_a_user_defined_question_that_merged_with_traffic_still_counts_as_demand():
    """Naming a badly-answered question must not delete it from the headline.

    ``dedup_asks`` marks a whole cluster ``source = "user_defined"`` as soon as one
    member is, keeping that partition's ``relevance``. Filtering the averages on
    the source instead of on ``relevance`` would drop those genuinely asked rows
    out of every mean — so anyone able to add a question could move
    ``memory_score`` by naming ones they know score badly.
    """
    params = _params(min_scored_questions_per_topic=1)
    badly = _row(coverage_score=1, relevance=3, text="Where do escalations go?")
    well = _row(coverage_score=5, relevance=1, text="Where is the billing runbook?")

    before = summarize([badly, well], params=params)
    after = summarize(
        [
            # The same ask, after a human wrote down matching text: one merged row.
            _row(
                source=QuestionSource.USER_DEFINED.value,
                coverage_score=1,
                relevance=3,
                text="Where do escalations go?",
            ),
            well,
        ],
        params=params,
    )

    assert before.memory_score == pytest.approx(3.0)
    assert after.memory_score == pytest.approx(3.0)
    assert [topic.question_count for topic in after.topics] == [2]


def test_a_merged_user_defined_row_is_still_pinned_above_observed_rows():
    """The pin is on the source; the averages are on ``relevance``."""
    merged = _row(source=QuestionSource.USER_DEFINED.value, coverage_score=5, relevance=1)
    observed = _row(coverage_score=0, relevance=9, text="asked a lot, answered never")

    assert default_row_order([observed, merged]) == [merged, observed]


def test_a_topic_whose_only_rows_were_never_asked_is_still_listed_and_named():
    """Otherwise the report returns a real ``topic_id`` labelled ``Uncategorized``.

    ``topics[]`` is the only place the frozen summary carries topic labels, so a
    topic missing from it cannot be named by the read endpoint at all.
    """
    params = _params(min_scored_questions_per_topic=1)
    speculative = uuid4()
    rows = _balanced_rows() + [
        _row(
            source=QuestionSource.USER_DEFINED.value,
            relevance=0,
            topic_id=speculative,
            topic="Billing & invoices",
            coverage_score=4,
            text="cur",
        )
    ]

    summary = summarize(rows, params=params)
    cell = next(topic for topic in summary.topics if topic.topic_id == speculative)

    assert cell.topic == "Billing & invoices"
    # The count and the mean are over asked rows, and this topic has none.
    assert (cell.question_count, cell.memory_score) == (0, None)
    assert summary.memory_score == pytest.approx(2.5)


# --- suggested topics --------------------------------------------------------


def test_suggested_topics_are_frozen_into_the_summary():
    """A per-run output: the review queue moves, a historical run must not.

    ``cohesion`` orders the candidates inside ``suggest.py`` and says nothing
    about memory, so it is deliberately absent — a number next to a proposed topic
    would be read as a score.
    """
    params = _params(min_scored_questions_per_topic=1)

    summary = summarize(
        _balanced_rows(),
        params=params,
        suggested_topics=[
            SuggestedTopic(
                suggestion_id=SUGGESTION_ONE, label="Deploy rollbacks", question_count=7
            ),
            SuggestedTopic(suggestion_id=SUGGESTION_TWO, label="SSO onboarding", question_count=5),
        ],
    )

    assert [(item.label, item.question_count) for item in summary.suggested_topics] == [
        ("Deploy rollbacks", 7),
        ("SSO onboarding", 5),
    ]
    # The id travels too: the run report is the only place the dismiss route's
    # ``suggestion_id`` is published, so freezing without it makes half the review
    # flow unreachable.
    assert summary.to_dict()["suggested_topics"] == [
        {
            "suggestion_id": str(SUGGESTION_ONE),
            "label": "Deploy rollbacks",
            "question_count": 7,
        },
        {"suggestion_id": str(SUGGESTION_TWO), "label": "SSO onboarding", "question_count": 5},
    ]


def test_a_run_that_proposed_nothing_reports_an_empty_list():
    """Not null: "no suggestions" is a real answer and the UI renders a list."""
    assert summarize(_balanced_rows(), params=_params()).to_dict()["suggested_topics"] == []


# --- counters ----------------------------------------------------------------


def test_the_run_counters_are_the_window_and_the_judged_rows():
    """``recall_count`` describes the window; ``question_count`` the rows judged.

    Both live on the run row because the window they describe is gone by read time,
    and they differ: a window truncated by ``window_row_cap``, or one of loops the
    cooldown swallowed, is only visible in the gap between them.
    """
    rows = _balanced_rows()

    counters = run_counters(rows, recall_count=42)

    assert counters.recall_count == 42
    assert counters.question_count == 6
    # A user-defined row nobody asked is still a row the run judged.
    assert (
        run_counters(
            rows + [_row(source=QuestionSource.USER_DEFINED.value, relevance=0, text="cur")],
            recall_count=42,
        ).question_count
        == 7
    )


# --- the frozen summary ------------------------------------------------------


def test_the_summary_freezes_the_report_as_json():
    """What lands in ``runs.summary``: the report, not the ingredients.

    Frozen because a deleted topic or a lost dataset permission would otherwise
    reshape a historical run and destroy the trend stable topic ids exist to carry.
    """
    params = _params(min_scored_questions_per_topic=1)
    summary = summarize(
        _balanced_rows() + [_row(topic_id=None, coverage_score=1, text="unplaceable")],
        params=params,
        suggested_topics=[
            SuggestedTopic(suggestion_id=SUGGESTION_ONE, label="Deploy rollbacks", question_count=7)
        ],
    )

    frozen = json.loads(json.dumps(summary.to_dict()))

    assert set(frozen) == {"memory_score", "topics", "suggested_topics"}
    assert frozen["memory_score"] == pytest.approx(2.5)
    # Two real topics plus the sink, and ids are strings in the frozen JSON.
    assert len(frozen["topics"]) == 3
    assert frozen["topics"][0]["topic_id"] in {str(TOPIC_ONE), str(TOPIC_TWO)}
    assert frozen["topics"][-1] == {
        "topic_id": None,
        "topic": SINK_TOPIC_LABEL,
        "question_count": 1,
        "memory_score": 1.0,
    }


def test_the_frozen_summary_keeps_snake_case_keys():
    """The agreed wire contract, and the reason no response model here is an OutDTO."""
    summary = summarize(_balanced_rows(), params=_params(min_scored_questions_per_topic=1))
    frozen = summary.to_dict()

    assert "memory_score" in frozen
    assert "memoryScore" not in frozen
    assert "question_count" in frozen["topics"][0]
    assert "questionCount" not in frozen["topics"][0]


# --- ordering ----------------------------------------------------------------


def test_user_defined_rows_are_pinned_above_observed_rows_then_ranked_by_demand():
    """Two incomparable kinds of importance, so the pin is not a weight.

    Deliberately not by score either: the report is a table the UI sorts, and a
    default order by score would make what the reader sees depend on which rows
    happened to be judged — so the unjudged row below is ranked by its demand like
    any other.
    """
    rows = [
        _row(coverage_score=1, relevance=8, text="worst observed"),
        _row(coverage_score=5, relevance=1, text="best observed"),
        _row(
            source=QuestionSource.USER_DEFINED.value,
            coverage_score=0,
            relevance=0,
            text="user defined",
        ),
        _row(coverage_score=None, error="boom", relevance=99, text="unjudged"),
    ]

    ordered = [row.question for row in default_row_order(rows)]

    assert ordered == ["user defined", "unjudged", "worst observed", "best observed"]


# --- build_rows --------------------------------------------------------------


def _deduped(
    text: str,
    *,
    dataset_id=INFRA,
    source=QuestionSource.OBSERVED.value,
    relevance=2,
    session_id=None,
):
    return DedupedQuestion(
        text=text,
        user_id=ANNA,
        dataset_id=dataset_id,
        source=source,
        relevance=relevance,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        canonical_index=0,
        ask_indices=[0],
        query_ids=[],
        session_id=session_id,
    )


def test_build_rows_zips_the_real_phase_three_outputs():
    """Also the guard that the concrete phase-3 classes satisfy this module's protocols."""
    question = _deduped("Where are the runbooks?")
    assignment = TopicAssignment(topic_id=TOPIC_ONE, topic="Runbooks", similarity=0.8)
    replayed = ReplayedRow(
        retrieval_context="Some context.", dataset_name="infra-docs", payload_count=1
    )
    judged = JudgedRow(coverage_score=2, answer="An answer.")

    rows = build_rows([question], [assignment], [replayed], [judged])

    assert len(rows) == 1
    row = rows[0]
    assert row.question == "Where are the runbooks?"
    assert row.topic_id == TOPIC_ONE
    assert row.topic == "Runbooks"
    assert row.retrieval_context == "Some context."
    assert row.dataset_name == "infra-docs"
    assert row.coverage_score == 2
    assert row.relevance == 2
    assert row.is_observed
    assert not row.is_sink
    assert not row.is_curated


def test_build_rows_resolves_each_rows_agent_from_its_own_session():
    """Not from the run's label: the default run is ``all``, and only the row knows.

    A callable rather than a fifth aligned list, because this module is
    deliberately config-free and the prefix map lives in the config.
    """
    codex = _deduped("a", session_id="codex_a1")
    claude = _deduped("b", session_id="claude_a1")
    no_session = _deduped("c", session_id=None)

    assignment = TopicAssignment(topic_id=TOPIC_ONE, topic="Runbooks", similarity=0.8)
    replayed = ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)
    judged = JudgedRow(coverage_score=4, answer="a")

    rows = build_rows(
        [codex, claude, no_session],
        [assignment] * 3,
        [replayed] * 3,
        [judged] * 3,
        agent_label_of=lambda session_id: {"codex_a1": "codex", "claude_a1": "claude-code"}.get(
            session_id, "api"
        ),
    )

    # An asked row with no session is "api" — the label for no session at all.
    assert [row.agent_label for row in rows] == ["codex", "claude-code", "api"]


def test_a_user_defined_row_nobody_asked_has_no_agent():
    """Gated on demand: nobody asked it, so no agent can be responsible for it."""
    never_asked = _deduped("cur", source=QuestionSource.USER_DEFINED.value, relevance=0)
    assignment = TopicAssignment(topic_id=None, topic=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)
    judged = JudgedRow(coverage_score=3, answer="a")

    rows = build_rows(
        [never_asked],
        [assignment],
        [replayed],
        [judged],
        agent_label_of=lambda session_id: "api",
    )

    assert rows[0].agent_label is None
    assert rows[0].relevance == 0
    assert not rows[0].is_observed
    assert rows[0].is_curated


def test_build_rows_leaves_the_agent_null_when_no_resolver_is_given():
    """aggregate.py imports no config, so without the callable there is no label."""
    rows = build_rows(
        [_deduped("a", session_id="codex_a1")],
        [TopicAssignment(topic_id=TOPIC_ONE, topic="Runbooks", similarity=0.8)],
        [ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)],
        [JudgedRow(coverage_score=1, answer="a")],
    )

    assert rows[0].agent_label is None


def test_build_rows_prefers_the_resolved_dataset_name_over_the_replays():
    """A row whose replay failed still has a dataset worth naming."""
    question = _deduped("q")
    sink = TopicAssignment(topic_id=None, topic=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context=None, dataset_name=None, payload_count=0, error="boom")
    judged = JudgedRow(coverage_score=None, answer=None, error="boom")

    rows = build_rows(
        [question],
        [sink],
        [replayed],
        [judged],
        dataset_names={INFRA: "infra-docs"},
    )

    assert rows[0].dataset_name == "infra-docs"
    assert rows[0].coverage_score is None
    assert rows[0].error == "boom"
    assert rows[0].is_sink


def test_build_rows_bounds_the_stored_context_but_not_the_judged_one():
    """``store_context_max_chars`` is a column bound, applied at persistence time.

    The score in the same row was computed over the whole context (see the judge's
    own guard), so the truncation here is an excerpt of the evidence and never the
    evidence itself. At ``0`` the column is NULL and the score is untouched.
    """
    question = _deduped("q")
    sink = TopicAssignment(topic_id=None, topic=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context="x" * 500, dataset_name=None, payload_count=1)
    judged = JudgedRow(coverage_score=4, answer="a")

    def _row_with(limit):
        return build_rows(
            [question],
            [sink],
            [replayed],
            [judged],
            store_context_max_chars=limit,
        )[0]

    assert _row_with(100).retrieval_context == "x" * 100
    assert _row_with(None).retrieval_context == "x" * 500
    assert _row_with(0).retrieval_context is None
    # The score is the judge's, whatever the column keeps.
    assert {_row_with(limit).coverage_score for limit in (0, 100, None)} == {4}


def test_build_rows_refuses_misaligned_inputs():
    """A short zip would score one question's context against another question."""
    question = _deduped("q")
    sink = TopicAssignment(topic_id=None, topic=SINK_TOPIC_LABEL, similarity=0.0)
    replayed = ReplayedRow(retrieval_context="c", dataset_name=None, payload_count=1)
    judged = JudgedRow(coverage_score=1, answer="a")

    with pytest.raises(ValueError, match="index-aligned"):
        build_rows(
            [question, question],
            [sink],
            [replayed, replayed],
            [judged, judged],
        )


# --- breakdown details -------------------------------------------------------


def test_topic_breakdown_contains_the_sink_as_a_member_row():
    """It used to be a block of its own, which made every reader special-case it."""
    rows = [_row(topic_id=None, coverage_score=0, text=f"q{index}") for index in range(3)]

    topics = topic_breakdown(rows, min_scored_questions_per_topic=1)

    assert [(topic.topic_id, topic.topic) for topic in topics] == [(None, SINK_TOPIC_LABEL)]
    assert topics[0].question_count == 3
    # And it is still out of the headline, which is a mean over real topics.
    assert memory_score(topics) is None


def test_the_sink_row_is_labelled_uncategorized_whatever_the_row_carried():
    """The label is derived: ``Uncategorized`` is not a row in the topics table."""
    rows = [_row(topic_id=None, topic="something else", coverage_score=1, text="a")]

    assert topic_breakdown(rows, min_scored_questions_per_topic=1)[0].topic == SINK_TOPIC_LABEL
