"""Unit tests for the pure logic of the memory accuracy score.

These cover the parts where a silent mistake would corrupt the one number the
feature exists to produce, or would spend money nobody authorised:

* ``_allocate_counts`` — the spend split, and that the target is a hard ceiling.
* ``_score_synthetic_question`` — which failures score 0 and which stay
  unmeasured, since that choice decides ``overall_accuracy``'s denominator.
* ``_aggregate_topics`` / ``build_memory_score_document`` — the response shape
  and the reported denominator.
"""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.memory_score.methods.build_topics import Topic, TopicPlan
from cognee.modules.memory_score.methods.generate_questions import _allocate_counts
from cognee.modules.memory_score.methods.run_memory_score import (
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    _aggregate_topics,
    _score_synthetic_question,
    _synthetic_scores,
    build_memory_score_document,
)

# import_module, not `import ... as`: the methods package re-exports the
# run_memory_score FUNCTION under the submodule's own name, so attribute-style
# import would hand back the function instead of the module to patch.
run_module = import_module("cognee.modules.memory_score.methods.run_memory_score")


def _topic(label: str, real_question_count: int = 0) -> Topic:
    return Topic(label=label, node_ids=[f"{label}-node"], real_question_count=real_question_count)


# --------------------------------------------------------------------------
# _allocate_counts
# --------------------------------------------------------------------------


def test_allocate_counts_sums_to_target():
    topics = [_topic("a"), _topic("b", 4), _topic("c", 1)]

    counts = _allocate_counts(topics, 100)

    assert sum(counts) == 100
    assert all(count >= 1 for count in counts)


def test_allocate_counts_weights_by_real_traffic():
    quiet, busy = _topic("quiet"), _topic("busy", 9)

    counts = _allocate_counts([quiet, busy], 100)

    # base weight 1 vs 1+9: the busy topic takes ~10x the discretionary budget.
    assert sum(counts) == 100
    assert counts[1] > counts[0] * 5


def test_allocate_counts_treats_target_as_a_hard_ceiling():
    """A target below the topic count must not be overshot by the per-topic floor."""
    topics = [_topic(label) for label in "abcde"]

    counts = _allocate_counts(topics, 2)

    assert sum(counts) == 2
    assert sorted(counts) == [0, 0, 0, 1, 1]


def test_allocate_counts_spends_a_short_budget_on_the_busiest_topics():
    topics = [_topic("quiet"), _topic("busy", 7), _topic("quieter")]

    counts = _allocate_counts(topics, 1)

    assert counts == [0, 1, 0]


def test_allocate_counts_target_equal_to_topic_count_gives_one_each():
    topics = [_topic("a"), _topic("b", 3)]

    assert _allocate_counts(topics, 2) == [1, 1]


@pytest.mark.parametrize("target", [0, -5])
def test_allocate_counts_zero_or_negative_target_spends_nothing(target):
    assert _allocate_counts([_topic("a"), _topic("b")], target) == [0, 0]


def test_allocate_counts_no_topics():
    assert _allocate_counts([], 100) == []


# --------------------------------------------------------------------------
# _score_synthetic_question: what counts as measured
# --------------------------------------------------------------------------


class _StubJudge:
    """Correctness judge that records whether it was consulted."""

    def __init__(self, verdict=None):
        self.verdict = verdict or {"score": 0.75, "reason": "close enough"}
        self.calls = 0

    async def evaluate_correctness(self, question, answer, golden_answer):
        self.calls += 1
        return self.verdict


def _question():
    return SimpleNamespace(
        text="What did the report conclude?",
        expected_answer="That throughput doubled.",
        topic="reports",
    )


def _run_scoring(monkeypatch, answered, judge):
    """Drive _score_synthetic_question with a stubbed recall result."""

    async def fake_answer(_executor, _question, _golden):
        return answered

    monkeypatch.setattr(run_module, "_answer", fake_answer)

    return asyncio.run(
        _score_synthetic_question(
            executor=object(),
            judge=judge,
            question=_question(),
            semaphore=asyncio.Semaphore(1),
        )
    )


def test_empty_recall_answer_scores_zero_rather_than_vanishing(monkeypatch):
    """An answerable question the memory could not answer is a 0, not a null.

    The question was generated from an ingested chunk, so an empty answer is the
    failure the score exists to measure. Leaving it unmeasured would drop it out
    of overall_accuracy's denominator and inflate the headline number on exactly
    the degraded datasets the feature should flag.
    """
    judge = _StubJudge()

    row = _run_scoring(monkeypatch, {"answer": "   ", "context": "ctx", "error": None}, judge)

    assert row["score"] == 0.0
    assert row["reason"] == "Recall produced an empty answer."
    assert judge.calls == 0, "no judge call should be paid for on an empty answer"
    assert row["grounded"] is None


def test_recall_failure_stays_unmeasured(monkeypatch):
    """An outage is not evidence about the memory, so score stays NULL."""
    judge = _StubJudge()

    row = _run_scoring(
        monkeypatch, {"answer": "", "context": "", "error": "Recall failed: boom"}, judge
    )

    assert row["score"] is None
    assert row["reason"] == "Recall failed: boom"
    assert judge.calls == 0


def test_judge_failure_stays_unmeasured(monkeypatch):
    class _FailingJudge(_StubJudge):
        async def evaluate_correctness(self, question, answer, golden_answer):
            self.calls += 1
            raise RuntimeError("judge exploded")

    judge = _FailingJudge()

    row = _run_scoring(
        monkeypatch, {"answer": "An answer.", "context": "ctx", "error": None}, judge
    )

    assert row["score"] is None
    assert "judge exploded" in row["reason"]
    assert judge.calls == 1


def test_judged_answer_records_the_verdict(monkeypatch):
    judge = _StubJudge({"score": 0.5, "reason": "partly right"})

    row = _run_scoring(
        monkeypatch, {"answer": "An answer.", "context": "ctx", "error": None}, judge
    )

    assert row["score"] == 0.5
    assert row["reason"] == "partly right"
    assert row["source"] == SOURCE_SYNTHETIC
    assert row["grounded"] is None, "synthetic questions never carry a groundedness verdict"


def test_synthetic_scores_keeps_zeros_and_drops_nulls():
    rows = [{"score": 0.0}, {"score": 1.0}, {"score": None}]

    assert _synthetic_scores(rows) == [0.0, 1.0]


# --------------------------------------------------------------------------
# _aggregate_topics
# --------------------------------------------------------------------------


def test_aggregate_topics_reports_null_for_an_unjudged_topic():
    """Null accuracy means unmeasured; 0.0 would read as wrong."""
    plan = TopicPlan(
        topics=[_topic("measured"), _topic("unmeasured")],
        chunk_count=80,
        below_data_floor=False,
        floor_reason=None,
    )
    rows = [
        {"topic": "measured", "score": 1.0},
        {"topic": "unmeasured", "score": None},
    ]

    by_label = {row["topic"]: row for row in _aggregate_topics(plan, rows)}

    assert by_label["measured"]["accuracy"] == 1.0
    assert by_label["unmeasured"]["accuracy"] is None
    assert by_label["unmeasured"]["synthetic_count"] == 1


def test_aggregate_topics_merges_colliding_labels():
    plan = TopicPlan(
        topics=[_topic("billing", 2), _topic("billing", 3)],
        chunk_count=80,
        below_data_floor=False,
        floor_reason=None,
    )

    aggregated = _aggregate_topics(plan, [{"topic": "billing", "score": 0.5}])

    assert len(aggregated) == 1
    assert aggregated[0]["real_count"] == 5
    assert aggregated[0]["from_real_traffic"] is True


# --------------------------------------------------------------------------
# build_memory_score_document
# --------------------------------------------------------------------------


def _run_row(**overrides):
    row = {
        "id": uuid4(),
        "status": "COMPLETED",
        "dataset_id": uuid4(),
        "below_data_floor": False,
        "floor_reason": None,
        "schema_defined": True,
        "overall_accuracy": 0.5,
        "synthetic_question_count": 3,
        "real_question_count": 2,
        "topics": [],
        "created_at": None,
        "completed_at": None,
    }
    row.update(overrides)
    return SimpleNamespace(**row)


def _question_row(**overrides):
    row = {
        "text": "q",
        "topic": None,
        "source": SOURCE_SYNTHETIC,
        "answer": "a",
        "expected_answer": None,
        "score": None,
        "grounded": None,
        "answered": None,
        "reason": None,
    }
    row.update(overrides)
    return SimpleNamespace(**row)


def test_document_reports_the_real_accuracy_denominator():
    """judged_synthetic_question_count is not derivable from the asked count."""
    questions = [
        _question_row(score=1.0),
        _question_row(score=0.0),
        _question_row(score=None),  # unmeasured: recall or judge failed
        _question_row(source=SOURCE_REAL, grounded=True),
    ]

    document = build_memory_score_document(_run_row(), questions)

    assert document["synthetic_question_count"] == 3, "the stored count of questions asked"
    assert document["judged_synthetic_question_count"] == 2, "only the ones with a verdict"


def test_document_lists_only_ungrounded_real_questions():
    questions = [
        _question_row(source=SOURCE_REAL, text="ungrounded", grounded=False),
        _question_row(source=SOURCE_REAL, text="grounded", grounded=True),
        _question_row(source=SOURCE_REAL, text="unmeasured", grounded=None),
        _question_row(source=SOURCE_SYNTHETIC, text="synthetic", grounded=False),
    ]

    document = build_memory_score_document(_run_row(), questions)

    assert document["ungrounded_real_questions"] == ["ungrounded"]


def test_document_shape_is_stable():
    document = build_memory_score_document(_run_row(), [])

    assert set(document) == {
        "run_id",
        "status",
        "dataset_id",
        "below_data_floor",
        "floor_reason",
        "schema_defined",
        "overall_accuracy",
        "synthetic_question_count",
        "judged_synthetic_question_count",
        "real_question_count",
        "coverage",
        "measured_real_question_count",
        "answered_real_question_count",
        "created_at",
        "completed_at",
        "topics",
        "questions",
        "ungrounded_real_questions",
        "unanswerable_real_questions",
    }


# --------------------------------------------------------------------------
# Topic ceiling
# --------------------------------------------------------------------------


def test_topic_ceiling_keeps_per_topic_accuracy_meaningful():
    """More topics than questions makes every per-topic accuracy noise.

    _allocate_counts floors each topic at one question, so the topic count is a
    direct divisor of the question budget. compute_clusters' visualization default
    goes to 12, which leaves 1-2 questions per topic at a 20-question target and a
    per-topic accuracy that can only read 0/50/100%.
    """
    from cognee.modules.memory_score.methods.build_topics import MAX_TOPICS
    from cognee.modules.visualization.semantic_clusters import default_k

    assert MAX_TOPICS < default_k(2000), "the cap must actually bind on a large graph"
    assert MAX_TOPICS >= 3, "must stay at or above build_topics' min_topics=3"

    # At the default target every topic gets a usable sample.
    assert 100 // MAX_TOPICS >= 20

    topics = [_topic(f"t{index}") for index in range(MAX_TOPICS)]
    counts = _allocate_counts(topics, 100)
    assert sum(counts) == 100
    assert min(counts) >= 20


def test_topic_ceiling_does_not_force_clusters_on_a_small_graph():
    """min(default_k, MAX_TOPICS): overriding upward would gate a thin graph.

    default_k shrinks k on a small graph; forcing MAX_TOPICS there would split it
    into clusters below min_nodes_per_topic, failing the floor for a reason that is
    an artifact of the ceiling rather than of the data.
    """
    from cognee.modules.memory_score.methods.build_topics import MAX_TOPICS
    from cognee.modules.visualization.semantic_clusters import default_k

    assert min(default_k(10), MAX_TOPICS) == default_k(10) == 2
    assert min(default_k(5000), MAX_TOPICS) == MAX_TOPICS


# --------------------------------------------------------------------------
# Context batching: making synthetic_target reachable
# --------------------------------------------------------------------------


def test_context_batches_scale_with_the_allocation():
    """One call over 3 chunks grounds ~15 pairs; 39 pairs needs more calls.

    Measured before this existed: synthetic_target=100 over 5 topics produced 59
    questions, because each topic got a single call over MAX_CHUNKS_PER_PROMPT
    chunks no matter how many questions it was allocated.
    """
    from cognee.modules.memory_score.methods.generate_questions import (
        MAX_CHUNKS_PER_PROMPT,
        MAX_PAIRS_PER_CALL,
        _context_batches,
    )

    texts = [f"chunk {index} " + "x" * 300 for index in range(30)]

    assert len(_context_batches(texts, MAX_PAIRS_PER_CALL)) == 1
    assert len(_context_batches(texts, MAX_PAIRS_PER_CALL + 1)) == 2
    assert len(_context_batches(texts, 39)) == -(-39 // MAX_PAIRS_PER_CALL)

    # Distinct chunks per batch — reusing the same three would just re-ask.
    batches = _context_batches(texts, 45)
    assert len(batches) == 3
    assert len(set(batches)) == 3
    assert all(batch for batch in batches)
    assert MAX_CHUNKS_PER_PROMPT == 3


def test_context_batches_bounded_by_available_text():
    """A thin topic generates what it can ground, not what it was allocated."""
    from cognee.modules.memory_score.methods.generate_questions import _context_batches

    assert _context_batches([], 100) == []
    assert _context_batches(["only one chunk " + "y" * 300], 100) == ["only one chunk " + "y" * 300]


def test_context_batches_respect_the_char_budget():
    from cognee.modules.memory_score.methods.generate_questions import (
        MAX_CONTEXT_CHARS,
        _context_batches,
    )

    texts = ["z" * 10_000 for _ in range(9)]
    for batch in _context_batches(texts, 45):
        assert len(batch) <= MAX_CONTEXT_CHARS + 32  # + joiner


# --------------------------------------------------------------------------
# Coverage: the second KPI
# --------------------------------------------------------------------------


def test_coverage_is_separate_from_accuracy():
    """A thin memory scores HIGH on accuracy and LOW on coverage.

    This is the inversion the data-floor gate was added to paper over: questions
    drawn from a handful of chunks are trivially answerable from those same chunks,
    so accuracy alone flatters exactly the customer who needs to upload more. The two
    numbers must therefore stay separate — blended, 0.6 could mean either wrong
    answers or absent data, which are opposite remedies.
    """
    questions = [
        # Perfect recall on everything it actually holds.
        _question_row(source=SOURCE_SYNTHETIC, score=1.0),
        _question_row(source=SOURCE_SYNTHETIC, score=1.0),
        # But it could answer only one of four real questions.
        _question_row(source=SOURCE_REAL, text="held", answered=True, grounded=True),
        _question_row(source=SOURCE_REAL, text="gap a", answered=False, grounded=True),
        _question_row(source=SOURCE_REAL, text="gap b", answered=False, grounded=True),
        _question_row(source=SOURCE_REAL, text="gap c", answered=False, grounded=True),
    ]

    document = build_memory_score_document(_run_row(overall_accuracy=1.0), questions)

    assert document["overall_accuracy"] == 1.0, "accuracy stays flattering"
    assert document["coverage"] == 0.25, "coverage tells the real story"
    assert document["answered_real_question_count"] == 1
    assert document["measured_real_question_count"] == 4
    assert document["unanswerable_real_questions"] == ["gap a", "gap b", "gap c"]


def test_coverage_gap_is_not_reported_as_a_hallucination():
    """An honest refusal is answered=False, grounded=True — opposite failure modes.

    Collapsing them was the original defect: a live run returned grounded=true for a
    question the corpus could not answer, because the refusal's claim really is
    traceable to the context, so ungrounded_real_questions stayed empty for exactly
    the customer the CTA targets.
    """
    questions = [
        _question_row(source=SOURCE_REAL, text="refused", answered=False, grounded=True),
        _question_row(source=SOURCE_REAL, text="made up", answered=True, grounded=False),
    ]

    document = build_memory_score_document(_run_row(), questions)

    assert document["unanswerable_real_questions"] == ["refused"]
    assert document["ungrounded_real_questions"] == ["made up"]


def test_unmeasured_real_questions_leave_coverage_alone():
    """A recall outage must lower confidence in coverage, not coverage itself."""
    questions = [
        _question_row(source=SOURCE_REAL, text="ok", answered=True),
        _question_row(source=SOURCE_REAL, text="outage", answered=None),
    ]

    document = build_memory_score_document(_run_row(), questions)

    assert document["coverage"] == 1.0
    assert document["measured_real_question_count"] == 1, "the NULL row is excluded"
    assert document["unanswerable_real_questions"] == []


def test_coverage_is_null_without_real_questions():
    """No real questions means coverage is unknown, not zero."""
    document = build_memory_score_document(_run_row(), [_question_row(score=1.0)])

    assert document["coverage"] is None
    assert document["measured_real_question_count"] == 0
