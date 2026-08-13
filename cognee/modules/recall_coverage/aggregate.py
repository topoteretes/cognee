"""Turn judged rows into question rows and one frozen summary.

Phase 4 of a recall-coverage run. Pure functions over value objects: nothing here
calls an LLM, an embedding engine or the database — persistence is
:mod:`cognee.modules.recall_coverage.repository`, which imports this module. That
direction is why the three phase-3 inputs arrive as small protocols instead of the
concrete ``TopicAssignment`` / ``ReplayedRow`` / ``JudgedRow`` classes: importing
them would drag the LLM, embedding and search stacks into the persistence layer
for the sake of five attributes.

The report is one flat question table plus one grouping of it. Four rules decide
what the numbers mean, and each exists because its absence produces a number that
looks right and is not:

* **A row counts as demand when it was asked, which is ``relevance > 0`` and
  never ``source``.** Dedup marks a whole cluster ``user_defined`` as soon as one
  member is, so grouping on the source would drop genuinely asked traffic out of
  the headline the moment somebody writes down a question resembling it — and a
  user could then move the headline by naming questions they know score badly. A
  user-defined question nobody has asked has ``relevance: 0``, appears in
  ``questions[]``, and feeds no average: a user who could raise ``memory_score``
  by adding questions memory happens to answer would be able to game it, and one
  who could lower it by adding aspirational ones would stop adding them. See
  :attr:`CoverageRow.is_observed`.
* **``memory_score`` averages topics, not rows** — the one deliberate asymmetry.
  A single chatty dataset would otherwise dominate the headline just by asking
  more, and the point of the taxonomy is that "Billing 4.2, Runbooks 1.8" is a
  sentence about coverage rather than about traffic volume. Excluded from that
  mean: the ``Uncategorized`` row (it is the absence of a topic, not a topic) and
  any topic with fewer than ``min_scored_questions_per_topic`` scored rows, which
  reports ``memory_score: null`` rather than letting one question speak for a
  theme.
* **``null``, never ``0.0``, when nothing qualifies.** ``0.0`` is a real score
  meaning "memory answered nothing"; a run with no qualifying topic has not
  measured that. Conflating the two turns "we do not know" into the worst possible
  verdict, and a trend line built on it is fiction.
* **A NULL score is not a zero.** A row whose replay or judge failed carries no
  score and is absent from every mean — "we could not ask" is not evidence about
  memory.

Nothing is rounded. Presentation precision is the UI's business, and a rounding
constant here would be the one magic number in the module.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from uuid import UUID

from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_LABEL,
    CoverageParams,
    QuestionSource,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")


class AssignmentLike(Protocol):
    """What a topic assignment contributes to a row: where it landed.

    Satisfied by :class:`cognee.modules.recall_coverage.assign.TopicAssignment`.
    ``topic_id is None`` is the sink, reported as ``topic_id: null`` with the label
    ``"Uncategorized"``.
    """

    topic_id: Optional[UUID]
    topic: str


class ReplayedLike(Protocol):
    """What a replayed row contributes: the context, and the dataset it named.

    Satisfied by :class:`cognee.modules.recall_coverage.replay.ReplayedRow`.
    """

    retrieval_context: Optional[str]
    dataset_name: Optional[str]


class JudgedLike(Protocol):
    """What the judge contributes.

    Satisfied by :class:`cognee.modules.recall_coverage.judge.JudgedRow`. Note
    that ``error`` being set does **not** imply ``coverage_score`` is ``None``: a
    coverage score that survived a failed answer generation is kept, because it is
    the number every mean here is over.
    """

    coverage_score: Optional[int]
    answer: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class CoverageRow:
    """One question row, judged, attributed and ready to persist.

    Mirrors ``recall_coverage_questions`` one-for-one with a single exception:
    ``topic`` is derived, because the label lives on the topic row and
    ``"Uncategorized"`` is not a row at all.
    """

    question: str
    user_id: UUID
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    agent_label: Optional[str]
    source: str
    curated_question_id: Optional[UUID]
    topic_id: Optional[UUID]
    topic: str
    answer: Optional[str]
    coverage_score: Optional[int]
    retrieval_context: Optional[str]
    error: Optional[str]
    first_asked_at: Optional[datetime]
    last_asked_at: Optional[datetime]
    relevance: int

    @property
    def is_curated(self) -> bool:
        """True when a human wrote this row's text — provenance, not participation.

        Dedup marks a cluster ``source = "user_defined"`` as soon as one member is,
        so a written-down question that landed within ``dedup_threshold`` of real
        traffic carries this *and* a non-zero ``relevance``. This is what pins
        user-defined rows to the top of the report; it is deliberately not what the
        averages filter on — see :attr:`is_observed`.
        """
        return self.source == QuestionSource.USER_DEFINED.value

    @property
    def is_observed(self) -> bool:
        """True when an agent really asked this, whoever wrote the text.

        ``relevance`` counts distinct asks and never counts the user-defined
        member, so this is exactly "somebody asked it". Filtering on ``source``
        instead would delete real, already-counted traffic from ``memory_score``
        the moment a human writes down a question resembling it: the observed row
        merges, becomes ``user_defined``, and drops out of every mean. A user could
        then move the headline number by naming questions they know score badly.

        A merged row was asked N times before the written-down text existed and is
        asked N times after, so it stays in every mean. A row nobody asked
        (``relevance == 0``) is nobody's demand yet and feeds no average.
        """
        return self.relevance > 0

    @property
    def is_sink(self) -> bool:
        """True when the row matched no topic. Tested on the id, never on the label."""
        return self.topic_id is None

    @property
    def is_scored(self) -> bool:
        return self.coverage_score is not None


@dataclass(frozen=True)
class TopicScore:
    """One row of ``topics[]``. ``topic_id is None`` is the ``Uncategorized`` row."""

    topic_id: Optional[UUID]
    topic: str
    question_count: int
    memory_score: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": None if self.topic_id is None else str(self.topic_id),
            "topic": self.topic,
            "question_count": self.question_count,
            "memory_score": self.memory_score,
        }


@dataclass(frozen=True)
class SuggestedTopic:
    """One pending suggestion, as the run report carries it.

    A per-run output, so it is frozen into ``summary`` alongside the topics rather
    than read live: the review queue moves as the owner accepts and dismisses, and
    a historical run must keep showing what it actually proposed. ``cohesion``
    stays internal — it orders the candidates
    (:mod:`cognee.modules.recall_coverage.suggest`) and says nothing about memory.

    ``suggestion_id`` is carried because the report is the only place a client can
    learn it, and ``POST /suggestions/{suggestion_id}/dismiss`` is keyed on it:
    without it the dismiss half of "accept or dismiss the topics Cognee proposes"
    addresses nothing. Accepting needs no id — that goes by label through
    ``POST /topics``, which is why the id is required here rather than optional: a
    suggestion frozen without one is a row the owner can only accept.
    """

    suggestion_id: UUID
    label: str
    question_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": str(self.suggestion_id),
            "label": self.label,
            "question_count": self.question_count,
        }


@dataclass(frozen=True)
class CoverageSummary:
    """The frozen report. Serialized into ``recall_coverage_runs.summary``.

    Frozen rather than recomputed at read time because it names things that
    change: delete a topic and its historical rows would silently move into
    ``Uncategorized``; lose access to a dataset and it would vanish from a run that
    genuinely covered it. Either would reshape a past run and destroy the trend
    that stable topic ids exist to carry.
    """

    memory_score: Optional[float]
    topics: list[TopicScore]
    suggested_topics: list[SuggestedTopic]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_score": self.memory_score,
            "topics": [topic.to_dict() for topic in self.topics],
            "suggested_topics": [suggestion.to_dict() for suggestion in self.suggested_topics],
        }


@dataclass(frozen=True)
class RunCounters:
    """The two counters written onto the run row when it completes.

    ``recall_count`` counts raw ``queries`` rows in the window, so a truncated
    window is visible; ``question_count`` counts the rows this run judged. Both are
    on the run row rather than derived at read time because the window they
    describe is gone by then.
    """

    recall_count: int
    question_count: int


def _mean(values: Sequence[float]) -> Optional[float]:
    """The mean, or ``None`` for an empty sequence — never ``0.0``."""
    return (sum(values) / len(values)) if values else None


def _scores(rows: Sequence[CoverageRow]) -> list[float]:
    return [float(row.coverage_score) for row in rows if row.coverage_score is not None]


def _stored_context(
    context: Optional[str], store_context_max_chars: Optional[int]
) -> Optional[str]:
    """The judged context, bounded to what the column keeps. NULL, never ``""``.

    The bound is a storage decision and nothing else: the score in the same row was
    computed over the whole context, so a reader who finds this field cut off is
    looking at an excerpt of the evidence, not at the evidence the judge saw.
    """
    if context is None or store_context_max_chars is None:
        return context
    return context[:store_context_max_chars] or None


def build_rows(
    questions: Sequence[DedupedQuestion],
    assignments: Sequence[AssignmentLike],
    replayed: Sequence[ReplayedLike],
    judged: Sequence[JudgedLike],
    *,
    store_context_max_chars: Optional[int] = None,
    dataset_names: Optional[Mapping[UUID, str]] = None,
    agent_label_of: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> list[CoverageRow]:
    """Zip the four index-aligned phase outputs into persistable question rows.

    The four lists are aligned by **position** and nothing else — dedup,
    assignment, replay and judging all preserve the order they were handed — so a
    length mismatch raises rather than being zipped short. A silent
    ``zip``-shortest here would not fail: it would attribute one question's
    retrieved context and score to a different question, in a report whose whole
    purpose is to say which questions memory cannot answer.

    ``dataset_names`` is preferred over the name the replay reported, because a
    row whose replay failed still has a dataset worth naming. The replay's name is
    the fallback, and it is only ever set when the search resolved to exactly one
    dataset.

    ``agent_label_of`` resolves one raw ``session_id`` into a label; the pipeline
    passes :func:`cognee.modules.recall_coverage.agent_scope.classify_session`. A
    callable rather than a fifth aligned list, because this module is deliberately
    config-free and the prefix map lives in the config. It is applied **only to
    rows that were asked**: a user-defined question nobody asked has no session and
    reports ``agent: null``, while an asked row whose session id is NULL is
    correctly ``api`` — the label for "no session at all".

    ``store_context_max_chars`` is applied to ``retrieval_context`` **here**, which
    is the only place it belongs: it is a bound on what the column holds, and the
    judge has already scored the whole retrieved context by the time these rows are
    built. Truncating in the replay instead would let a storage knob decide the
    score. ``None`` stores the context whole.
    """
    lengths = {
        "questions": len(questions),
        "assignments": len(assignments),
        "replayed": len(replayed),
        "judged": len(judged),
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "recall-coverage aggregation requires index-aligned inputs, got "
            + ", ".join(f"{name}={count}" for name, count in lengths.items())
        )

    names = dict(dataset_names or {})

    rows: list[CoverageRow] = []
    for question, assignment, replay, verdict in zip(questions, assignments, replayed, judged):
        dataset_name = names.get(question.dataset_id) if question.dataset_id else None
        stored_context = _stored_context(replay.retrieval_context, store_context_max_chars)
        asked = question.relevance > 0

        rows.append(
            CoverageRow(
                question=question.text,
                user_id=question.user_id,
                dataset_id=question.dataset_id,
                dataset_name=dataset_name or replay.dataset_name,
                agent_label=(
                    agent_label_of(question.session_id) if (agent_label_of and asked) else None
                ),
                source=question.source,
                curated_question_id=question.curated_question_id,
                topic_id=assignment.topic_id,
                topic=assignment.topic,
                answer=verdict.answer,
                coverage_score=verdict.coverage_score,
                retrieval_context=stored_context,
                error=verdict.error,
                first_asked_at=question.first_asked_at,
                last_asked_at=question.last_asked_at,
                relevance=question.relevance,
            )
        )

    return rows


def report_order_key(row: Any) -> tuple:
    """Sort key for the default order of ``questions[]``: user-defined first, then demand.

    User-defined rows are **pinned** above observed rows rather than given a
    synthetic weight that would let them compete on the same scale: "somebody cares
    about this" and "this was asked forty times" are two incomparable kinds of
    importance, and blending them into one number invents an exchange rate between
    them. Within each half, most-asked first, then by text so the order is stable.

    Deliberately not by score: the report is a table the UI sorts, and a default
    order that put the worst-answered rows first would make the number the reader
    sees depend on which rows happened to be judged.

    Duck-typed on ``is_curated``, ``relevance`` and ``question`` so the write side
    (:class:`CoverageRow`) and the read side (``repository.QuestionRecord``) cannot
    drift into two different default orders. The pin is on the **source**, not on
    ``relevance``: a written-down question that merged into real traffic is still
    the row a human asked for.
    """
    return (not row.is_curated, -row.relevance, row.question)


def default_row_order(rows: Sequence[CoverageRow]) -> list[CoverageRow]:
    """Rows in the default report order. See :func:`report_order_key`."""
    return sorted(rows, key=report_order_key)


def topic_breakdown(
    rows: Sequence[CoverageRow], *, min_scored_questions_per_topic: int
) -> list[TopicScore]:
    """Per-topic means over asked rows, with ``Uncategorized`` as the last row.

    The sink is a member of this list rather than a block of its own: "questions
    the taxonomy could not place" is one line of the same table, and giving it a
    separate shape made every reader special-case it. It is sorted last rather than
    by count, because it is the absence of a topic and reading it as the biggest
    topic is exactly the wrong impression.

    ``memory_score`` is ``null`` — and the topic is therefore out of the run's
    ``memory_score`` — below ``min_scored_questions_per_topic`` scored rows. The
    rows still count in ``question_count``: the topic was asked about, we just will
    not put a number on it yet.

    **Every topic that received a row appears**, including one whose only rows
    nobody asked: it then reports ``question_count: 0`` and ``memory_score: null``.
    It is listed rather than omitted because this breakdown is the only place the
    frozen summary carries topic *labels*, so a topic missing from here comes back
    out of the run report as a real ``topic_id`` with no name.
    """
    grouped: dict[Optional[UUID], list[CoverageRow]] = {}
    labels: dict[Optional[UUID], str] = {}

    for row in rows:
        grouped.setdefault(row.topic_id, [])
        labels.setdefault(row.topic_id, SINK_TOPIC_LABEL if row.topic_id is None else row.topic)
        if row.is_observed:
            grouped[row.topic_id].append(row)

    topics: list[TopicScore] = []
    for topic_id, members in grouped.items():
        scores = _scores(members)
        qualifies = len(scores) >= min_scored_questions_per_topic
        topics.append(
            TopicScore(
                topic_id=topic_id,
                topic=labels[topic_id],
                question_count=len(members),
                memory_score=_mean(scores) if qualifies else None,
            )
        )

    # Real topics by size, then the sink. ``topic_id is None`` sorts last.
    return sorted(
        topics, key=lambda topic: (topic.topic_id is None, -topic.question_count, topic.topic)
    )


def memory_score(topics: Sequence[TopicScore]) -> Optional[float]:
    """Unweighted mean of the qualifying topic averages. ``None`` when none qualify.

    Unweighted so a dataset that asks ten times as much does not become the
    headline; ``None`` rather than ``0.0`` because a run with no qualifying topic
    has measured nothing, while ``0.0`` is the real and much worse claim that
    memory answered nothing.

    ``Uncategorized`` is excluded: it is the absence of a topic, and averaging it
    in would let an unplaceable question lower the score of a taxonomy that never
    claimed to cover it. Its own row still reports its average, which is the
    signal that the taxonomy is missing something.
    """
    return _mean(
        [
            topic.memory_score
            for topic in topics
            if topic.topic_id is not None and topic.memory_score is not None
        ]
    )


def summarize(
    rows: Sequence[CoverageRow],
    *,
    params: CoverageParams,
    suggested_topics: Sequence[SuggestedTopic] = (),
) -> CoverageSummary:
    """Build the frozen summary from the run's rows.

    Deliberately a pure function of ``rows`` plus the run's parameters and this
    run's suggestions: the same inputs always produce the same summary, which is
    what makes freezing it into ``runs.summary`` equivalent to the report a reader
    would have got at the time.
    """
    topics = topic_breakdown(
        rows, min_scored_questions_per_topic=params.min_scored_questions_per_topic
    )

    summary = CoverageSummary(
        memory_score=memory_score(topics),
        topics=topics,
        suggested_topics=list(suggested_topics),
    )

    logger.debug(
        "recall_coverage: summarized %s rows into %s topics, memory_score=%s",
        len(rows),
        len(topics),
        summary.memory_score,
    )
    return summary


def run_counters(rows: Sequence[CoverageRow], *, recall_count: int) -> RunCounters:
    """The two counters the run row reports."""
    return RunCounters(recall_count=recall_count, question_count=len(rows))


__all__ = [
    "AssignmentLike",
    "CoverageRow",
    "CoverageSummary",
    "JudgedLike",
    "ReplayedLike",
    "RunCounters",
    "SuggestedTopic",
    "TopicScore",
    "build_rows",
    "default_row_order",
    "memory_score",
    "report_order_key",
    "run_counters",
    "summarize",
    "topic_breakdown",
]
