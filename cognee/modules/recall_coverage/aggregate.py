"""Turn judged rows into question rows and one frozen summary.

Phase 4 step 13 of the recall-coverage spec. Pure functions over value objects:
nothing here calls an LLM, an embedding engine or the database — persistence is
:mod:`cognee.modules.recall_coverage.repository`, which imports this module.
That direction is why the three phase-3 inputs arrive as small protocols instead
of the concrete ``TopicAssignment`` / ``ReplayedRow`` / ``JudgedRow`` classes:
importing them would drag the LLM, embedding and search stacks into the
persistence layer for the sake of six attributes.

**Every headline number is a mean over rows.** No weighting, no max-over-cells,
no synthetic scores. Five rules decide *which* rows, and each one exists because
its absence produces a number that looks right and is not:

* **Curated rows feed nothing except ``benchmark_score_pct``.** A user who could
  raise ``overall_score`` by adding questions their memory happens to answer
  would be able to game the headline; a user who could lower it by adding
  aspirational ones would stop adding them. Only **shared** curated rows — the
  benchmark set, identical prompts across agents — feed
  ``benchmark_score_pct``, and agent-scoped curated rows feed no aggregate at
  all. Every other breakdown here groups **observed** rows only.
* **One row set, four groupings.** Topics, the sink, datasets and users are four
  ``GROUP BY``s over the same observed rows, so they reconcile exactly: the
  question counts of each grouping sum to the same total, and the count-weighted
  mean of each grouping's averages is the same number (see
  :func:`reconciles`, which the tests assert).
* **``overall_score`` averages topics, not rows** — the one deliberate
  asymmetry. A single chatty dataset would otherwise dominate the headline just
  by asking more, and the point of the taxonomy is that "Billing 4.2, Runbooks
  1.8" is a sentence about coverage rather than about traffic volume. The sink is
  excluded (it is the absence of a topic, not a topic) and so is any topic with
  fewer than ``min_scored_questions_per_topic`` scored rows, which reports
  ``avg_score: null`` rather than letting one question speak for a theme.
* **``null``, never ``0.0``, when nothing qualifies.** ``0.0`` is a real score
  meaning "memory answered nothing"; a run with no qualifying topic has not
  measured that. Conflating the two turns "we do not know" into the worst
  possible verdict, and a trend line built on it is fiction.
* **A NULL score is not a zero.** A row whose replay or judge failed carries no
  score, is absent from every mean, and its ``impact`` is NULL too — "we could
  not ask" is not evidence about memory. This is why ``question_count`` and
  ``scored_question_count`` are both reported on every cell: the gap between
  them is how much of the group the averages actually describe.

Nothing is rounded. Presentation precision is the UI's business, and a rounding
constant here would be the one magic number in the module.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Mapping, Optional, Protocol, Sequence
from uuid import UUID

from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_ID,
    SINK_TOPIC_LABEL,
    CoverageParams,
    QuestionSource,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")

# Alert codes. Strings rather than an enum column: they are wire values inside the
# frozen ``summary`` JSON, so a historical run must keep resolving them even after
# the code that produced them is gone.
SINK_SHARE_ALERT = "sink_share_above_threshold"
SINK_CLUSTER_ALERT = "large_sink_cluster"

# Percentage scale for ``benchmark_score_pct``. Not a tunable: a percentage is
# defined as parts per hundred, and the tunable in that formula is
# ``judge_score_max``, which is the denominator.
PERCENT = 100.0


class AssignmentLike(Protocol):
    """What a topic assignment contributes to a row: where it landed.

    Satisfied by :class:`cognee.modules.recall_coverage.assign.TopicAssignment`.
    ``topic_id is None`` is the sink, whose wire id is the literal ``"other"``.
    """

    topic_id: Optional[UUID]
    topic_label: str


class ReplayedLike(Protocol):
    """What a replayed row contributes: the context, and the dataset it named.

    Satisfied by :class:`cognee.modules.recall_coverage.replay.ReplayedRow`.
    """

    retrieval_context: Optional[str]
    dataset_name: Optional[str]


class JudgedLike(Protocol):
    """What the judge contributes.

    Satisfied by :class:`cognee.modules.recall_coverage.judge.JudgedRow`. Note
    that ``error`` being set does **not** imply ``judge_score`` is ``None``: a
    coverage score that survived a failed answer generation is kept, because it
    is the number every mean here is over.
    """

    judge_score: Optional[int]
    judge_answered: Optional[bool]
    answer: Optional[str]
    error: Optional[str]


def row_impact(
    occurrence_count: int, judge_score: Optional[int], judge_score_max: int
) -> Optional[float]:
    """``occurrence_count * (judge_score_max - judge_score)``, NULL when unscored.

    How much demand went unmet: asked often and answered badly ranks above asked
    once and answered badly, and a perfectly answered question has an impact of
    zero however often it was asked. A purely curated row has
    ``occurrence_count = 0`` and therefore an impact of 0 no matter how it scored
    — nobody has asked it yet, so nothing is currently being lost.

    NULL rather than 0 when the score is NULL: a row we could not judge is not a
    row with nothing at stake, and sorting it alongside the perfect answers would
    hide exactly the failures worth looking at.
    """
    if judge_score is None:
        return None
    return float(occurrence_count * (judge_score_max - judge_score))


@dataclass(frozen=True)
class CoverageRow:
    """One question row, judged, attributed and ready to persist.

    Mirrors ``recall_coverage_questions`` one-for-one with two exceptions, both
    derived rather than stored: ``topic_label`` (the label lives on the topic
    row, and ``"Other"`` for the sink is not a row at all) and
    ``is_shared_curated`` (reconstructible by joining
    ``recall_coverage_curated_questions.scope``, and carried here only so the
    summary can be computed in one pass).
    """

    question_text: str
    user_id: UUID
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    question_group_id: Optional[UUID]
    source: str
    was_asked: bool
    curated_question_id: Optional[UUID]
    topic_id: Optional[UUID]
    topic_label: str
    answer: Optional[str]
    judge_score: Optional[int]
    judge_answered: Optional[bool]
    retrieval_context: Optional[str]
    error: Optional[str]
    first_asked_at: Optional[datetime]
    last_asked_at: Optional[datetime]
    occurrence_count: int
    impact: Optional[float]
    is_shared_curated: bool = False

    @property
    def is_observed(self) -> bool:
        return self.source == QuestionSource.OBSERVED.value

    @property
    def is_sink(self) -> bool:
        """True when the row matched no topic. Tested on the id, never on ``"other"``."""
        return self.topic_id is None

    @property
    def is_scored(self) -> bool:
        return self.judge_score is not None

    @property
    def wire_topic_id(self) -> str:
        return SINK_TOPIC_ID if self.topic_id is None else str(self.topic_id)


@dataclass(frozen=True)
class CoverageAlert:
    """One thing worth telling the reader about, as a stable code plus prose."""

    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class TopicScore:
    """One topic's cell. ``topic_id is None`` is the sink."""

    topic_id: Optional[UUID]
    label: str
    question_count: int
    scored_question_count: int
    avg_score: Optional[float]

    @property
    def wire_topic_id(self) -> str:
        return SINK_TOPIC_ID if self.topic_id is None else str(self.topic_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.wire_topic_id,
            "label": self.label,
            "question_count": self.question_count,
            "scored_question_count": self.scored_question_count,
            "avg_score": self.avg_score,
        }


@dataclass(frozen=True)
class DatasetScore:
    """One dataset's cell. ``dataset_id is None`` collects the unscoped rows."""

    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    question_count: int
    scored_question_count: int
    avg_score: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": None if self.dataset_id is None else str(self.dataset_id),
            "dataset_name": self.dataset_name,
            "question_count": self.question_count,
            "scored_question_count": self.scored_question_count,
            "avg_score": self.avg_score,
        }


@dataclass(frozen=True)
class UserScore:
    """One asker's cell. A run is tenant-wide, so this is how the UI filters."""

    user_id: UUID
    question_count: int
    scored_question_count: int
    avg_score: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": str(self.user_id),
            "question_count": self.question_count,
            "scored_question_count": self.scored_question_count,
            "avg_score": self.avg_score,
        }


@dataclass(frozen=True)
class SinkReport:
    """The residual: questions the taxonomy could not place.

    ``share`` is over **observed** rows on both sides of the division. The spec
    writes the denominator as the run's ``question_row_count``, which also counts
    curated rows; using it would make the share — and the alert built on it —
    move when a human adds curated questions, since curated rows are assigned to
    topics like anything else. A user adding fifty off-taxonomy benchmark
    questions would trip the "your taxonomy is missing something" alarm about
    traffic that never happened, and a user adding on-taxonomy ones would silence
    it. The run row still reports ``question_row_count`` for anyone who wants the
    raw ratio.
    """

    question_count: int
    scored_question_count: int
    share: Optional[float]
    avg_score: Optional[float]
    alerts: list[CoverageAlert]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": SINK_TOPIC_ID,
            "label": SINK_TOPIC_LABEL,
            "question_count": self.question_count,
            "scored_question_count": self.scored_question_count,
            "share": self.share,
            "avg_score": self.avg_score,
            "alerts": [alert.to_dict() for alert in self.alerts],
        }


@dataclass(frozen=True)
class CoverageSummary:
    """The frozen report. Serialized into ``recall_coverage_runs.summary``.

    Frozen rather than recomputed at read time because the breakdowns name things
    that change: delete a topic and its historical rows would silently move into
    the sink; lose access to a dataset and it would vanish from a run that
    genuinely covered it. Either would reshape a past run and destroy the trend
    that stable topic ids exist to carry.
    """

    overall_score: Optional[float]
    benchmark_score_pct: Optional[float]
    unscoped_ask_share: Optional[float]
    observed_question_count: int
    scored_question_count: int
    curated_question_count: int
    topics: list[TopicScore]
    datasets: list[DatasetScore]
    users: list[UserScore]
    sink: SinkReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "benchmark_score_pct": self.benchmark_score_pct,
            "unscoped_ask_share": self.unscoped_ask_share,
            "observed_question_count": self.observed_question_count,
            "scored_question_count": self.scored_question_count,
            "curated_question_count": self.curated_question_count,
            "topics": [topic.to_dict() for topic in self.topics],
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "users": [user.to_dict() for user in self.users],
            "sink": self.sink.to_dict(),
        }


@dataclass(frozen=True)
class RunCounters:
    """The counters written onto the run row when it completes."""

    recall_row_count: int
    distinct_ask_count: int
    collapsed_retry_count: int
    question_row_count: int
    curated_question_count: int
    topic_count: int
    dataset_count: int
    user_count: int
    taxonomy_version: int


def _mean(values: Sequence[float]) -> Optional[float]:
    """The mean, or ``None`` for an empty sequence — never ``0.0``."""
    return (sum(values) / len(values)) if values else None


def _scores(rows: Sequence[CoverageRow]) -> list[float]:
    return [float(row.judge_score) for row in rows if row.judge_score is not None]


def build_rows(
    questions: Sequence[DedupedQuestion],
    assignments: Sequence[AssignmentLike],
    replayed: Sequence[ReplayedLike],
    judged: Sequence[JudgedLike],
    *,
    judge_score_max: int,
    shared_curated_ids: Collection[UUID] = (),
    dataset_names: Optional[Mapping[UUID, str]] = None,
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

    shared_ids = set(shared_curated_ids)
    names = dict(dataset_names or {})

    rows: list[CoverageRow] = []
    for question, assignment, replay, verdict in zip(questions, assignments, replayed, judged):
        dataset_name = names.get(question.dataset_id) if question.dataset_id else None

        rows.append(
            CoverageRow(
                question_text=question.text,
                user_id=question.user_id,
                dataset_id=question.dataset_id,
                dataset_name=dataset_name or replay.dataset_name,
                question_group_id=question.question_group_id,
                source=question.source,
                was_asked=question.was_asked,
                curated_question_id=question.curated_question_id,
                topic_id=assignment.topic_id,
                topic_label=assignment.topic_label,
                answer=verdict.answer,
                judge_score=verdict.judge_score,
                judge_answered=verdict.judge_answered,
                retrieval_context=replay.retrieval_context,
                error=verdict.error,
                first_asked_at=question.first_asked_at,
                last_asked_at=question.last_asked_at,
                occurrence_count=question.occurrence_count,
                impact=row_impact(question.occurrence_count, verdict.judge_score, judge_score_max),
                is_shared_curated=(
                    question.source == QuestionSource.CURATED.value
                    and question.curated_question_id in shared_ids
                ),
            )
        )

    return rows


def report_order_key(row: Any) -> tuple:
    """Sort key for the default order of ``questions[]``: curated first, then by impact.

    Curated rows are **pinned** above observed rows rather than given a synthetic
    weight that would let them compete on the same scale: "somebody cares about
    this" and "this was asked forty times" are two incomparable kinds of
    importance, and blending them into one number invents an exchange rate
    between them. Within each half, highest unmet demand first, with unscored
    rows last — they carry no impact, so ranking them among the answered ones
    would bury the actual failures.

    Duck-typed on ``is_observed``, ``impact``, ``occurrence_count`` and
    ``question_text`` so the write side (:class:`CoverageRow`) and the read side
    (``repository.QuestionRecord``) cannot drift into two different default
    orders.
    """
    return (
        row.is_observed,
        row.impact is None,
        -(row.impact or 0.0),
        -row.occurrence_count,
        row.question_text,
    )


def default_row_order(rows: Sequence[CoverageRow]) -> list[CoverageRow]:
    """Rows in the default report order. See :func:`report_order_key`."""
    return sorted(rows, key=report_order_key)


def topic_breakdown(
    rows: Sequence[CoverageRow], *, min_scored_questions_per_topic: int
) -> list[TopicScore]:
    """Per-topic means over observed rows, sink excluded (it is :func:`sink_report`).

    ``avg_score`` is ``null`` — and the topic is therefore out of
    ``overall_score`` — below ``min_scored_questions_per_topic`` scored rows. The
    rows still count in ``question_count``: the topic was asked about, we just
    will not put a number on it yet.
    """
    grouped: dict[UUID, list[CoverageRow]] = {}
    labels: dict[UUID, str] = {}

    for row in rows:
        if not row.is_observed or row.topic_id is None:
            continue
        grouped.setdefault(row.topic_id, []).append(row)
        labels.setdefault(row.topic_id, row.topic_label)

    topics: list[TopicScore] = []
    for topic_id, members in grouped.items():
        scores = _scores(members)
        qualifies = len(scores) >= min_scored_questions_per_topic
        topics.append(
            TopicScore(
                topic_id=topic_id,
                label=labels[topic_id],
                question_count=len(members),
                scored_question_count=len(scores),
                avg_score=_mean(scores) if qualifies else None,
            )
        )

    return sorted(topics, key=lambda topic: (-topic.question_count, topic.label))


def dataset_breakdown(rows: Sequence[CoverageRow]) -> list[DatasetScore]:
    """Per-dataset means over the same observed rows the topics were built from.

    Rows with no dataset attribution — a search that spanned several, or a curated
    question that merged with nothing — group under ``dataset_id: null`` rather
    than being dropped, so the counts still reconcile and the unscoped share is
    visible instead of silently missing.
    """
    grouped: dict[Optional[UUID], list[CoverageRow]] = {}
    names: dict[Optional[UUID], Optional[str]] = {}

    for row in rows:
        if not row.is_observed:
            continue
        grouped.setdefault(row.dataset_id, []).append(row)
        if names.get(row.dataset_id) is None:
            names[row.dataset_id] = row.dataset_name

    datasets: list[DatasetScore] = []
    for dataset_id, members in grouped.items():
        scores = _scores(members)
        datasets.append(
            DatasetScore(
                dataset_id=dataset_id,
                dataset_name=names.get(dataset_id),
                question_count=len(members),
                scored_question_count=len(scores),
                avg_score=_mean(scores),
            )
        )

    return sorted(
        datasets,
        key=lambda dataset: (
            -dataset.question_count,
            dataset.dataset_name or "",
            str(dataset.dataset_id),
        ),
    )


def user_breakdown(rows: Sequence[CoverageRow]) -> list[UserScore]:
    """Per-asker means over the same observed rows. A run covers every user."""
    grouped: dict[UUID, list[CoverageRow]] = {}
    for row in rows:
        if not row.is_observed:
            continue
        grouped.setdefault(row.user_id, []).append(row)

    users: list[UserScore] = []
    for user_id, members in grouped.items():
        scores = _scores(members)
        users.append(
            UserScore(
                user_id=user_id,
                question_count=len(members),
                scored_question_count=len(scores),
                avg_score=_mean(scores),
            )
        )

    return sorted(users, key=lambda user: (-user.question_count, str(user.user_id)))


def sink_report(
    rows: Sequence[CoverageRow],
    *,
    sink_cluster_sizes: Sequence[int] = (),
    sink_share_alert: float,
    sink_cluster_alert_size: int,
) -> SinkReport:
    """The sink's cell plus the two alerts, over observed rows.

    Both alerts say the same thing from different angles — the taxonomy is
    missing something — and both are reported rather than one being derived from
    the other: a diffuse sink can be large without containing any single dense
    theme, and a small sink can contain one very dense one.
    """
    observed = [row for row in rows if row.is_observed]
    sink_rows = [row for row in observed if row.is_sink]
    scores = _scores(sink_rows)

    share = (len(sink_rows) / len(observed)) if observed else None

    alerts: list[CoverageAlert] = []
    if share is not None and share > sink_share_alert:
        alerts.append(
            CoverageAlert(
                code=SINK_SHARE_ALERT,
                # The counts, not the raw share: an unrounded float reads as
                # noise, and a rounding constant here would be the one magic
                # number in the module. ``sink.share`` carries the exact value.
                message=(
                    f"{len(sink_rows)} of {len(observed)} observed questions matched no topic, "
                    f"above the alert share of {sink_share_alert}."
                ),
            )
        )

    largest = max(sink_cluster_sizes) if sink_cluster_sizes else 0
    if largest >= sink_cluster_alert_size and sink_cluster_alert_size > 0:
        alerts.append(
            CoverageAlert(
                code=SINK_CLUSTER_ALERT,
                message=(
                    f"An unmatched cluster of {largest} questions is at or above the "
                    f"alert size of {sink_cluster_alert_size}."
                ),
            )
        )

    return SinkReport(
        question_count=len(sink_rows),
        scored_question_count=len(scores),
        share=share,
        avg_score=_mean(scores),
        alerts=alerts,
    )


def overall_score(topics: Sequence[TopicScore]) -> Optional[float]:
    """Unweighted mean of the qualifying topic averages. ``None`` when none qualify.

    Unweighted so a dataset that asks ten times as much does not become the
    headline; ``None`` rather than ``0.0`` because a run with no qualifying topic
    has measured nothing, while ``0.0`` is the real and much worse claim that
    memory answered nothing. ``topic_breakdown`` has already excluded the sink and
    nulled the under-populated topics, so this is just the mean of what is left.
    """
    return _mean([topic.avg_score for topic in topics if topic.avg_score is not None])


def benchmark_score_pct(rows: Sequence[CoverageRow], *, judge_score_max: int) -> Optional[float]:
    """Mean score over **shared** curated rows, as a percentage of ``judge_score_max``.

    The benchmark set is the only reason two agents' numbers are comparable at
    all: identical prompts, so the difference is the memory rather than the
    questions. A question replicated into three dataset partitions counts three
    times — accepted, because each partition is a separate memory that either
    answered it or did not.

    Agent-scoped curated rows are excluded: they are one person's list for one
    tool, so including them would make the "comparable across agents" number not
    comparable across agents.
    """
    if judge_score_max <= 0:
        return None

    scores = _scores([row for row in rows if row.is_shared_curated])
    mean = _mean(scores)
    return None if mean is None else (mean / judge_score_max) * PERCENT


def unscoped_ask_share(rows: Sequence[CoverageRow], *, distinct_ask_count: int) -> Optional[float]:
    """Share of the window's asks that ended up with no dataset attribution.

    A high value means most recalls fanned across several datasets (or ran with
    none resolvable), which is worth knowing precisely because those rows are the
    ones whose per-dataset coverage the report cannot break down.

    The numerator counts **observed** rows only. The denominator is
    ``distinct_ask_count``, the ask count *before* the ``max_questions``
    truncation, so this is a statement about the window rather than about the
    sample — which also means it can be small even when a large share of the
    judged rows are unscoped. ``None`` when the window held no asks at all,
    rather than a zero share that would read as "everything is attributed".
    """
    if distinct_ask_count <= 0:
        return None

    unscoped = [row for row in rows if row.is_observed and row.dataset_id is None]
    return len(unscoped) / distinct_ask_count


def summarize(
    rows: Sequence[CoverageRow],
    *,
    params: CoverageParams,
    distinct_ask_count: int = 0,
    sink_cluster_sizes: Sequence[int] = (),
) -> CoverageSummary:
    """Build the frozen summary from the run's rows.

    Deliberately a pure function of ``rows`` plus the run's parameters: the same
    rows always produce the same summary, which is what makes freezing it into
    ``runs.summary`` equivalent to the report a reader would have got at the time
    — and what makes the reconciliation between the breakdowns testable without a
    database.
    """
    topics = topic_breakdown(
        rows, min_scored_questions_per_topic=params.min_scored_questions_per_topic
    )
    observed = [row for row in rows if row.is_observed]

    summary = CoverageSummary(
        overall_score=overall_score(topics),
        benchmark_score_pct=benchmark_score_pct(rows, judge_score_max=params.judge_score_max),
        unscoped_ask_share=unscoped_ask_share(rows, distinct_ask_count=distinct_ask_count),
        observed_question_count=len(observed),
        scored_question_count=len(_scores(observed)),
        curated_question_count=len(rows) - len(observed),
        topics=topics,
        datasets=dataset_breakdown(rows),
        users=user_breakdown(rows),
        sink=sink_report(
            rows,
            sink_cluster_sizes=sink_cluster_sizes,
            sink_share_alert=params.sink_share_alert,
            sink_cluster_alert_size=params.sink_cluster_alert_size,
        ),
    )

    logger.debug(
        "recall_coverage: summarized %s rows into %s topics, overall_score=%s",
        len(rows),
        len(topics),
        summary.overall_score,
    )
    return summary


def reconciles(summary: CoverageSummary) -> bool:
    """True when the three breakdowns describe exactly the same rows.

    Topics-plus-sink, datasets and users are three ``GROUP BY``s over one row set,
    so their question counts must sum to the same total and their scored counts
    must too. Shipped as a function rather than left implicit in the tests because
    it is the property that makes the report internally consistent, and a future
    change that quietly filters one breakdown differently should fail loudly here
    rather than produce a report whose halves disagree.
    """
    totals = [
        sum(topic.question_count for topic in summary.topics) + summary.sink.question_count,
        sum(dataset.question_count for dataset in summary.datasets),
        sum(user.question_count for user in summary.users),
        summary.observed_question_count,
    ]
    scored = [
        sum(topic.scored_question_count for topic in summary.topics)
        + summary.sink.scored_question_count,
        sum(dataset.scored_question_count for dataset in summary.datasets),
        sum(user.scored_question_count for user in summary.users),
        summary.scored_question_count,
    ]
    return len(set(totals)) == 1 and len(set(scored)) == 1


def run_counters(
    rows: Sequence[CoverageRow],
    *,
    recall_row_count: int,
    distinct_ask_count: int,
    collapsed_retry_count: int,
    taxonomy_version: int,
) -> RunCounters:
    """The counters the run row reports, derived from the rows plus phase-1 counts.

    ``topic_count`` counts topics that **received a row**, not the size of the
    owner's taxonomy: a topic nobody asked about did not participate in this run,
    and counting it would make the number move whenever somebody accepts a
    suggestion rather than when coverage changes. ``dataset_count`` likewise
    ignores the unscoped partition, which is not a dataset.
    """
    return RunCounters(
        recall_row_count=recall_row_count,
        distinct_ask_count=distinct_ask_count,
        collapsed_retry_count=collapsed_retry_count,
        question_row_count=len(rows),
        curated_question_count=len([row for row in rows if not row.is_observed]),
        topic_count=len({row.topic_id for row in rows if row.topic_id is not None}),
        dataset_count=len({row.dataset_id for row in rows if row.dataset_id is not None}),
        user_count=len({row.user_id for row in rows}),
        taxonomy_version=taxonomy_version,
    )


__all__ = [
    "PERCENT",
    "SINK_CLUSTER_ALERT",
    "SINK_SHARE_ALERT",
    "AssignmentLike",
    "CoverageAlert",
    "CoverageRow",
    "CoverageSummary",
    "DatasetScore",
    "JudgedLike",
    "ReplayedLike",
    "RunCounters",
    "SinkReport",
    "TopicScore",
    "UserScore",
    "benchmark_score_pct",
    "build_rows",
    "dataset_breakdown",
    "default_row_order",
    "overall_score",
    "reconciles",
    "report_order_key",
    "row_impact",
    "run_counters",
    "sink_report",
    "summarize",
    "topic_breakdown",
    "unscoped_ask_share",
    "user_breakdown",
]
