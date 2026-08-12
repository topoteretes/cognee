"""Assign question rows to the owner's topics, or to the sink.

Phase 2 step 7 of the recall-coverage spec. Pure numpy and value objects: no
database access (that is :mod:`cognee.modules.recall_coverage.repository`) and no
LLM calls (topic *labels* are written once, on accept — see
:mod:`cognee.modules.recall_coverage.suggest`).

Four rules, each of which exists because its absence produces a plausible-looking
wrong number:

* **Threshold AND margin.** A question is assigned only when its best topic
  similarity clears ``assignment_threshold`` *and* beats the runner-up by
  ``assignment_margin``. The threshold alone would assign a question sitting
  almost exactly between two topics to whichever one won by 0.001 — and since
  ``topics[].avg_score`` is a mean over assigned rows, that coin flip moves two
  topic scores. Such a question belongs in the sink, which is precisely the
  signal that the taxonomy is missing something.
* **One question, one topic.** No multi-label assignment, no max-over-topics:
  every headline number is a mean over rows, and a row counted under two topics
  would be double counted.
* **The sink is the wire literal ``"other"``**, not a row in
  ``recall_coverage_topics``. Unassigned rows carry ``topic_id = NULL`` in the
  database, so the sink can never be deleted, accepted, or renamed.
* **A fingerprint mismatch fails the run.** Comparing a centroid embedded by one
  model against a question embedded by another yields a confident, meaningless
  similarity. Re-embedding the topics instead would silently move every stored
  centroid and reset the trend that stable topic ids exist to carry, so the run
  stops and asks the operator to decide.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from uuid import UUID

import numpy as np

from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.embedding import EmbeddingFingerprint
from cognee.modules.recall_coverage.exceptions import EmbeddingFingerprintMismatchError
from cognee.modules.recall_coverage.types import SINK_TOPIC_ID, SINK_TOPIC_LABEL
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")


class TopicLike(Protocol):
    """What this module reads off a topic — nothing else.

    Satisfied by :class:`cognee.modules.recall_coverage.repository.TopicRecord`.
    A protocol rather than that class so a pure numpy module does not import the
    repository, and with it the relational engine, to compare five attributes.
    """

    id: UUID
    label: str
    centroid: tuple[float, ...]
    embedding_model: str
    embedding_dimensions: int


@dataclass(frozen=True)
class TopicAssignment:
    """Where one question row landed, and how convincingly.

    ``topic_id is None`` is the sink. The similarities are kept for logging and
    threshold tuning; nothing scored is derived from them.
    """

    topic_id: Optional[UUID]
    topic_label: str
    similarity: float
    # ``None`` when there was no second topic to compare against, in which case
    # the margin rule has nothing to reject and is treated as satisfied.
    runner_up_similarity: Optional[float] = None

    @property
    def is_sink(self) -> bool:
        return self.topic_id is None

    @property
    def wire_topic_id(self) -> str:
        """The id the API reports: a topic UUID as a string, or ``"other"``."""
        return SINK_TOPIC_ID if self.topic_id is None else str(self.topic_id)


@dataclass(frozen=True)
class AssignmentResult:
    """One :class:`TopicAssignment` per question, index-aligned with the input."""

    assignments: list[TopicAssignment]
    # Positions of the questions that fell through to the sink. This is the input
    # to sink clustering, so it is returned rather than recomputed there.
    sink_indices: list[int]
    # Topics that received at least one question, in the order the topics were
    # given. ``topic_count`` on the run row counts these, not the taxonomy: a
    # topic nobody asked about did not participate in this run.
    assigned_topic_ids: tuple[UUID, ...]

    @property
    def sink_question_count(self) -> int:
        return len(self.sink_indices)


def canonical_matrix(questions: Sequence[DedupedQuestion], normalized: np.ndarray) -> np.ndarray:
    """Gather each question's canonical vector out of the ask-level matrix.

    Dedup records a ``canonical_index`` into the matrix the asks were embedded
    into precisely so that every later phase reuses that vector. Re-embedding the
    canonical text here would be both a second bill and, if the engine's config
    moved in between, a comparison across two embedding spaces.
    """
    if not questions:
        return np.zeros((0, 0))

    return np.asarray([normalized[question.canonical_index] for question in questions], dtype=float)


def require_matching_fingerprint(
    topics: Sequence[TopicLike], fingerprint: EmbeddingFingerprint
) -> None:
    """Raise unless every topic centroid was embedded by the live engine.

    The model is always compared. The dimension is compared only when the engine
    reports one — an engine that will not name its width cannot contradict a
    stored width, and inventing a comparison there would fail runs for a missing
    accessor rather than for a real mismatch.
    """
    mismatched = [
        topic
        for topic in topics
        if topic.embedding_model != fingerprint.model
        or (fingerprint.dimensions > 0 and topic.embedding_dimensions != fingerprint.dimensions)
    ]
    if not mismatched:
        return

    raise EmbeddingFingerprintMismatchError(
        message=(
            f"{len(mismatched)} topic centroid(s) were embedded with "
            f"{mismatched[0].embedding_model!r}/{mismatched[0].embedding_dimensions} but the "
            f"live embedding engine is {fingerprint.model!r}/{fingerprint.dimensions}. "
            "Delete the stale topics or restore the previous embedding configuration; "
            "recall coverage will not re-embed them silently."
        )
    )


def stack_centroids(topics: Sequence[TopicLike], width: int) -> np.ndarray:
    """Stack the topic centroids into a ``T x D`` matrix for one matmul.

    ``width`` is the width of the question matrix. A centroid of a different
    length would either crash the matmul or, worse, broadcast; it means the
    stored vector disagrees with its own recorded fingerprint, so it is reported
    as the fingerprint mismatch it is.
    """
    rows: list[list[float]] = []
    for topic in topics:
        centroid = [float(value) for value in topic.centroid]
        if width and len(centroid) != width:
            raise EmbeddingFingerprintMismatchError(
                message=(
                    f"Topic {topic.id} has a {len(centroid)}-dimensional centroid but "
                    f"questions are embedded in {width} dimensions."
                )
            )
        rows.append(centroid)

    if not rows:
        return np.zeros((0, 0))
    return np.asarray(rows, dtype=float)


def assign_topics(
    question_vectors: np.ndarray,
    topics: Sequence[TopicLike],
    *,
    fingerprint: EmbeddingFingerprint,
    assignment_threshold: float,
    assignment_margin: float,
) -> AssignmentResult:
    """Assign each question vector to at most one topic, else to the sink.

    ``question_vectors`` and every ``topic.centroid`` must be L2-normalized (they
    are: see :func:`cognee.modules.recall_coverage.embedding.normalize_rows` and
    the ``centroid`` column comment), so ``questions @ centroids.T`` is already
    the cosine-similarity matrix and the whole assignment is one matmul.

    A question whose embedding failed is an all-zero row, so every similarity is
    0 and it lands in the sink. That is the correct outcome — a question we could
    not embed has not been shown to belong anywhere — and it is visible, because
    the sink share is reported and alerted on.
    """
    question_count = int(question_vectors.shape[0]) if question_vectors.ndim == 2 else 0
    if question_count == 0:
        return AssignmentResult(assignments=[], sink_indices=[], assigned_topic_ids=())

    sink = TopicAssignment(
        topic_id=None, topic_label=SINK_TOPIC_LABEL, similarity=0.0, runner_up_similarity=None
    )

    if not topics:
        # Everything is unassigned before the owner has accepted a single topic.
        # Not an error: the first run's whole job is to propose the taxonomy.
        return AssignmentResult(
            assignments=[sink] * question_count,
            sink_indices=list(range(question_count)),
            assigned_topic_ids=(),
        )

    require_matching_fingerprint(topics, fingerprint)
    centroids = stack_centroids(topics, int(question_vectors.shape[1]))

    similarity = question_vectors @ centroids.T

    assignments: list[TopicAssignment] = []
    sink_indices: list[int] = []
    assigned: list[UUID] = []

    for index in range(question_count):
        row = similarity[index]
        best_position = int(np.argmax(row))
        best = float(row[best_position])

        runner_up: Optional[float] = None
        if len(topics) > 1:
            remaining = np.delete(row, best_position)
            runner_up = float(np.max(remaining))

        clears_threshold = best >= assignment_threshold
        clears_margin = runner_up is None or (best - runner_up) >= assignment_margin

        if clears_threshold and clears_margin:
            topic = topics[best_position]
            assignments.append(
                TopicAssignment(
                    topic_id=topic.id,
                    topic_label=topic.label,
                    similarity=best,
                    runner_up_similarity=runner_up,
                )
            )
            if topic.id not in assigned:
                assigned.append(topic.id)
            continue

        assignments.append(
            TopicAssignment(
                topic_id=None,
                topic_label=SINK_TOPIC_LABEL,
                similarity=best,
                runner_up_similarity=runner_up,
            )
        )
        sink_indices.append(index)

    logger.debug(
        "recall_coverage: assigned %s of %s questions across %s topics",
        question_count - len(sink_indices),
        question_count,
        len(topics),
    )

    return AssignmentResult(
        assignments=assignments,
        sink_indices=sink_indices,
        assigned_topic_ids=tuple(assigned),
    )


__all__ = [
    "AssignmentResult",
    "TopicAssignment",
    "assign_topics",
    "canonical_matrix",
    "require_matching_fingerprint",
    "stack_centroids",
]
