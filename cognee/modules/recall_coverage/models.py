"""SQLAlchemy models for recall coverage — five tables.

House pattern, following ``cognee/modules/search/models/Query.py`` and
``cognee/modules/sync/models/SyncOperation.py``:

* ``Column(UUID, primary_key=True, default=uuid4)``;
* ``DateTime(timezone=True)`` with Python-side defaults on ``created_at`` and
  ``updated_at``;
* status/source are ``Column(String)`` with an app-level ``str, Enum``
  (:mod:`cognee.modules.recall_coverage.types`) — never ``Enum(...)``, because
  adding a value to a native Postgres enum needs raw DDL (see
  ``cognee/alembic/versions/1d0bb7fede17_add_pipeline_run_status.py``);
* ``owner_id`` / ``user_id`` / ``dataset_id`` are bare indexed UUIDs with **no
  foreign keys**, exactly like ``Query.user_id`` and ``Query.dataset_id``.

Naming is fixed: ``owner_id`` is the scope owner of a topic or user-defined
question, ``user_id`` is the human who asked a given question row, and
``agent_label`` is a plain ``String`` — on the run (which agent it covered) and
on each question row (which agent asked *that* row). There is no ``agent_id``
column anywhere.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UUID,
)

from cognee.infrastructure.databases.relational.ModelBase import Base
from cognee.modules.recall_coverage.types import (
    QuestionSource,
    RunStatus,
    SuggestionStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RecallCoverageRun(Base):
    """One coverage run over one ``agent_label``.

    Owns its own id and status instead of borrowing ``PipelineRun``: a run spans
    every dataset in the tenant, while ``PipelineRun.dataset_id`` is scalar, so
    routing this through ``run_pipeline`` would mint N run rows and hold N
    dataset locks against ``add()``/``cognify()`` for the run's whole duration.

    ``summary`` is frozen at completion. Recomputing breakdowns at read time
    would let a deleted topic — or an owner losing dataset access — silently
    reshape historical runs and destroy the trend that stable topic ids exist to
    protect.
    """

    __tablename__ = "recall_coverage_runs"

    id = Column(UUID, primary_key=True, default=uuid4)

    # The tool this run analysed: a label from the configured prefix map, or the
    # literal "api" / "all". A String, not a FK — an agent exists because it
    # asked something, not because it was registered.
    agent_label = Column(String, nullable=False, index=True)

    # The caller who started the run.
    owner_id = Column(UUID, nullable=False, index=True)

    # See RunStatus. Indexed because the in-flight guard for
    # (owner_id, agent_label) filters on it before scheduling a new run.
    status = Column(String, nullable=False, default=RunStatus.PENDING.value, index=True)

    # The CoverageParams snapshot the run executed under, so a historical run
    # stays readable after the deployment's defaults move.
    params = Column(JSON, nullable=True)
    # Frozen memory score, topic breakdown and suggested topics; see the class
    # docstring. Also where fail_run records the error, so a failed run needs no
    # error column of its own.
    summary = Column(JSON, nullable=True)

    finished_at = Column(DateTime(timezone=True), nullable=True)

    # The two counters the report returns. recall_count counts raw `queries` rows
    # in the window (so a truncated window is visible); question_count counts the
    # (user, dataset, canonical text) rows this run actually judged.
    recall_count = Column(Integer, nullable=False, default=0)
    question_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, index=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageQuestion(Base):
    """One judged question row: ``(user_id, dataset_id, canonical text)``.

    The same text from two teammates, or against two datasets, is deliberately
    two rows — there are no per-dataset child cells, no ``best_dataset_id`` and
    no max-over-cells anywhere. The report is one flat table the UI filters.
    """

    __tablename__ = "recall_coverage_questions"

    id = Column(UUID, primary_key=True, default=uuid4)

    run_id = Column(UUID, nullable=False, index=True)

    # The human who asked. Not the caller: a run covers every user in the
    # tenant, and the UI filters.
    user_id = Column(UUID, nullable=False, index=True)

    # NULL when the ask had no single dataset to attribute it to (a search that
    # spanned several, or a user-defined question). Such rows form their own dedup
    # partition and replay with dataset_ids=None.
    dataset_id = Column(UUID, nullable=True, index=True)
    dataset_name = Column(String, nullable=True)

    question = Column(Text, nullable=False)

    # The agent whose session produced this row, resolved from that session's id
    # by the prefix map — not the run's label, because the default run is "all"
    # and this column is what narrows one flat table to one agent. Stored rather
    # than re-derived at read time so editing the prefix map cannot relabel
    # history. NULL for a user-defined question nobody asked.
    #
    # Not indexed: rows are only ever loaded by run_id, and one run's rows are a
    # bounded set the caller filters in memory.
    agent_label = Column(String, nullable=True)

    # See QuestionSource. User-defined rows are pinned above observed rows in the
    # default order.
    source = Column(String, nullable=False, default=QuestionSource.OBSERVED.value)
    curated_question_id = Column(UUID, nullable=True)

    answer = Column(Text, nullable=True)
    # 0..judge_score_max. NULL when the row errored out.
    coverage_score = Column(Integer, nullable=True)
    retrieval_context = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # NULL means the sink: reported as topic_id null with the label
    # "Uncategorized", never a row in recall_coverage_topics.
    topic_id = Column(UUID, nullable=True, index=True)

    first_asked_at = Column(DateTime(timezone=True), nullable=True)
    last_asked_at = Column(DateTime(timezone=True), nullable=True)

    # Distinct asks in this row's cluster, never raw rows, and never counting the
    # user-defined member — a human writing a question is not demand for it.
    #
    # This doubles as the "was it actually asked" test, which is why there is no
    # was_asked column: a user-defined question that merged with real traffic has
    # relevance > 0 and stays in every mean, while one nobody asked has
    # relevance 0. Testing `source` instead would be gameable — naming a question
    # you know scores badly would relabel the matching observed row as
    # user_defined and quietly lift the headline score.
    relevance = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageTopic(Base):
    """An owner-scoped, permanent topic with a stable id.

    Owner-scoped rather than per-agent or per-dataset on purpose: one taxonomy
    across all of an owner's agents is what makes "Codex 4.2 on Billing, Claude
    Code 2.1 on Billing" a sentence at all.

    Lifecycle is create, delete, dismiss only — no rename, no merge, no split,
    and hence no ``merged_into_id``. Delete is soft (``deleted_at``) because a
    historical run's question rows carry this id: hard-deleting would leave those
    rows pointing at nothing and silently reshape a finished report. A deleted
    topic's questions fall back to the sink on the next run and are never deleted.
    """

    __tablename__ = "recall_coverage_topics"

    __table_args__ = (Index("ix_recall_coverage_topics_owner_deleted", "owner_id", "deleted_at"),)

    id = Column(UUID, primary_key=True, default=uuid4)

    owner_id = Column(UUID, nullable=False, index=True)
    # Stays ``label`` here and on suggestions, where it is one field among many;
    # the wire calls it ``topic`` because there the object *is* the topic.
    label = Column(Text, nullable=False)

    # L2-normalized centroid, so a dot product against a normalized question
    # vector is already the cosine similarity.
    centroid = Column(JSON, nullable=False)

    # The embedding fingerprint the centroid was computed under. A run whose
    # live engine disagrees fails rather than silently re-embedding into a
    # different space.
    embedding_model = Column(String, nullable=False)
    embedding_dimensions = Column(Integer, nullable=False)

    deleted_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageTopicSuggestion(Base):
    """A dense sink cluster proposed as a new topic.

    Owner-scoped like the topics it may become; ``agent_label`` and ``run_id``
    are provenance only ("this came out of the Codex run"), never scope. The
    topic id is minted when the owner posts the label, not here — that is what
    makes accepted topic ids stable across runs.
    """

    __tablename__ = "recall_coverage_topic_suggestions"

    id = Column(UUID, primary_key=True, default=uuid4)

    owner_id = Column(UUID, nullable=False, index=True)

    # Provenance: which agent's run surfaced this, and which run.
    agent_label = Column(String, nullable=True)
    run_id = Column(UUID, nullable=True, index=True)

    label = Column(Text, nullable=False)
    centroid = Column(JSON, nullable=False)
    embedding_model = Column(String, nullable=False)
    embedding_dimensions = Column(Integer, nullable=False)

    question_count = Column(Integer, nullable=False, default=0)
    # Mean intra-cluster cosine. Orders the candidates; never scored.
    cohesion = Column(Float, nullable=True)

    # See SuggestionStatus. Indexed because the re-proposal guard scans an
    # owner's dismissed and accepted suggestions on every run.
    status = Column(String, nullable=False, default=SuggestionStatus.PENDING.value, index=True)
    accepted_topic_id = Column(UUID, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageCuratedQuestion(Base):
    """A question a human typed that memory *should* answer.

    Not a separate report: these enter the same window, the same dedup, the same
    replay and the same judge, and appear in the same ``questions[]`` with
    ``source = "user_defined"``.

    One flat list per owner — no agent-scoped or shared variants. Every run for
    that owner replays every one of them, so a question is worth writing down
    once rather than once per tool.
    """

    __tablename__ = "recall_coverage_curated_questions"

    id = Column(UUID, primary_key=True, default=uuid4)

    owner_id = Column(UUID, nullable=False, index=True)

    question = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


__all__ = [
    "RecallCoverageCuratedQuestion",
    "RecallCoverageQuestion",
    "RecallCoverageRun",
    "RecallCoverageTopic",
    "RecallCoverageTopicSuggestion",
]
