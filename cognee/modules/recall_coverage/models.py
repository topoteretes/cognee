"""SQLAlchemy models for recall coverage — five tables.

House pattern, following ``cognee/modules/search/models/Query.py`` and
``cognee/modules/sync/models/SyncOperation.py``:

* ``Column(UUID, primary_key=True, default=uuid4)``;
* ``DateTime(timezone=True)`` with Python-side defaults on ``created_at`` and
  ``updated_at``;
* status/scope/source are ``Column(String)`` with an app-level ``str, Enum``
  (:mod:`cognee.modules.recall_coverage.types`) — never ``Enum(...)``, because
  adding a value to a native Postgres enum needs raw DDL (see
  ``cognee/alembic/versions/1d0bb7fede17_add_pipeline_run_status.py``);
* ``owner_id`` / ``user_id`` / ``dataset_id`` are bare indexed UUIDs with **no
  foreign keys**, exactly like ``Query.user_id`` and ``Query.dataset_id``.

Naming is fixed: ``owner_id`` is the scope owner of a topic or curated question,
``user_id`` is the human who asked a given question row, and ``agent_label`` is
a plain ``String`` on the run. There is no ``agent_id`` column anywhere.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
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
    CuratedScope,
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
    # Frozen topic/dataset/user breakdowns; see the class docstring.
    summary = Column(JSON, nullable=True)

    finished_at = Column(DateTime(timezone=True), nullable=True)

    # Counters, all reported on the run DTO. recall_row_count counts raw
    # `queries` rows in the window; distinct_ask_count counts asks after the
    # retry cooldown collapsed them; question_row_count counts the
    # (user, dataset, canonical text) rows this run actually judged.
    recall_row_count = Column(Integer, nullable=False, default=0)
    distinct_ask_count = Column(Integer, nullable=False, default=0)
    collapsed_retry_count = Column(Integer, nullable=False, default=0)
    question_row_count = Column(Integer, nullable=False, default=0)
    curated_question_count = Column(Integer, nullable=False, default=0)
    topic_count = Column(Integer, nullable=False, default=0)
    dataset_count = Column(Integer, nullable=False, default=0)
    user_count = Column(Integer, nullable=False, default=0)

    # The owner's taxonomy version this run was scored against.
    taxonomy_version = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now, index=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageQuestion(Base):
    """One judged question row: ``(user_id, dataset_id, canonical text)``.

    The same text from two teammates, or against two datasets, is deliberately
    two rows — there are no per-dataset child cells, no ``best_dataset_id`` and
    no max-over-cells anywhere. Grouping is the UI's job, which is what
    ``question_group_id`` is for: it lets the UI collapse two partitions' rows on
    an id instead of exact-string matching, which would miss them whenever the
    partitions settled on different canonical text.
    """

    __tablename__ = "recall_coverage_questions"

    id = Column(UUID, primary_key=True, default=uuid4)

    run_id = Column(UUID, nullable=False, index=True)

    # Shared across partitions whose canonical texts matched. Every row gets
    # one (its own, when nothing matched); nullable only because it is stamped
    # in a second pass after the per-partition dedup.
    question_group_id = Column(UUID, nullable=True, index=True)

    # The human who asked. Not the caller: a run covers every user in the
    # tenant, and the UI filters.
    user_id = Column(UUID, nullable=False, index=True)

    # NULL when the ask had no single dataset to attribute it to (a search that
    # spanned several, or a curated question). Such rows form their own dedup
    # partition and replay with dataset_ids=None.
    dataset_id = Column(UUID, nullable=True, index=True)
    dataset_name = Column(String, nullable=True)

    question_text = Column(Text, nullable=False)

    # See QuestionSource. Curated rows are pinned above observed rows in the
    # default order and never feed overall_score.
    source = Column(String, nullable=False, default=QuestionSource.OBSERVED.value)
    was_asked = Column(Boolean, nullable=False, default=True)
    curated_question_id = Column(UUID, nullable=True)

    answer = Column(Text, nullable=True)
    # 0..judge_score_max. NULL when the row errored out.
    judge_score = Column(Integer, nullable=True)
    judge_answered = Column(Boolean, nullable=True)
    retrieval_context = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # NULL means the sink, which is the wire literal "other" and not a row in
    # recall_coverage_topics.
    topic_id = Column(UUID, nullable=True, index=True)

    first_asked_at = Column(DateTime(timezone=True), nullable=True)
    last_asked_at = Column(DateTime(timezone=True), nullable=True)

    # Distinct asks in this row's cluster, never raw rows.
    occurrence_count = Column(Integer, nullable=False, default=0)
    # occurrence_count * (judge_score_max - judge_score); NULL when the score is.
    impact = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageTopic(Base):
    """An owner-scoped, permanent topic with a stable id.

    Owner-scoped rather than per-agent or per-dataset on purpose: one taxonomy
    across all of an owner's agents is what makes "Codex 4.2 on Billing, Claude
    Code 2.1 on Billing" a sentence at all.

    Lifecycle is accept, delete, dismiss only — no rename, no merge, no split,
    and hence no ``merged_into_id``. Delete is soft (``deleted_at``) so
    ``taxonomy_version`` stays monotone and historical runs keep resolving their
    topic ids; a deleted topic's questions fall back to the sink on the next run
    and are never deleted.
    """

    __tablename__ = "recall_coverage_topics"

    __table_args__ = (Index("ix_recall_coverage_topics_owner_deleted", "owner_id", "deleted_at"),)

    id = Column(UUID, primary_key=True, default=uuid4)

    owner_id = Column(UUID, nullable=False, index=True)
    label = Column(Text, nullable=False)

    # L2-normalized centroid, so a dot product against a normalized question
    # vector is already the cosine similarity.
    centroid = Column(JSON, nullable=False)

    # The embedding fingerprint the centroid was computed under. A run whose
    # live engine disagrees fails rather than silently re-embedding into a
    # different space.
    embedding_model = Column(String, nullable=False)
    embedding_dimensions = Column(Integer, nullable=False)

    seed_question_count = Column(Integer, nullable=False, default=0)
    taxonomy_version = Column(Integer, nullable=False, default=0)

    deleted_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )


class RecallCoverageTopicSuggestion(Base):
    """A dense sink cluster proposed as a new topic.

    Owner-scoped like the topics it may become; ``agent_label`` and ``run_id``
    are provenance only ("this came out of the Codex run"), never scope. The
    topic id is minted on accept, not here — that is what makes accepted topic
    ids stable across runs.
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

    Not a separate report: curated questions enter the same window, the same
    dedup, the same replay and the same judge, and appear in the same
    ``questions[]`` with ``source = "curated"``.

    ``scope`` is ``agent`` (one ``agent_label``, which is then required) or
    ``shared`` — the benchmark set, whose owner is the caller's tenant when it
    has one and the caller otherwise.
    """

    __tablename__ = "recall_coverage_curated_questions"

    __table_args__ = (
        Index("ix_recall_coverage_curated_questions_owner_scope", "owner_id", "scope"),
    )

    id = Column(UUID, primary_key=True, default=uuid4)

    owner_id = Column(UUID, nullable=False, index=True)

    # See CuratedScope.
    scope = Column(String, nullable=False, default=CuratedScope.AGENT.value)
    # Required when scope is "agent", NULL when it is "shared". Enforced at the
    # app layer: SQLite cannot add a CHECK constraint after the fact.
    agent_label = Column(String, nullable=True)

    question_text = Column(Text, nullable=False)

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
