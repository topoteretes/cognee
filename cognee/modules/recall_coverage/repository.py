"""Persistence for recall coverage: curated questions, topics and suggestions.

Spec sections 2 (phase 2) and 4. A curated question is one a human typed that
memory *should* answer, whether or not any agent asked it. It is deliberately
**not** a separate report: curated questions enter the same window, the same
dedup, the same replay and the same judge as observed traffic, and come back in
the same ``questions[]`` with ``source = "curated"``.

Topics and topic suggestions are **owner-scoped**, never per-agent and never
per-dataset: one taxonomy across all of an owner's agents is what makes "Codex
4.2 on Billing, Claude Code 2.1 on Billing" a sentence at all. ``agent_label`` on
a suggestion is provenance ("this came out of the Codex run"), never scope.

Three things in here are easy to get wrong and are load-bearing:

* **Owner scope.** ``shared`` rows — the benchmark set, identical prompts across
  agents, which is the only reason ``benchmark_score_pct`` compares agents at
  all — are owned by ``user.tenant_id`` when the caller has one, and by
  ``user.id`` otherwise. ``agent`` rows are always owned by ``user.id``: they are
  one person's questions for one tool, not a tenant-wide contract. See
  :func:`resolve_curated_owner`.
* **The duplicate guard is casefold-exact, and it runs in Python.** Same reason
  the dedup collapse key does: SQL ``lower()`` is not ``str.casefold()`` and no
  portable SQL expression collapses interior whitespace. See
  :func:`_duplicate_of`.
* **Rows leave the session as value objects.** ``RecallCoverageCuratedQuestion``
  instances expire on commit, so touching an attribute after the session closes
  raises ``DetachedInstanceError``; the pipeline holds these rows across embed,
  replay and judge. Same choice, same reason as
  ``cognee.modules.search.operations.get_queries.QueryWindowRow``.

Id-keyed lookups filter on **both** id and owner scope and raise 404 on a
mismatch — never 403, which would confirm that someone else's row with that id
exists.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import Ask, collapse_text_key
from cognee.modules.recall_coverage.exceptions import (
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
    EmptyCuratedQuestionError,
    InvalidCuratedQuestionScopeError,
)
from cognee.modules.recall_coverage.models import (
    RecallCoverageCuratedQuestion,
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.types import (
    AgentScope,
    CuratedScope,
    QuestionSource,
    SuggestionStatus,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")


@dataclass(frozen=True)
class CuratedQuestion:
    """One curated question, detached from the session that read it.

    A value object rather than the ORM row: the pipeline carries these across
    embedding, replay and judging, long after the reading session is gone, and a
    committed ORM instance raises on the first attribute access once it is
    detached.
    """

    id: UUID
    owner_id: UUID
    scope: str
    agent_label: Optional[str]
    question_text: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_shared(self) -> bool:
        """True for benchmark-set rows: the only curated rows any aggregate reads."""
        return self.scope == CuratedScope.SHARED.value


def _to_value_object(row: RecallCoverageCuratedQuestion) -> CuratedQuestion:
    return CuratedQuestion(
        id=row.id,
        owner_id=row.owner_id,
        scope=row.scope,
        agent_label=row.agent_label,
        question_text=row.question_text,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def resolve_curated_owner(user: Any, scope: str) -> UUID:
    """The ``owner_id`` a curated question in ``scope`` belongs to.

    ``shared`` resolves to the caller's tenant when it has one, else to the
    caller — the benchmark set is a property of the organisation, so two
    teammates comparing Codex against Claude Code must be reading the same
    prompts or ``benchmark_score_pct`` compares nothing.

    ``agent`` stays with the caller. ``create_user`` never populates
    ``User.tenant_id``, so in practice both collapse to ``user.id`` today; the
    distinction only starts to matter in a deployment that does set it, and
    getting it backwards later would silently re-home existing rows.
    """
    if scope == CuratedScope.SHARED.value:
        return getattr(user, "tenant_id", None) or user.id
    return user.id


def curated_owner_ids(user: Any) -> tuple[UUID, ...]:
    """Every owner scope this caller may read or delete curated questions in.

    Their own, plus their tenant's when they have one. Id-keyed routes filter on
    this rather than on a single owner, so a teammate can delete a shared
    benchmark question while nobody can touch another tenant's row.
    """
    owners: list[UUID] = [user.id]
    tenant = getattr(user, "tenant_id", None)
    if tenant and tenant != user.id:
        owners.append(tenant)
    return tuple(owners)


def normalize_curated_scope(
    scope: Optional[str],
    agent_label: Optional[str] = None,
    *,
    user: Any = None,
    config: Optional[RecallCoverageConfig] = None,
) -> tuple[str, Optional[str]]:
    """Validate ``(scope, agent_label)`` and return the canonical pair.

    ``agent`` requires a label and ``shared`` forbids one, because a shared row
    carrying a label would be a benchmark question that silently only applies to
    one tool. The label itself is validated by ``resolve_agent_scope``, the one
    authority on labels, so a typo 404s here exactly as it does on a run.
    """
    normalized = (scope or CuratedScope.AGENT.value).strip().lower()

    if normalized not in (CuratedScope.AGENT.value, CuratedScope.SHARED.value):
        raise InvalidCuratedQuestionScopeError(
            message=(
                f"Unknown curated question scope: {normalized!r}. Expected "
                f"'{CuratedScope.AGENT.value}' or '{CuratedScope.SHARED.value}'."
            )
        )

    label = (agent_label or "").strip()

    if normalized == CuratedScope.SHARED.value:
        if label:
            raise InvalidCuratedQuestionScopeError(
                message="A shared curated question must not carry an agent_label."
            )
        return normalized, None

    if not label:
        raise InvalidCuratedQuestionScopeError(
            message="An agent-scoped curated question requires an agent_label."
        )

    return normalized, resolve_agent_scope(label, user=user, config=config).label


def _scope_filter(scope: str, agent_label: Optional[str]) -> list[Any]:
    """WHERE terms selecting exactly one ``(scope, agent_label)`` bucket."""
    if scope == CuratedScope.SHARED.value:
        return [
            RecallCoverageCuratedQuestion.scope == CuratedScope.SHARED.value,
            RecallCoverageCuratedQuestion.agent_label.is_(None),
        ]
    return [
        RecallCoverageCuratedQuestion.scope == CuratedScope.AGENT.value,
        RecallCoverageCuratedQuestion.agent_label == agent_label,
    ]


def _duplicate_of(question_text: str, existing: Sequence[CuratedQuestion]) -> Optional[UUID]:
    """Id of an existing row whose text is the same question, else ``None``.

    Compared with :func:`cognee.modules.recall_coverage.dedup.collapse_text_key`
    — casefolded and whitespace-collapsed — so this module and dedup agree on
    what "the same question" means. Deliberately in Python and not in SQL:
    ``lower()`` is not ``casefold()`` (German ``ß`` folds to ``ss``, so
    ``STRASSE`` and ``Straße`` are one question to dedup and two to SQL), and no
    portable SQL expression collapses interior whitespace. A functional unique
    index cannot enforce it either — SQLite cannot add a constraint after the
    fact — so two racing inserts can still both land; the loser is a duplicate
    row, not corruption, and the next run's dedup merges them.
    """
    key = collapse_text_key(question_text)
    for row in existing:
        if collapse_text_key(row.question_text) == key:
            return row.id
    return None


async def create_curated_question(
    user: Any,
    question_text: str,
    scope: Optional[str] = None,
    agent_label: Optional[str] = None,
    *,
    config: Optional[RecallCoverageConfig] = None,
) -> CuratedQuestion:
    """Add a curated question, refusing a casefold-exact duplicate in the same scope.

    Duplicates are refused rather than merged so the writer learns the question
    is already covered; a silent merge would leave them believing they had added
    something.
    """
    text = (question_text or "").strip()
    if not text:
        raise EmptyCuratedQuestionError()

    normalized_scope, label = normalize_curated_scope(scope, agent_label, user=user, config=config)
    owner_id = resolve_curated_owner(user, normalized_scope)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageCuratedQuestion).where(
                RecallCoverageCuratedQuestion.owner_id == owner_id,
                *_scope_filter(normalized_scope, label),
            )
        )
        existing = [_to_value_object(row) for row in result.scalars().all()]

        duplicate_id = _duplicate_of(text, existing)
        if duplicate_id is not None:
            raise DuplicateCuratedQuestionError(
                message=f"This curated question already exists in this scope (id {duplicate_id})."
            )

        row = RecallCoverageCuratedQuestion(
            owner_id=owner_id,
            scope=normalized_scope,
            agent_label=label,
            question_text=text,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_value_object(row)


async def list_curated_questions(
    user: Any,
    agent_label: Optional[str] = None,
    *,
    config: Optional[RecallCoverageConfig] = None,
) -> list[CuratedQuestion]:
    """The caller's agent-scoped rows plus every shared row, newest first.

    ``agent_label`` narrows the agent-scoped half to one validated label; omitted,
    every one of the caller's agent-scoped rows comes back. Two flat selects and
    no join: the two halves have different owner scopes, so a join would need the
    owner predicate twice anyway.
    """
    label = (agent_label or "").strip()
    resolved_label = resolve_agent_scope(label, user=user, config=config).label if label else None

    agent_owner = resolve_curated_owner(user, CuratedScope.AGENT.value)
    shared_owner = resolve_curated_owner(user, CuratedScope.SHARED.value)

    agent_terms: list[Any] = [
        RecallCoverageCuratedQuestion.owner_id == agent_owner,
        RecallCoverageCuratedQuestion.scope == CuratedScope.AGENT.value,
    ]
    if resolved_label is not None:
        agent_terms.append(RecallCoverageCuratedQuestion.agent_label == resolved_label)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        agent_result = await session.execute(
            select(RecallCoverageCuratedQuestion)
            .where(*agent_terms)
            .order_by(RecallCoverageCuratedQuestion.created_at.desc())
        )
        agent_rows = [_to_value_object(row) for row in agent_result.scalars().all()]

        shared_result = await session.execute(
            select(RecallCoverageCuratedQuestion)
            .where(
                RecallCoverageCuratedQuestion.owner_id == shared_owner,
                RecallCoverageCuratedQuestion.scope == CuratedScope.SHARED.value,
            )
            .order_by(RecallCoverageCuratedQuestion.created_at.desc())
        )
        shared_rows = [_to_value_object(row) for row in shared_result.scalars().all()]

    return agent_rows + shared_rows


async def load_curated_questions_for_scope(
    user: Any,
    scope: AgentScope,
) -> list[CuratedQuestion]:
    """Curated questions one run must include: this label's rows plus every shared row.

    Takes the resolved :class:`AgentScope` rather than a label string, like
    everything else the pipeline calls, so an unvalidated label cannot reach a
    query.
    """
    return await list_curated_questions(user, agent_label=scope.label)


async def delete_curated_question(user: Any, question_id: UUID) -> None:
    """Delete one curated question, or raise 404.

    Filtered by id **and** owner scope, and 404 on either miss: a 403 would
    confirm that a row with this id exists under another owner.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageCuratedQuestion).where(
                RecallCoverageCuratedQuestion.id == question_id,
                RecallCoverageCuratedQuestion.owner_id.in_(curated_owner_ids(user)),
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise CuratedQuestionNotFoundError()

        await session.delete(row)
        await session.commit()


def curated_asks(
    curated: Sequence[CuratedQuestion],
    *,
    user_id: UUID,
    dataset_ids: Sequence[Optional[UUID]],
) -> list[Ask]:
    """Replicate curated questions into one ask per readable dataset partition.

    A curated question has no dataset of its own, and a question row *is*
    ``(user_id, dataset_id, canonical text)`` — so instead of special-casing the
    empty dataset, the question is replicated into every dataset the asking user
    can read and deduped there like anything else. Where it lands within
    ``dedup_threshold`` of that dataset's real traffic it merges and, because
    :func:`cognee.modules.recall_coverage.dedup._canonical_member` prefers the
    curated member, keeps the human's wording while inheriting that partition's
    distinct-ask count. Where it matches nothing it stands alone with
    ``occurrence_count = 0`` and ``was_asked = False``, hence ``impact = 0``.

    Two properties the caller must preserve:

    * these asks are appended **after** the ``max_questions`` truncation, so
      curated questions never displace observed traffic (spec phase 1 step 3);
    * ``first_seen`` / ``last_seen`` stay ``None``. A curated ask is not demand:
      it contributes nothing to ``occurrence_count`` and nothing to
      ``first_asked_at`` / ``last_asked_at``.

    With no readable datasets there is nothing to replicate into, so the question
    still enters the run once, in the NULL-dataset partition — the same partition
    a multi-dataset search lands in, and it replays with ``dataset_ids=None``.
    Dropping it instead would make a curated question silently vanish for exactly
    the user whose memory has nothing in it.
    """
    partitions: list[Optional[UUID]] = []
    for dataset_id in dataset_ids:
        if dataset_id not in partitions:
            partitions.append(dataset_id)
    if not partitions:
        partitions = [None]

    return [
        Ask(
            text=question.question_text,
            user_id=user_id,
            dataset_id=dataset_id,
            first_seen=None,
            last_seen=None,
            query_type=None,
            query_ids=[],
            source=QuestionSource.CURATED.value,
            curated_question_id=question.id,
        )
        for question in curated
        for dataset_id in partitions
    ]


async def load_curated_asks(
    user: Any,
    scope: AgentScope,
    *,
    dataset_ids: Optional[Sequence[UUID]] = None,
) -> list[Ask]:
    """Load this run's curated questions and replicate them into dataset partitions.

    The partitions are the datasets **the caller** can read, and the asks are
    attributed to the caller. A teammate who asked the same thing keeps their own
    row: coverage is answered out of a specific user's readable brains, so
    merging the two would report one user's answer under the other's name.

    ``dataset_ids`` is accepted so a caller that already resolved its datasets
    (the pipeline needs their names anyway) does not resolve them twice.
    """
    curated = await load_curated_questions_for_scope(user, scope)
    if not curated:
        return []

    if dataset_ids is None:
        datasets = await get_authorized_existing_datasets(
            datasets=None, permission_type="read", user=user
        )
        dataset_ids = [dataset.id for dataset in datasets]

    logger.debug(
        "recall_coverage: %s curated questions replicated into %s dataset partitions",
        len(curated),
        len(dataset_ids) or 1,
    )
    return curated_asks(curated, user_id=user.id, dataset_ids=list(dataset_ids))


# --- Topics and topic suggestions -------------------------------------------


@dataclass(frozen=True)
class TopicRecord:
    """One active topic, detached from the session that read it.

    ``centroid`` is a tuple rather than a list so the record stays immutable all
    the way into the matmul: a phase that mutated a centroid in place would
    change what later runs are scored against without touching the database.
    """

    id: UUID
    owner_id: UUID
    label: str
    centroid: tuple[float, ...]
    embedding_model: str
    embedding_dimensions: int
    seed_question_count: int = 0
    taxonomy_version: int = 0


@dataclass(frozen=True)
class SuggestionRecord:
    """One topic suggestion, detached from the session that read it."""

    id: UUID
    owner_id: UUID
    label: str
    centroid: tuple[float, ...]
    embedding_model: str
    embedding_dimensions: int
    question_count: int
    cohesion: Optional[float]
    status: str
    agent_label: Optional[str] = None
    run_id: Optional[UUID] = None
    accepted_topic_id: Optional[UUID] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class SuggestionDraft:
    """A suggestion about to be written. Always lands as ``pending``.

    The topic id is minted on accept, not here — that is what makes an accepted
    topic id stable across runs, and it is why this carries no id of its own.
    """

    owner_id: UUID
    label: str
    centroid: tuple[float, ...]
    embedding_model: str
    embedding_dimensions: int
    question_count: int
    cohesion: Optional[float] = None
    agent_label: Optional[str] = None
    run_id: Optional[UUID] = None


def _to_topic_record(row: RecallCoverageTopic) -> TopicRecord:
    return TopicRecord(
        id=row.id,
        owner_id=row.owner_id,
        label=row.label,
        centroid=tuple(float(value) for value in (row.centroid or ())),
        embedding_model=row.embedding_model,
        embedding_dimensions=row.embedding_dimensions,
        seed_question_count=row.seed_question_count or 0,
        taxonomy_version=row.taxonomy_version or 0,
    )


def _to_suggestion_record(row: RecallCoverageTopicSuggestion) -> SuggestionRecord:
    return SuggestionRecord(
        id=row.id,
        owner_id=row.owner_id,
        label=row.label,
        centroid=tuple(float(value) for value in (row.centroid or ())),
        embedding_model=row.embedding_model,
        embedding_dimensions=row.embedding_dimensions,
        question_count=row.question_count or 0,
        cohesion=row.cohesion,
        status=row.status,
        agent_label=row.agent_label,
        run_id=row.run_id,
        accepted_topic_id=row.accepted_topic_id,
        created_at=row.created_at,
    )


async def load_active_topics(owner_id: UUID) -> list[TopicRecord]:
    """Every topic the owner has accepted and not deleted, oldest first.

    Owner-scoped with **no ``agent_label`` filter**: one taxonomy serves all of an
    owner's agents, so a Codex run and a Claude Code run are scored against the
    same topics and their per-topic scores are comparable. Deleted topics are
    excluded (soft delete keeps ``taxonomy_version`` monotone), and their
    questions fall back to the sink on the next run rather than disappearing.

    Ordered by ``created_at`` then ``id`` so the centroid matrix — and therefore
    which topic wins an exact similarity tie — is stable between runs.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageTopic)
            .where(
                RecallCoverageTopic.owner_id == owner_id,
                RecallCoverageTopic.deleted_at.is_(None),
            )
            .order_by(RecallCoverageTopic.created_at, RecallCoverageTopic.id)
        )
        return [_to_topic_record(row) for row in result.scalars().all()]


async def load_settled_suggestions(owner_id: UUID) -> list[SuggestionRecord]:
    """The owner's accepted and dismissed suggestions — the re-proposal guard's input.

    Owner-scoped and label-blind on purpose: dismissing a suggestion is a
    statement about the owner's taxonomy, so it must also suppress the same theme
    on another agent's run. ``pending`` rows are excluded — an undecided
    suggestion is not a decision, so re-surfacing the same theme is legitimate.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageTopicSuggestion)
            .where(
                RecallCoverageTopicSuggestion.owner_id == owner_id,
                RecallCoverageTopicSuggestion.status.in_(
                    (SuggestionStatus.ACCEPTED.value, SuggestionStatus.DISMISSED.value)
                ),
            )
            .order_by(RecallCoverageTopicSuggestion.created_at)
        )
        return [_to_suggestion_record(row) for row in result.scalars().all()]


async def load_pending_suggestions(owner_id: UUID) -> list[SuggestionRecord]:
    """The owner's undecided suggestions, newest first. The review queue."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageTopicSuggestion)
            .where(
                RecallCoverageTopicSuggestion.owner_id == owner_id,
                RecallCoverageTopicSuggestion.status == SuggestionStatus.PENDING.value,
            )
            .order_by(RecallCoverageTopicSuggestion.created_at.desc())
        )
        return [_to_suggestion_record(row) for row in result.scalars().all()]


async def create_topic_suggestions(
    drafts: Sequence[SuggestionDraft],
) -> list[SuggestionRecord]:
    """Write ``pending`` suggestion rows, in one transaction, in draft order."""
    if not drafts:
        return []

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = [
            RecallCoverageTopicSuggestion(
                owner_id=draft.owner_id,
                agent_label=draft.agent_label,
                run_id=draft.run_id,
                label=draft.label,
                centroid=list(draft.centroid),
                embedding_model=draft.embedding_model,
                embedding_dimensions=draft.embedding_dimensions,
                question_count=draft.question_count,
                cohesion=draft.cohesion,
                status=SuggestionStatus.PENDING.value,
            )
            for draft in drafts
        ]
        session.add_all(rows)
        await session.commit()
        for row in rows:
            await session.refresh(row)
        return [_to_suggestion_record(row) for row in rows]


__all__ = [
    "CuratedQuestion",
    "SuggestionDraft",
    "SuggestionRecord",
    "TopicRecord",
    "create_curated_question",
    "create_topic_suggestions",
    "curated_asks",
    "curated_owner_ids",
    "delete_curated_question",
    "list_curated_questions",
    "load_active_topics",
    "load_curated_asks",
    "load_curated_questions_for_scope",
    "load_pending_suggestions",
    "load_settled_suggestions",
    "normalize_curated_scope",
    "resolve_curated_owner",
]
