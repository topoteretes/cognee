"""Persistence for recall coverage: runs, question rows, curated questions, topics.

Spec sections 2 (phases 2 and 4) and 4. A curated question is one a human typed that
memory *should* answer, whether or not any agent asked it. It is deliberately
**not** a separate report: curated questions enter the same window, the same
dedup, the same replay and the same judge as observed traffic, and come back in
the same ``questions[]`` with ``source = "curated"``.

Topics and topic suggestions are **owner-scoped**, never per-agent and never
per-dataset: one taxonomy across all of an owner's agents is what makes "Codex
4.2 on Billing, Claude Code 2.1 on Billing" a sentence at all. ``agent_label`` on
a suggestion is provenance ("this came out of the Codex run"), never scope. Their
lifecycle is **accept, delete, dismiss and nothing else**, with
``taxonomy_version`` monotone per owner — see the section comment above
:func:`parse_topic_id`.

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

The run half (phase 4 step 14) is the only writer of ``recall_coverage_runs`` and
``recall_coverage_questions``. A run's rows and its frozen ``summary`` are written
in **one** transaction by :func:`persist_run_results`: a run whose rows landed but
whose summary did not would report ``overall_score: null`` over a table full of
scores, which reads as "memory answered nothing" rather than as the partial write
it is.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence, Union
from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.aggregate import (
    CoverageRow,
    CoverageSummary,
    RunCounters,
    default_row_order,
    report_order_key,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import Ask, collapse_text_key
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunNotFoundError,
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
    EmptyCuratedQuestionError,
    InvalidCuratedQuestionScopeError,
    SinkTopicNotEditableError,
)
from cognee.modules.recall_coverage.models import (
    RecallCoverageCuratedQuestion,
    RecallCoverageQuestion,
    RecallCoverageRun,
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_ID,
    AgentScope,
    AgentScopeMode,
    CoverageParams,
    CuratedScope,
    QuestionSource,
    RunStatus,
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

    An ``all`` run loads **every** agent-scoped row rather than the rows labelled
    with the literal ``"all"``. ``all`` is "no session predicate at all — every
    recall in the window regardless of session" (spec section 1), so the curated
    half has to mean every agent too. Narrowing on the label there would leave a
    question added for ``claude-code`` out of the default run — the only mode that
    returns rows until ``Query.session_id`` ships — with nothing to tell the
    writer their question was silently out of scope.
    """
    label = None if scope.mode is AgentScopeMode.ALL else scope.label
    return await list_curated_questions(user, agent_label=label)


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
    # Soft delete. A deleted topic keeps its row so ``taxonomy_version`` stays
    # monotone and historical runs keep resolving their topic ids; its questions
    # fall back to the sink on the next run rather than disappearing.
    deleted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


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
        deleted_at=row.deleted_at,
        created_at=row.created_at,
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


# --- Topic lifecycle: accept, delete, dismiss --------------------------------
#
# Those three and nothing else. There is deliberately no rename, no merge and no
# split, and hence no ``merged_into_id`` column: a topic id is the join key a
# score trend is carried on, so an operation that silently changes what an id
# means — a merge above all — would rewrite history that looks unchanged.
# Renaming is the same problem in miniature, since every historical run froze the
# label it reported. Delete a topic and accept a new suggestion instead: the trend
# then visibly starts over, which is the truth.
#
# ``taxonomy_version`` is monotone per owner. It is derived from
# ``max(taxonomy_version)`` over **all** of the owner's topics, deleted included,
# and both accept and delete stamp the bumped value onto the row they touch. That
# is what soft delete buys: a hard delete would drop the highest version with the
# row, the next accept would reuse a number a historical run already reported, and
# two runs claiming version 4 would be scored against different taxonomies.


def parse_topic_id(value: Union[str, UUID]) -> UUID:
    """Turn a path-parameter topic id into a UUID, or raise the right error.

    The sink is the wire literal ``"other"`` and not a row, so asking to delete it
    is a 422 rather than a 404: the id is real and recognised, it just does not
    name something that can be modified. Anything else unparseable is a 404 — it
    names nothing, and reporting *why* it is malformed tells a caller nothing they
    can act on.
    """
    if isinstance(value, UUID):
        return value

    text = str(value or "").strip()
    if text.casefold() == SINK_TOPIC_ID:
        raise SinkTopicNotEditableError()

    try:
        return UUID(text)
    except (ValueError, AttributeError, TypeError):
        raise CoverageTopicNotFoundError(message=f"Recall coverage topic not found: {text!r}")


async def _max_taxonomy_version(session: Any, owner_id: UUID) -> int:
    """The owner's highest taxonomy version, counting deleted topics.

    Counting deleted topics is the whole point of the soft delete; see the section
    comment above. ``0`` for an owner with no topics at all, which is also what a
    run stamps before the first suggestion is accepted.
    """
    result = await session.execute(
        select(func.max(RecallCoverageTopic.taxonomy_version)).where(
            RecallCoverageTopic.owner_id == owner_id
        )
    )
    return int(result.scalar() or 0)


async def current_taxonomy_version(owner_id: UUID) -> int:
    """The version a run records as the taxonomy it was scored against."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return await _max_taxonomy_version(session, owner_id)


async def list_topics(
    owner_ids: Sequence[UUID], *, include_deleted: bool = False
) -> list[TopicRecord]:
    """The owner scope's topics, oldest first, deleted ones only when asked for.

    Same order as :func:`load_active_topics` (``created_at`` then ``id``) so the
    list a reviewer sees is the order the assignment matmul used. No
    ``agent_label`` filter exists to offer: one taxonomy serves all of an owner's
    agents, which is what makes two agents' per-topic scores comparable.
    """
    terms: list[Any] = [RecallCoverageTopic.owner_id.in_(tuple(owner_ids))]
    if not include_deleted:
        terms.append(RecallCoverageTopic.deleted_at.is_(None))

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageTopic)
            .where(*terms)
            .order_by(RecallCoverageTopic.created_at, RecallCoverageTopic.id)
        )
        return [_to_topic_record(row) for row in result.scalars().all()]


async def _load_suggestion_for_decision(
    session: Any, suggestion_id: UUID, owner_ids: Sequence[UUID]
) -> RecallCoverageTopicSuggestion:
    """A pending suggestion in the caller's owner scope, or the matching error.

    404 on an id that is not in scope (never 403 — that would confirm someone
    else's suggestion with this id exists) and 409 on one that has already been
    decided. Re-accepting is refused rather than treated as idempotent because it
    would mint a *second* topic id for the same cluster, and two ids for one theme
    split its trend in half.
    """
    result = await session.execute(
        select(RecallCoverageTopicSuggestion).where(
            RecallCoverageTopicSuggestion.id == suggestion_id,
            RecallCoverageTopicSuggestion.owner_id.in_(tuple(owner_ids)),
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise CoverageSuggestionNotFoundError()

    if row.status != SuggestionStatus.PENDING.value:
        raise CoverageSuggestionNotPendingError(
            message=(
                f"This topic suggestion is already {row.status}; only a pending "
                "suggestion can be accepted or dismissed."
            )
        )
    return row


async def accept_topic_suggestion(
    suggestion_id: UUID, owner_ids: Sequence[UUID]
) -> tuple[TopicRecord, SuggestionRecord]:
    """Mint the topic a suggestion described, in one transaction.

    **This call is where a topic id comes from.** The suggestion carried none
    precisely so that the id is created by a human decision and then never moves:
    every later run assigns questions to that id, and the per-topic score trend is
    keyed on it.

    The topic inherits the suggestion's owner rather than the caller's, so a
    tenant peer accepting a suggestion cannot fork the taxonomy into a second
    owner scope where nothing else would ever be scored against it. The centroid,
    fingerprint and question count are copied verbatim: re-embedding the label
    would put the topic in a different place from the cluster that motivated it.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        suggestion = await _load_suggestion_for_decision(session, suggestion_id, owner_ids)

        version = await _max_taxonomy_version(session, suggestion.owner_id) + 1

        topic = RecallCoverageTopic(
            owner_id=suggestion.owner_id,
            label=suggestion.label,
            centroid=list(suggestion.centroid or []),
            embedding_model=suggestion.embedding_model,
            embedding_dimensions=suggestion.embedding_dimensions,
            seed_question_count=suggestion.question_count or 0,
            taxonomy_version=version,
        )
        session.add(topic)
        await session.flush()

        suggestion.status = SuggestionStatus.ACCEPTED.value
        suggestion.accepted_topic_id = topic.id

        await session.commit()
        await session.refresh(topic)
        await session.refresh(suggestion)

        logger.debug(
            "recall_coverage: accepted suggestion %s as topic %s at taxonomy version %s",
            suggestion_id,
            topic.id,
            version,
        )
        return _to_topic_record(topic), _to_suggestion_record(suggestion)


async def dismiss_topic_suggestion(
    suggestion_id: UUID, owner_ids: Sequence[UUID]
) -> SuggestionRecord:
    """Refuse a suggestion, permanently and across every agent label.

    No taxonomy version bump: nothing about the taxonomy changed, so a run scored
    before and after this call was scored against the same topics. The row is kept
    rather than deleted because it *is* the record of the decision — the
    re-proposal guard reads dismissed suggestions to keep the same dense sink
    cluster from being proposed again on every run, on every agent.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        suggestion = await _load_suggestion_for_decision(session, suggestion_id, owner_ids)
        suggestion.status = SuggestionStatus.DISMISSED.value
        await session.commit()
        await session.refresh(suggestion)
        return _to_suggestion_record(suggestion)


async def delete_topic(topic_id: UUID, owner_ids: Sequence[UUID]) -> int:
    """Soft-delete a topic and return the owner's new taxonomy version.

    Soft because a hard delete would take three things with it: the monotonicity
    of ``taxonomy_version`` (see the section comment), the ability of a historical
    run to resolve the ``topic_id`` on its own question rows, and the audit trail
    of a taxonomy that was edited. The topic's questions are **never** deleted —
    the next run simply finds no active topic for them and they land in the sink,
    which is exactly the "your taxonomy is missing something" signal the sink
    exists to give.

    Deleting an already-deleted topic is idempotent and does **not** bump the
    version again: a retried request must not inflate a number that historical
    runs are compared on.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageTopic).where(
                RecallCoverageTopic.id == topic_id,
                RecallCoverageTopic.owner_id.in_(tuple(owner_ids)),
            )
        )
        topic = result.scalar_one_or_none()
        if topic is None:
            raise CoverageTopicNotFoundError()

        if topic.deleted_at is not None:
            return await _max_taxonomy_version(session, topic.owner_id)

        version = await _max_taxonomy_version(session, topic.owner_id) + 1
        topic.deleted_at = _utc_now()
        topic.taxonomy_version = version

        await session.commit()

        logger.debug(
            "recall_coverage: soft-deleted topic %s, taxonomy version now %s", topic_id, version
        )
        return version


# --- Runs and question rows --------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunRecord:
    """One coverage run, detached from the session that read it.

    ``summary`` is the frozen report (see
    :class:`cognee.modules.recall_coverage.aggregate.CoverageSummary`) on a
    complete run, ``{"error": ...}`` on a failed one, and ``None`` before the run
    finished. ``params`` is the :class:`CoverageParams` snapshot it executed
    under, so a historical run stays readable after the deployment's defaults
    move.
    """

    id: UUID
    agent_label: str
    owner_id: UUID
    status: str
    params: Optional[dict]
    summary: Optional[dict]
    finished_at: Optional[datetime]
    recall_row_count: int = 0
    distinct_ask_count: int = 0
    collapsed_retry_count: int = 0
    question_row_count: int = 0
    curated_question_count: int = 0
    topic_count: int = 0
    dataset_count: int = 0
    user_count: int = 0
    taxonomy_version: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_in_flight(self) -> bool:
        """Pending or running: what the ``(owner, agent_label)`` 409 guard tests."""
        return self.status in (RunStatus.PENDING.value, RunStatus.RUNNING.value)


@dataclass(frozen=True)
class QuestionRecord:
    """One persisted question row, detached from the session that read it.

    ``topic_id`` is ``None`` for the sink — the wire literal ``"other"`` is
    produced by the DTO, never stored — so a reader must test the id and not the
    string.
    """

    id: UUID
    run_id: UUID
    question_group_id: Optional[UUID]
    user_id: UUID
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    question_text: str
    source: str
    was_asked: bool
    curated_question_id: Optional[UUID]
    answer: Optional[str]
    judge_score: Optional[int]
    judge_answered: Optional[bool]
    retrieval_context: Optional[str]
    error: Optional[str]
    topic_id: Optional[UUID]
    first_asked_at: Optional[datetime]
    last_asked_at: Optional[datetime]
    occurrence_count: int
    impact: Optional[float]

    @property
    def is_curated(self) -> bool:
        """Read by :func:`report_order_key`, which pins curated rows to the top.

        Provenance, like :attr:`aggregate.CoverageRow.is_curated`: a row that was
        both curated and asked (``was_asked`` with ``source = "curated"``) is
        pinned, and still counts as demand everywhere the aggregates look.
        """
        return self.source == QuestionSource.CURATED.value


def _to_run_record(row: RecallCoverageRun) -> RunRecord:
    return RunRecord(
        id=row.id,
        agent_label=row.agent_label,
        owner_id=row.owner_id,
        status=row.status,
        params=row.params,
        summary=row.summary,
        finished_at=row.finished_at,
        recall_row_count=row.recall_row_count or 0,
        distinct_ask_count=row.distinct_ask_count or 0,
        collapsed_retry_count=row.collapsed_retry_count or 0,
        question_row_count=row.question_row_count or 0,
        curated_question_count=row.curated_question_count or 0,
        topic_count=row.topic_count or 0,
        dataset_count=row.dataset_count or 0,
        user_count=row.user_count or 0,
        taxonomy_version=row.taxonomy_version or 0,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_question_record(row: RecallCoverageQuestion) -> QuestionRecord:
    return QuestionRecord(
        id=row.id,
        run_id=row.run_id,
        question_group_id=row.question_group_id,
        user_id=row.user_id,
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        question_text=row.question_text,
        source=row.source,
        was_asked=bool(row.was_asked),
        curated_question_id=row.curated_question_id,
        answer=row.answer,
        judge_score=row.judge_score,
        judge_answered=row.judge_answered,
        retrieval_context=row.retrieval_context,
        error=row.error,
        topic_id=row.topic_id,
        first_asked_at=row.first_asked_at,
        last_asked_at=row.last_asked_at,
        occurrence_count=row.occurrence_count or 0,
        impact=row.impact,
    )


def _question_model(run_id: UUID, row: CoverageRow) -> RecallCoverageQuestion:
    """Map one aggregated row onto its ORM row.

    ``topic_label`` and ``is_shared_curated`` are deliberately not persisted: the
    label lives on the topic row (and "Other" is not a row at all), and shared
    membership is a property of the curated question, reachable through
    ``curated_question_id``. Storing either would let a run's copy drift from the
    thing it describes.
    """
    return RecallCoverageQuestion(
        run_id=run_id,
        question_group_id=row.question_group_id,
        user_id=row.user_id,
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        question_text=row.question_text,
        source=row.source,
        was_asked=row.was_asked,
        curated_question_id=row.curated_question_id,
        answer=row.answer,
        judge_score=row.judge_score,
        judge_answered=row.judge_answered,
        retrieval_context=row.retrieval_context,
        error=row.error,
        topic_id=row.topic_id,
        first_asked_at=row.first_asked_at,
        last_asked_at=row.last_asked_at,
        occurrence_count=row.occurrence_count,
        impact=row.impact,
    )


async def create_run(
    owner_id: UUID,
    agent_label: str,
    *,
    params: Optional[CoverageParams] = None,
    taxonomy_version: int = 0,
) -> RunRecord:
    """Insert a ``pending`` run row and return it.

    Written **before** the coroutine is scheduled so the row is what the in-flight
    guard sees and what the 202 response reports: a run that existed only inside a
    task would be invisible to the guard, and two concurrent requests would both
    start replaying and judging the same window at full LLM cost.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = RecallCoverageRun(
            owner_id=owner_id,
            agent_label=agent_label,
            status=RunStatus.PENDING.value,
            params=params.model_dump(mode="json") if params is not None else None,
            taxonomy_version=taxonomy_version,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _to_run_record(row)


async def _load_run(session: Any, run_id: UUID) -> RecallCoverageRun:
    """The run row inside an open session, for the writers only.

    Deliberately not owner-filtered: its three callers are the pipeline's own
    status transitions, driven by a ``run_id`` the pipeline minted itself. Owner
    scope belongs on the *reads* a request can reach — :func:`get_run`,
    :func:`list_runs` — and adding it here would mean threading the caller into a
    background task that has no request.
    """
    result = await session.execute(select(RecallCoverageRun).where(RecallCoverageRun.id == run_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise CoverageRunNotFoundError()
    return row


async def mark_run_running(run_id: UUID) -> RunRecord:
    """Flip a pending run to ``running``. The coroutine's first act."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await _load_run(session, run_id)
        row.status = RunStatus.RUNNING.value
        await session.commit()
        await session.refresh(row)
        return _to_run_record(row)


async def persist_run_results(
    run_id: UUID,
    rows: Sequence[CoverageRow],
    summary: CoverageSummary,
    counters: RunCounters,
    *,
    params: Optional[CoverageParams] = None,
) -> RunRecord:
    """Write the run's question rows and its frozen summary in one transaction.

    Rows are inserted in :func:`cognee.modules.recall_coverage.aggregate.default_row_order`
    — curated first, then by impact — but nothing depends on insertion order:
    ``load_run_questions`` re-applies that order in Python, because the table has
    no ordering column and ``id`` is a ``uuid4``.

    ``params`` is optional: :func:`create_run` already stored the snapshot, and it
    is accepted here only so a caller that resolved overrides after inserting the
    row can correct it. An empty ``rows`` is a legitimate complete run — an empty
    window completes immediately with ``overall_score: null`` and no questions,
    which is the *expected* result for every label but ``all`` until
    ``Query.session_id`` ships, and must read as "nothing asked yet" rather than
    as a failure.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        run = await _load_run(session, run_id)

        if rows:
            session.add_all([_question_model(run_id, row) for row in default_row_order(rows)])

        run.status = RunStatus.COMPLETE.value
        run.finished_at = _utc_now()
        run.summary = summary.to_dict()
        if params is not None:
            run.params = params.model_dump(mode="json")

        run.recall_row_count = counters.recall_row_count
        run.distinct_ask_count = counters.distinct_ask_count
        run.collapsed_retry_count = counters.collapsed_retry_count
        run.question_row_count = counters.question_row_count
        run.curated_question_count = counters.curated_question_count
        run.topic_count = counters.topic_count
        run.dataset_count = counters.dataset_count
        run.user_count = counters.user_count
        run.taxonomy_version = counters.taxonomy_version

        await session.commit()
        await session.refresh(run)

        logger.debug(
            "recall_coverage: run %s complete with %s question rows",
            run_id,
            counters.question_row_count,
        )
        return _to_run_record(run)


async def fail_run(run_id: UUID, message: Optional[str] = None) -> RunRecord:
    """Mark a run ``failed``, recording why in ``summary``.

    The message goes into ``summary`` rather than into a column of its own: a
    failed run has no breakdowns, so the field is free, and adding an ``error``
    column would invite writing partial breakdowns next to it — at which point a
    reader could not tell a complete report from half of one. Any question rows
    already inserted are left alone; they belong to a run whose status says not to
    trust its numbers.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await _load_run(session, run_id)
        row.status = RunStatus.FAILED.value
        row.finished_at = _utc_now()
        row.summary = {"error": message} if message else {"error": "Recall coverage run failed."}
        await session.commit()
        await session.refresh(row)
        return _to_run_record(row)


async def get_run(run_id: UUID, owner_ids: Sequence[UUID]) -> RunRecord:
    """One run, filtered by id **and** owner scope, 404 on either miss.

    Never 403: that would confirm a run with this id exists under another owner.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageRun).where(
                RecallCoverageRun.id == run_id,
                RecallCoverageRun.owner_id.in_(tuple(owner_ids)),
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise CoverageRunNotFoundError()
        return _to_run_record(row)


async def list_runs(
    owner_ids: Sequence[UUID],
    agent_label: Optional[str] = None,
    *,
    limit: Optional[int] = None,
) -> list[RunRecord]:
    """The caller's runs, newest first, optionally narrowed to one label.

    ``limit=None`` applies no ``LIMIT``; the route supplies
    ``runs_list_default_limit``. Ordered by ``created_at`` then ``id`` so two runs
    created inside the same clock tick still come back in a stable order.
    """
    terms: list[Any] = [RecallCoverageRun.owner_id.in_(tuple(owner_ids))]
    if agent_label:
        terms.append(RecallCoverageRun.agent_label == agent_label)

    statement = (
        select(RecallCoverageRun)
        .where(*terms)
        .order_by(RecallCoverageRun.created_at.desc(), RecallCoverageRun.id)
    )
    if limit is not None:
        statement = statement.limit(limit)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(statement)
        return [_to_run_record(row) for row in result.scalars().all()]


async def runs_in_flight(
    owner_id: UUID, agent_label: str, *, stale_after_seconds: Optional[int] = None
) -> list[RunRecord]:
    """Pending or running runs for this ``(owner_id, agent_label)``, newest first.

    The 409 guard's input. Scoped to the single owner rather than to
    ``curated_owner_ids``: a run is started by one caller and paid for by them, so
    a teammate's run must not block theirs — unlike curated questions, where the
    shared benchmark set is deliberately maintained by anyone in the tenant.

    ``stale_after_seconds`` bounds how long a row keeps blocking. Status alone is
    not liveness: the background task lives in one process, so a pod rescheduled
    mid-run leaves ``running`` on the row for ever and an unbounded guard would
    then 409 every later run for that pair — with no cancel or delete route, the
    only recovery would be manual SQL. ``None`` keeps the unbounded behaviour, for
    callers that want to see the row whatever its age.
    """
    terms: list[Any] = [
        RecallCoverageRun.owner_id == owner_id,
        RecallCoverageRun.agent_label == agent_label,
        RecallCoverageRun.status.in_((RunStatus.PENDING.value, RunStatus.RUNNING.value)),
    ]
    if stale_after_seconds is not None and stale_after_seconds > 0:
        terms.append(
            RecallCoverageRun.created_at >= _utc_now() - timedelta(seconds=stale_after_seconds)
        )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageRun).where(*terms).order_by(RecallCoverageRun.created_at.desc())
        )
        return [_to_run_record(row) for row in result.scalars().all()]


async def load_run_questions(run_id: UUID) -> list[QuestionRecord]:
    """A run's question rows in the default report order.

    The order is re-applied here rather than trusted from the table: there is no
    ordering column, ``id`` is a ``uuid4``, and every row in a run shares a
    ``created_at`` to the microsecond. Read-side ordering also means changing the
    default order does not require a migration.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageQuestion).where(RecallCoverageQuestion.run_id == run_id)
        )
        records = [_to_question_record(row) for row in result.scalars().all()]

    return sorted(records, key=report_order_key)


async def latest_complete_runs(
    owner_ids: Sequence[UUID], agent_labels: Optional[Sequence[str]] = None
) -> dict[str, RunRecord]:
    """The newest **complete** run per agent label, keyed by label.

    Complete only: a pending run has no numbers and a failed one has numbers
    nobody should read, so joining either to an agent row would show a coverage
    score that no finished run ever produced. Newest first, first wins.
    """
    terms: list[Any] = [
        RecallCoverageRun.owner_id.in_(tuple(owner_ids)),
        RecallCoverageRun.status == RunStatus.COMPLETE.value,
    ]
    if agent_labels:
        terms.append(RecallCoverageRun.agent_label.in_(tuple(agent_labels)))

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(RecallCoverageRun)
            .where(*terms)
            .order_by(RecallCoverageRun.created_at.desc(), RecallCoverageRun.id)
        )
        latest: dict[str, RunRecord] = {}
        for row in result.scalars().all():
            latest.setdefault(row.agent_label, _to_run_record(row))
        return latest


@dataclass(frozen=True)
class BenchmarkCell:
    """One cell of the datasets x agents benchmark matrix.

    Restricted to **shared** curated rows: identical prompts across agents is the
    only reason two agents' numbers are comparable at all, so an agent-scoped
    curated row — one person's list for one tool — would make the comparable
    number not comparable.
    """

    agent_label: str
    run_id: UUID
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    question_count: int
    scored_question_count: int
    avg_score: Optional[float]


async def benchmark_cells(run_ids_by_label: Mapping[str, UUID]) -> list[BenchmarkCell]:
    """Group the shared curated rows of the given runs by ``(label, dataset)``.

    One ``GROUP BY`` over ``recall_coverage_questions``, joined to
    ``recall_coverage_curated_questions`` for the scope — the shared/agent
    distinction lives on the curated question, deliberately not copied onto the
    question row, so that changing a question's scope cannot leave two records
    disagreeing.

    ``AVG`` and the scored count both ignore NULL scores, which is the same rule
    :mod:`cognee.modules.recall_coverage.aggregate` applies in Python: a row we
    could not judge is absent from the mean rather than counted as a zero.
    """
    if not run_ids_by_label:
        return []

    label_by_run = {run_id: label for label, run_id in run_ids_by_label.items()}

    statement = (
        select(
            RecallCoverageQuestion.run_id,
            RecallCoverageQuestion.dataset_id,
            func.max(RecallCoverageQuestion.dataset_name).label("dataset_name"),
            func.count(RecallCoverageQuestion.id).label("question_count"),
            func.count(RecallCoverageQuestion.judge_score).label("scored_question_count"),
            func.avg(RecallCoverageQuestion.judge_score).label("avg_score"),
        )
        .join(
            RecallCoverageCuratedQuestion,
            RecallCoverageQuestion.curated_question_id == RecallCoverageCuratedQuestion.id,
        )
        .where(
            RecallCoverageQuestion.run_id.in_(tuple(label_by_run)),
            RecallCoverageQuestion.source == QuestionSource.CURATED.value,
            RecallCoverageCuratedQuestion.scope == CuratedScope.SHARED.value,
        )
        .group_by(RecallCoverageQuestion.run_id, RecallCoverageQuestion.dataset_id)
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (await session.execute(statement)).all()

    cells = [
        BenchmarkCell(
            agent_label=label_by_run[row.run_id],
            run_id=row.run_id,
            dataset_id=row.dataset_id,
            dataset_name=row.dataset_name,
            question_count=int(row.question_count or 0),
            scored_question_count=int(row.scored_question_count or 0),
            avg_score=None if row.avg_score is None else float(row.avg_score),
        )
        for row in rows
    ]

    return sorted(
        cells,
        key=lambda cell: (cell.agent_label, cell.dataset_name or "", str(cell.dataset_id)),
    )


__all__ = [
    "BenchmarkCell",
    "CuratedQuestion",
    "QuestionRecord",
    "RunRecord",
    "SuggestionDraft",
    "SuggestionRecord",
    "TopicRecord",
    "accept_topic_suggestion",
    "benchmark_cells",
    "create_curated_question",
    "create_run",
    "create_topic_suggestions",
    "curated_asks",
    "curated_owner_ids",
    "current_taxonomy_version",
    "delete_curated_question",
    "delete_topic",
    "dismiss_topic_suggestion",
    "fail_run",
    "get_run",
    "list_curated_questions",
    "list_runs",
    "list_topics",
    "load_active_topics",
    "load_curated_asks",
    "load_curated_questions_for_scope",
    "load_pending_suggestions",
    "load_run_questions",
    "load_settled_suggestions",
    "latest_complete_runs",
    "mark_run_running",
    "normalize_curated_scope",
    "parse_topic_id",
    "persist_run_results",
    "resolve_curated_owner",
    "runs_in_flight",
]
