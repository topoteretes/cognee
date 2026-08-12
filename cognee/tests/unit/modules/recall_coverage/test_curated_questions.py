"""Guards on curated (human-added) recall-coverage questions — spec section 4.

Four invariants, each of which decides whether a reported number means anything:

* **Owner scope.** ``shared`` rows are the benchmark set and belong to the
  tenant when there is one; ``agent`` rows belong to the caller. Get this
  backwards and ``benchmark_score_pct`` compares two people's private prompt
  sets to each other.
* **The duplicate guard is casefold-exact and scoped.** The same question twice
  in one scope is refused; the same question in two different scopes is not.
* **Replication.** A curated question has no dataset, so it enters one partition
  per readable dataset — never a special empty-dataset case, and never dropped
  when the user has no datasets at all.
* **Merging.** Where a curated question lands on top of real traffic it keeps the
  human's wording and inherits that partition's distinct-ask count; where it
  matches nothing it stands alone with ``occurrence_count = 0``, hence
  ``impact = 0``.

Every similarity below comes from an explicit vector table, so no assertion here
depends on a live embedding provider, and the config is always built with
``_env_file=None`` so a developer's ``.env`` cannot move a threshold.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import collapse_asks, dedup_asks
from cognee.modules.recall_coverage.embedding import normalize_rows
from cognee.modules.recall_coverage.exceptions import (
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
    EmptyCuratedQuestionError,
    InvalidCuratedQuestionScopeError,
    UnknownAgentLabelError,
)
from cognee.modules.recall_coverage.models import RecallCoverageCuratedQuestion
from cognee.modules.recall_coverage.types import (
    AgentScope,
    AgentScopeMode,
    CuratedScope,
    QuestionSource,
)
from cognee.modules.search.operations.get_queries import QueryWindowRow

repository = importlib.import_module("cognee.modules.recall_coverage.repository")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

RUNBOOKS = "Where are the runbooks?"
# What a human would type for the same thing: cosine ~0.995 to RUNBOOKS below,
# so it merges at any sane ``dedup_threshold``.
CURATED_RUNBOOKS = "Where do I find our runbooks?"
CREDENTIALS = "How do I rotate credentials?"

VECTORS: dict[str, list[float]] = {
    RUNBOOKS: [1.0, 0.0, 0.0],
    CURATED_RUNBOOKS: [10.0, 1.0, 0.0],
    CREDENTIALS: [0.0, 0.0, 1.0],
}


def _config() -> RecallCoverageConfig:
    """A config that ignores the developer's ``.env``."""
    return RecallCoverageConfig(_env_file=None)


def _user(tenant_id=None) -> SimpleNamespace:
    """The minimum ``get_authenticated_user`` hands a route."""
    return SimpleNamespace(id=uuid4(), tenant_id=tenant_id)


def _scope(label: str = "claude-code") -> AgentScope:
    return AgentScope(label=label, prefixes=(), mode=AgentScopeMode.PREFIX)


async def _create(user, text, scope=CuratedScope.AGENT.value, agent_label="claude-code"):
    return await repository.create_curated_question(
        user,
        text,
        scope,
        None if scope == CuratedScope.SHARED.value else agent_label,
        config=_config(),
    )


def _row(text, *, user_id, dataset_id, created_at) -> QueryWindowRow:
    return QueryWindowRow(
        query_id=uuid4(),
        text=text,
        query_type="GRAPH_COMPLETION",
        user_id=user_id,
        dataset_id=dataset_id,
        created_at=created_at,
    )


def _matrix(asks):
    return normalize_rows([VECTORS[ask.text] for ask in asks])


@pytest_asyncio.fixture
async def curated_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the curated-questions table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="curated_questions_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all, tables=[RecallCoverageCuratedQuestion.__table__]
        )

    monkeypatch.setattr(repository, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


# --- CRUD ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_then_list_returns_the_row(curated_engine):
    user = _user()

    created = await _create(user, "  Where are the runbooks?  ")

    # Outer whitespace is stripped; the wording is otherwise the human's.
    assert created.question_text == "Where are the runbooks?"
    assert created.scope == CuratedScope.AGENT.value
    assert created.agent_label == "claude-code"
    assert created.owner_id == user.id

    listed = await repository.list_curated_questions(user, "claude-code", config=_config())
    assert [row.id for row in listed] == [created.id]


@pytest.mark.asyncio
async def test_casefold_exact_duplicate_is_refused(curated_engine):
    """Same question, different casing and spacing, is the same question.

    Uses the same key as dedup's collapse, so the CRUD and the pipeline agree on
    what a duplicate is.
    """
    user = _user()
    await _create(user, RUNBOOKS)

    with pytest.raises(DuplicateCuratedQuestionError):
        await _create(user, "  where  ARE the RUNBOOKS?  ")

    # A genuinely different question is still accepted.
    other = await _create(user, CREDENTIALS)
    assert other.question_text == CREDENTIALS


@pytest.mark.asyncio
async def test_duplicate_guard_is_per_scope(curated_engine):
    """One question can exist once per (owner, scope, agent_label) bucket.

    A benchmark question and a Claude-Code-only question may legitimately read
    identically; only a repeat inside one bucket is a duplicate.
    """
    user = _user()

    await _create(user, RUNBOOKS, agent_label="claude-code")
    await _create(user, RUNBOOKS, agent_label="codex")
    await _create(user, RUNBOOKS, scope=CuratedScope.SHARED.value)

    listed = await repository.list_curated_questions(user, config=_config())
    assert len(listed) == 3


@pytest.mark.asyncio
async def test_shared_scope_is_owned_by_the_tenant_when_set(curated_engine):
    tenant_id = uuid4()
    user = _user(tenant_id=tenant_id)

    created = await _create(user, RUNBOOKS, scope=CuratedScope.SHARED.value)

    assert created.owner_id == tenant_id
    assert created.agent_label is None
    assert created.is_shared


@pytest.mark.asyncio
async def test_shared_scope_falls_back_to_the_user_without_a_tenant(curated_engine):
    """``create_user`` never populates ``User.tenant_id``, so this is the common path."""
    user = _user()

    created = await _create(user, RUNBOOKS, scope=CuratedScope.SHARED.value)

    assert created.owner_id == user.id


@pytest.mark.asyncio
async def test_agent_scope_stays_with_the_caller_even_inside_a_tenant(curated_engine):
    """Only the benchmark set is tenant-wide; an agent-scoped row is one person's."""
    user = _user(tenant_id=uuid4())

    created = await _create(user, RUNBOOKS)

    assert created.owner_id == user.id


def test_agent_scope_requires_a_label():
    with pytest.raises(InvalidCuratedQuestionScopeError):
        repository.normalize_curated_scope(CuratedScope.AGENT.value, None, config=_config())


def test_shared_scope_forbids_a_label():
    """A labelled benchmark question would silently apply to one tool only."""
    with pytest.raises(InvalidCuratedQuestionScopeError):
        repository.normalize_curated_scope(
            CuratedScope.SHARED.value, "claude-code", config=_config()
        )


def test_unknown_scope_is_refused():
    with pytest.raises(InvalidCuratedQuestionScopeError):
        repository.normalize_curated_scope("global", "claude-code", config=_config())


def test_unknown_agent_label_is_refused_by_the_one_label_authority():
    """Same 404 a run gets: ``resolve_agent_scope`` validates labels everywhere."""
    with pytest.raises(UnknownAgentLabelError):
        repository.normalize_curated_scope(
            CuratedScope.AGENT.value, "claude-codex", config=_config()
        )


def test_known_label_normalizes_to_itself():
    scope, label = repository.normalize_curated_scope(
        CuratedScope.AGENT.value, "  claude-code  ", config=_config()
    )
    assert (scope, label) == (CuratedScope.AGENT.value, "claude-code")


@pytest.mark.asyncio
async def test_blank_question_text_is_refused(curated_engine):
    user = _user()

    with pytest.raises(EmptyCuratedQuestionError):
        await _create(user, "   ")


@pytest.mark.asyncio
async def test_list_shows_the_label_plus_shared_but_not_another_label(curated_engine):
    user = _user()

    mine = await _create(user, RUNBOOKS, agent_label="claude-code")
    theirs = await _create(user, CREDENTIALS, agent_label="codex")
    shared = await _create(user, "What is the escalation path?", scope=CuratedScope.SHARED.value)

    listed = await repository.list_curated_questions(user, "claude-code", config=_config())

    assert {row.id for row in listed} == {mine.id, shared.id}
    assert theirs.id not in {row.id for row in listed}


@pytest.mark.asyncio
async def test_a_tenant_peer_reads_and_deletes_the_shared_benchmark_row(curated_engine):
    """The benchmark set is the organisation's, so a teammate can maintain it."""
    tenant_id = uuid4()
    author = _user(tenant_id=tenant_id)
    peer = _user(tenant_id=tenant_id)

    shared = await _create(author, RUNBOOKS, scope=CuratedScope.SHARED.value)

    peer_view = await repository.list_curated_questions(peer, config=_config())
    assert [row.id for row in peer_view] == [shared.id]

    await repository.delete_curated_question(peer, shared.id)
    assert await repository.list_curated_questions(peer, config=_config()) == []


@pytest.mark.asyncio
async def test_deleting_another_owners_row_raises_not_found(curated_engine):
    """404, never 403: a 403 would confirm the id exists under another owner."""
    author = _user()
    stranger = _user()

    row = await _create(author, RUNBOOKS)

    with pytest.raises(CuratedQuestionNotFoundError):
        await repository.delete_curated_question(stranger, row.id)

    # Still there for its owner.
    assert [item.id for item in await repository.list_curated_questions(author, config=_config())]


@pytest.mark.asyncio
async def test_deleting_a_missing_row_raises_not_found(curated_engine):
    with pytest.raises(CuratedQuestionNotFoundError):
        await repository.delete_curated_question(_user(), uuid4())


@pytest.mark.asyncio
async def test_load_for_scope_takes_the_agent_scope_not_a_label_string(curated_engine):
    user = _user()

    mine = await _create(user, RUNBOOKS, agent_label="claude-code")
    await _create(user, CREDENTIALS, agent_label="codex")
    shared = await _create(user, "Who owns billing?", scope=CuratedScope.SHARED.value)

    loaded = await repository.load_curated_questions_for_scope(user, _scope("claude-code"))

    assert {row.id for row in loaded} == {mine.id, shared.id}


# --- Replication into dataset partitions ------------------------------------


def test_curated_asks_replicate_into_every_readable_dataset_partition():
    """One curated question, N readable datasets, N asks — one per partition."""
    user_id = uuid4()
    datasets = [uuid4(), uuid4(), uuid4()]
    question = repository.CuratedQuestion(
        id=uuid4(),
        owner_id=user_id,
        scope=CuratedScope.SHARED.value,
        agent_label=None,
        question_text=CURATED_RUNBOOKS,
    )

    asks = repository.curated_asks(
        [question], user_id=user_id, dataset_ids=datasets + [datasets[0]]
    )

    # The repeated dataset id is not a second partition.
    assert [ask.dataset_id for ask in asks] == datasets
    assert {ask.partition_key for ask in asks} == {(user_id, dataset) for dataset in datasets}
    for ask in asks:
        assert ask.source == QuestionSource.CURATED.value
        assert ask.is_observed is False
        assert ask.first_seen is None and ask.last_seen is None
        assert ask.query_ids == []
        assert ask.curated_question_id == question.id


def test_curated_asks_with_no_readable_datasets_land_in_the_null_partition():
    """A user with no datasets must still get the question judged, not dropped."""
    user_id = uuid4()
    question = repository.CuratedQuestion(
        id=uuid4(),
        owner_id=user_id,
        scope=CuratedScope.AGENT.value,
        agent_label="claude-code",
        question_text=CURATED_RUNBOOKS,
    )

    asks = repository.curated_asks([question], user_id=user_id, dataset_ids=[])

    assert len(asks) == 1
    assert asks[0].dataset_id is None


@pytest.mark.asyncio
async def test_load_curated_asks_replicates_over_the_callers_readable_datasets(
    curated_engine, monkeypatch
):
    user = _user()
    readable = [uuid4(), uuid4()]

    async def _authorized_datasets(datasets=None, permission_type=None, user=None):
        # Read permission over every dataset the caller can read (no id filter).
        assert permission_type == "read"
        assert datasets is None
        return [SimpleNamespace(id=dataset_id) for dataset_id in readable]

    monkeypatch.setattr(repository, "get_authorized_existing_datasets", _authorized_datasets)

    await _create(user, CURATED_RUNBOOKS, agent_label="claude-code")
    await _create(user, CREDENTIALS, scope=CuratedScope.SHARED.value)

    asks = await repository.load_curated_asks(user, _scope("claude-code"))

    # Two questions x two datasets, all attributed to the caller.
    assert len(asks) == 4
    assert {ask.user_id for ask in asks} == {user.id}
    assert {ask.dataset_id for ask in asks} == set(readable)


@pytest.mark.asyncio
async def test_load_curated_asks_is_empty_without_curated_questions(curated_engine, monkeypatch):
    """No curated rows means no dataset resolution at all."""

    async def _never(**_kwargs):
        raise AssertionError("datasets must not be resolved when there is nothing to replicate")

    monkeypatch.setattr(repository, "get_authorized_existing_datasets", _never)

    assert await repository.load_curated_asks(_user(), _scope("claude-code")) == []


# --- Integration with dedup -------------------------------------------------


def test_curated_question_merging_traffic_keeps_its_text_and_inherits_the_ask_count():
    """The human's wording wins, and the row reports the partition's real demand.

    Three separate asks of the same thing against dataset A, one curated question
    replicated into A and B. In A the curated ask merges: canonical text is the
    curated text (a human wrote it, so it is the better label), ``was_asked`` is
    true, and ``occurrence_count`` is the three *observed* asks — the curated
    member contributes nothing, because a human adding a question is not demand.
    """
    user_id = uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    # Three days apart, so the retry cooldown does not swallow them.
    observed = collapse_asks(
        [
            _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_a, created_at=BASE_TIME - offset)
            for offset in (timedelta(0), timedelta(days=1), timedelta(days=2))
        ],
        fanout_window_seconds=5,
        retry_cooldown_seconds=300,
        max_questions=150,
    )
    assert observed.distinct_ask_count == 3

    question = repository.CuratedQuestion(
        id=uuid4(),
        owner_id=user_id,
        scope=CuratedScope.SHARED.value,
        agent_label=None,
        question_text=CURATED_RUNBOOKS,
    )
    asks = observed.asks + repository.curated_asks(
        [question], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)
    by_dataset = {row.dataset_id: row for row in result.questions}

    merged = by_dataset[dataset_a]
    assert merged.text == CURATED_RUNBOOKS
    assert merged.source == QuestionSource.CURATED.value
    assert merged.curated_question_id == question.id
    assert merged.was_asked is True
    assert merged.occurrence_count == 3
    assert merged.first_asked_at == BASE_TIME - timedelta(days=2)
    assert merged.last_asked_at == BASE_TIME


def test_curated_question_matching_nothing_stands_alone_with_zero_occurrences():
    """No traffic in that partition means ``occurrence_count = 0``, hence impact 0.

    ``impact = occurrence_count * (judge_score_max - judge_score)``, so a row
    nobody asked can never claim priority however badly it scores.
    """
    user_id = uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    observed = collapse_asks(
        [_row(RUNBOOKS, user_id=user_id, dataset_id=dataset_a, created_at=BASE_TIME)],
        fanout_window_seconds=5,
        retry_cooldown_seconds=300,
        max_questions=150,
    )
    question = repository.CuratedQuestion(
        id=uuid4(),
        owner_id=user_id,
        scope=CuratedScope.AGENT.value,
        agent_label="claude-code",
        question_text=CURATED_RUNBOOKS,
    )
    asks = observed.asks + repository.curated_asks(
        [question], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)
    lonely = {row.dataset_id: row for row in result.questions}[dataset_b]

    assert lonely.source == QuestionSource.CURATED.value
    assert lonely.was_asked is False
    assert lonely.occurrence_count == 0
    assert lonely.first_asked_at is None and lonely.last_asked_at is None
    assert lonely.occurrence_count * (5 - 0) == 0


def test_curated_asks_do_not_count_against_max_questions():
    """Truncation applies to observed traffic only; curated rows are appended after.

    Otherwise adding benchmark questions would silently evict the newest real
    traffic from the run — the report would get quieter the more you curated.
    """
    user_id = uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    observed = collapse_asks(
        [
            _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_a, created_at=BASE_TIME),
            _row(CREDENTIALS, user_id=user_id, dataset_id=dataset_b, created_at=BASE_TIME),
        ],
        fanout_window_seconds=5,
        retry_cooldown_seconds=300,
        max_questions=1,
    )
    assert len(observed.asks) == 1
    assert observed.dropped_ask_count == 1

    question = repository.CuratedQuestion(
        id=uuid4(),
        owner_id=user_id,
        scope=CuratedScope.SHARED.value,
        agent_label=None,
        question_text=CURATED_RUNBOOKS,
    )
    asks = observed.asks + repository.curated_asks(
        [question], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    assert len(asks) == 3
    assert sum(1 for ask in asks if not ask.is_observed) == 2
