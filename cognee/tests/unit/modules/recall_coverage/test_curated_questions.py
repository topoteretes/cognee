"""Guards on user-defined recall-coverage questions.

A user-defined question is one a human typed that memory *should* answer, whether
or not any agent asked it. Four invariants, each of which decides whether a
reported number means anything:

* **One flat list per owner.** No shared-versus-per-agent split: a question is
  worth writing down once rather than once per tool, and every run for that owner
  replays every one of them. The list is keyed on ``user.id`` on create, read and
  delete, because what a person wants memory to answer is not the tenant's
  property and not a property of the tool that asks.
* **The duplicate guard is casefold-exact**, and it uses the same key as dedup's
  collapse, so the CRUD and the pipeline agree on what "the same question" is.
* **Replication.** A user-defined question has no dataset, so it enters one
  partition per readable dataset — never a special empty-dataset case, and never
  dropped when the user has no datasets at all.
* **Merging.** Where such a question lands on top of real traffic it keeps the
  human's wording and inherits that partition's distinct-ask count; where it
  matches nothing it stands alone with ``relevance = 0``, which is exactly what
  keeps it out of every average.

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
    CuratedQuestionLimitError,
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
    EmptyCuratedQuestionError,
)
from cognee.modules.recall_coverage.models import RecallCoverageCuratedQuestion
from cognee.modules.recall_coverage.types import QuestionSource
from cognee.modules.search.operations.get_queries import QueryWindowRow
from cognee.modules.users.models import User
from cognee.modules.users.models.Principal import Principal

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


def _config(**overrides) -> RecallCoverageConfig:
    """A config that ignores the developer's ``.env``."""
    return RecallCoverageConfig(_env_file=None, **overrides)


def _user(tenant_id=None) -> SimpleNamespace:
    """The minimum ``get_authenticated_user`` hands a route."""
    return SimpleNamespace(id=uuid4(), tenant_id=tenant_id)


async def _create(user, text, **overrides):
    return await repository.create_curated_question(user, text, config=_config(**overrides))


def _question(user_id, text=CURATED_RUNBOOKS) -> "repository.CuratedQuestion":
    return repository.CuratedQuestion(id=uuid4(), owner_id=user_id, question=text)


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
    """A SQLite engine holding only the user-defined questions table."""
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
            Base.metadata.create_all,
            tables=[
                RecallCoverageCuratedQuestion.__table__,
                # For the visible_user_ids tests below. User is joined-table
                # inheritance, so its select needs principals too.
                Principal.__table__,
                User.__table__,
            ],
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
    assert created.question == "Where are the runbooks?"
    assert created.owner_id == user.id

    listed = await repository.list_curated_questions(user)
    assert [row.id for row in listed] == [created.id]


@pytest.mark.asyncio
async def test_casefold_exact_duplicate_is_refused(curated_engine):
    """Same question, different casing and spacing, is the same question.

    Uses the same key as dedup's collapse, so the CRUD and the pipeline agree on
    what a duplicate is. Refused rather than merged so the writer learns the
    question is already covered.
    """
    user = _user()
    await _create(user, RUNBOOKS)

    with pytest.raises(DuplicateCuratedQuestionError):
        await _create(user, "  where  ARE the RUNBOOKS?  ")

    # A genuinely different question is still accepted.
    other = await _create(user, CREDENTIALS)
    assert other.question == CREDENTIALS


@pytest.mark.asyncio
async def test_two_people_may_each_write_down_the_same_question(curated_engine):
    """The list is one person's, so the duplicate guard is per owner and no wider.

    Coverage is answered out of a specific user's readable brains, so two teammates
    asking the same thing are two rows with two independent answers.
    """
    anna, ben = _user(), _user()

    mine = await _create(anna, RUNBOOKS)
    theirs = await _create(ben, RUNBOOKS)

    assert mine.id != theirs.id
    assert [row.id for row in await repository.list_curated_questions(anna)] == [mine.id]
    assert [row.id for row in await repository.list_curated_questions(ben)] == [theirs.id]


@pytest.mark.asyncio
async def test_the_list_is_the_callers_own_even_inside_a_tenant(curated_engine):
    """Deliberately *not* the tenant scope runs and topics use.

    Runs and topics are shared so two agents' scores are comparable; what a human
    wants memory to answer is theirs.
    """
    tenant_id = uuid4()
    author = _user(tenant_id=tenant_id)
    peer = _user(tenant_id=tenant_id)

    created = await _create(author, RUNBOOKS)

    assert created.owner_id == author.id
    assert await repository.list_curated_questions(peer) == []


@pytest.mark.asyncio
async def test_creation_is_refused_past_the_configured_cap(curated_engine):
    """The list multiplies every future run's replay and judge cost.

    Each question is replicated into one row per readable dataset, and each row is
    a replay plus a judge call plus an answer completion — a cost ``max_questions``
    does not bound.
    """
    user = _user()

    await _create(user, RUNBOOKS, max_curated_questions=2)
    await _create(user, CREDENTIALS, max_curated_questions=2)

    with pytest.raises(CuratedQuestionLimitError):
        await _create(user, "What is the escalation path?", max_curated_questions=2)

    # One bucket per owner, so raising the cap is the only way through.
    created = await _create(user, "What is the escalation path?", max_curated_questions=3)
    assert created.owner_id == user.id


@pytest.mark.asyncio
async def test_a_blank_question_is_refused(curated_engine):
    user = _user()

    with pytest.raises(EmptyCuratedQuestionError):
        await _create(user, "   ")


@pytest.mark.asyncio
async def test_the_list_is_newest_first(curated_engine):
    user = _user()

    first = await _create(user, RUNBOOKS)
    second = await _create(user, CREDENTIALS)

    async with curated_engine.get_async_session() as session:
        row = await session.get(RecallCoverageCuratedQuestion, first.id)
        row.created_at = BASE_TIME - timedelta(days=1)
        await session.commit()

    listed = await repository.list_curated_questions(user)
    assert [row.id for row in listed] == [second.id, first.id]


@pytest.mark.asyncio
async def test_deleting_another_owners_row_raises_not_found(curated_engine):
    """404, never 403: a 403 would confirm the id exists under another owner."""
    author = _user()
    stranger = _user()

    row = await _create(author, RUNBOOKS)

    with pytest.raises(CuratedQuestionNotFoundError):
        await repository.delete_curated_question(stranger, row.id)

    # Still there for its owner.
    assert [item.id for item in await repository.list_curated_questions(author)] == [row.id]


@pytest.mark.asyncio
async def test_deleting_a_missing_row_raises_not_found(curated_engine):
    with pytest.raises(CuratedQuestionNotFoundError):
        await repository.delete_curated_question(_user(), uuid4())


@pytest.mark.asyncio
async def test_delete_removes_the_row_from_the_owners_list(curated_engine):
    user = _user()
    row = await _create(user, RUNBOOKS)

    await repository.delete_curated_question(user, row.id)

    assert await repository.list_curated_questions(user) == []


# --- visible_user_ids: the boundary of a run ---------------------------------


async def _insert_user(engine, user_id, tenant_id):
    async with engine.engine.begin() as connection:
        await connection.execute(
            Principal.__table__.insert().values(id=user_id, type="user", created_at=BASE_TIME)
        )
        await connection.execute(
            User.__table__.insert().values(
                id=user_id,
                tenant_id=tenant_id,
                email=f"{user_id}@example.com",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=False,
            )
        )


@pytest.mark.asyncio
async def test_visible_user_ids_is_the_caller_alone_without_a_tenant(curated_engine):
    """A tenantless caller analyses only themselves — no users-table read at all."""
    user = _user()

    assert await repository.visible_user_ids(user) == (user.id,)


@pytest.mark.asyncio
async def test_visible_user_ids_is_the_tenant_membership(curated_engine):
    """A tenant caller sees the tenant's members and nobody else.

    The relational database is shared in OSS deployments, so this tuple — not
    the database — is what keeps one tenant's coverage run out of another
    tenant's question text.
    """
    tenant, other_tenant = uuid4(), uuid4()
    anna, ben, stranger = uuid4(), uuid4(), uuid4()
    await _insert_user(curated_engine, anna, tenant)
    await _insert_user(curated_engine, ben, tenant)
    await _insert_user(curated_engine, stranger, other_tenant)

    visible = await repository.visible_user_ids(SimpleNamespace(id=anna, tenant_id=tenant))

    assert set(visible) == {anna, ben}


# --- Replication into dataset partitions ------------------------------------


def test_curated_asks_replicate_into_every_readable_dataset_partition():
    """One question, N readable datasets, N asks — one per partition."""
    user_id = uuid4()
    datasets = [uuid4(), uuid4(), uuid4()]
    question = _question(user_id)

    asks = repository.curated_asks(
        [question], user_id=user_id, dataset_ids=datasets + [datasets[0]]
    )

    # The repeated dataset id is not a second partition.
    assert [ask.dataset_id for ask in asks] == datasets
    assert {ask.partition_key for ask in asks} == {(user_id, dataset) for dataset in datasets}
    for ask in asks:
        assert ask.source == QuestionSource.USER_DEFINED.value
        assert ask.is_observed is False
        assert ask.first_seen is None and ask.last_seen is None
        assert ask.query_ids == []
        assert ask.curated_question_id == question.id
        # No session, so a row nobody asked reports agent: null.
        assert ask.session_id is None


def test_curated_asks_with_no_readable_datasets_land_in_the_null_partition():
    """A user with no datasets must still get the question judged, not dropped."""
    user_id = uuid4()

    asks = repository.curated_asks([_question(user_id)], user_id=user_id, dataset_ids=[])

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

    await _create(user, CURATED_RUNBOOKS)
    await _create(user, CREDENTIALS)

    asks = await repository.load_curated_asks(user)

    # Two questions x two datasets, all attributed to the caller.
    assert len(asks) == 4
    assert {ask.user_id for ask in asks} == {user.id}
    assert {ask.dataset_id for ask in asks} == set(readable)


@pytest.mark.asyncio
async def test_load_curated_asks_takes_no_agent_scope(curated_engine):
    """Every run for an owner replays the whole list, whichever agent it covers.

    Narrowing by label is what the per-agent split used to do; the questions say
    what memory ought to be able to answer, which is not a property of the tool
    that asks. ``dataset_ids`` is accepted so the pipeline, which resolved them for
    their names anyway, does not resolve them twice.
    """
    user = _user()
    dataset_id = uuid4()

    first = await _create(user, RUNBOOKS)
    second = await _create(user, CREDENTIALS)

    asks = await repository.load_curated_asks(user, dataset_ids=[dataset_id])

    assert {ask.curated_question_id for ask in asks} == {first.id, second.id}
    assert {ask.dataset_id for ask in asks} == {dataset_id}


@pytest.mark.asyncio
async def test_load_curated_asks_is_empty_without_any_questions(curated_engine, monkeypatch):
    """No rows means no dataset resolution at all."""

    async def _never(**_kwargs):
        raise AssertionError("datasets must not be resolved when there is nothing to replicate")

    monkeypatch.setattr(repository, "get_authorized_existing_datasets", _never)

    assert await repository.load_curated_asks(_user()) == []


# --- Integration with dedup -------------------------------------------------


def test_a_question_merging_traffic_keeps_its_text_and_inherits_the_ask_count():
    """The human's wording wins, and the row reports the partition's real demand.

    Three separate asks of the same thing against dataset A, one written-down
    question replicated into A and B. In A the user-defined ask merges: canonical
    text is the human's (they wrote it, so it is the better label) and ``relevance``
    is the three *observed* asks — the user-defined member contributes nothing,
    because a human adding a question is not demand for it.
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

    question = _question(user_id)
    asks = observed.asks + repository.curated_asks(
        [question], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)
    by_dataset = {row.dataset_id: row for row in result.questions}

    merged = by_dataset[dataset_a]
    assert merged.text == CURATED_RUNBOOKS
    assert merged.source == QuestionSource.USER_DEFINED.value
    assert merged.curated_question_id == question.id
    # relevance > 0 is what keeps this row in every mean: it was asked three times
    # before the written-down text existed and is asked three times after.
    assert merged.relevance == 3
    assert merged.first_asked_at == BASE_TIME - timedelta(days=2)
    assert merged.last_asked_at == BASE_TIME


def test_a_question_matching_nothing_stands_alone_with_zero_relevance():
    """No traffic in that partition means ``relevance = 0``, and no average sees it.

    That is the whole reason a user cannot move ``memory_score`` by writing down
    questions: a row nobody has asked is nobody's demand yet.
    """
    user_id = uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    observed = collapse_asks(
        [_row(RUNBOOKS, user_id=user_id, dataset_id=dataset_a, created_at=BASE_TIME)],
        fanout_window_seconds=5,
        retry_cooldown_seconds=300,
        max_questions=150,
    )
    question = _question(user_id)
    asks = observed.asks + repository.curated_asks(
        [question], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)
    lonely = {row.dataset_id: row for row in result.questions}[dataset_b]

    assert lonely.source == QuestionSource.USER_DEFINED.value
    assert lonely.relevance == 0
    assert lonely.first_asked_at is None and lonely.last_asked_at is None
    assert lonely.session_id is None


def test_curated_asks_do_not_count_against_max_questions():
    """Truncation applies to observed traffic only; these rows are appended after.

    Otherwise writing down questions would silently evict the newest real traffic
    from the run — the report would get quieter the more you added.
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

    asks = observed.asks + repository.curated_asks(
        [_question(user_id)], user_id=user_id, dataset_ids=[dataset_a, dataset_b]
    )

    assert len(asks) == 3
    assert sum(1 for ask in asks if not ask.is_observed) == 2
