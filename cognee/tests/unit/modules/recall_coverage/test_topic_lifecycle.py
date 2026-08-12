"""Guards on the topic lifecycle — spec sections 3 and 5 (routes 6, 7, 8).

The lifecycle is **accept, delete, dismiss and nothing else**. What this file
pins down is why that is a design and not an omission:

* **Accept mints the topic id**, and nothing else ever does. A suggestion carries
  no id, so the id comes into existence on a human decision and then never moves —
  which is what lets ``topics[].avg_score`` be a trend across runs.
* **``taxonomy_version`` is monotone per owner.** Both accept and delete bump it
  and stamp the bumped value on the row they touched, and the maximum is taken
  over deleted topics too. If a delete could lower it, two different taxonomies
  would both call themselves version 4 and every historical comparison would be
  wrong.
* **Delete is soft, and idempotent.** The row survives, so a historical run's
  ``topic_id`` still resolves; its questions are never deleted and simply fall
  back to the sink on the next run, because ``load_active_topics`` stops
  returning the topic.
* **No rename, no merge, no split.** Asserted as an absence: the module must not
  grow a function for any of them, since all three change what an existing id
  means while looking like an edit.
* **Owner scope on every id-keyed call, 404 and never 403.** A 403 would confirm
  that someone else's topic with that id exists.

SQLite over ``tmp_path`` holding only the two tables under test. No LLM, no
embedding engine, no network.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.exceptions import (
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    SinkTopicNotEditableError,
)
from cognee.modules.recall_coverage.models import (
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.types import SINK_TOPIC_ID, SuggestionStatus

repository = importlib.import_module("cognee.modules.recall_coverage.repository")

MODEL = "openai/text-embedding-3-large"
DIMENSIONS = 3


@pytest_asyncio.fixture
async def topics_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the topic and suggestion tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recall_coverage_topics_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[RecallCoverageTopic.__table__, RecallCoverageTopicSuggestion.__table__],
        )

    monkeypatch.setattr(repository, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


def _draft(owner_id, label="Billing & invoices", *, centroid=(1.0, 0.0, 0.0), agent_label="codex"):
    return repository.SuggestionDraft(
        owner_id=owner_id,
        label=label,
        centroid=centroid,
        embedding_model=MODEL,
        embedding_dimensions=DIMENSIONS,
        question_count=7,
        cohesion=0.91,
        agent_label=agent_label,
        run_id=uuid4(),
    )


async def _suggest(owner_id, **kwargs):
    written = await repository.create_topic_suggestions([_draft(owner_id, **kwargs)])
    return written[0]


# --- accept ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_mints_the_topic_and_copies_the_cluster_verbatim(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label="Billing & invoices")

    topic, decided = await repository.accept_topic_suggestion(suggestion.id, [owner_id])

    # The id is minted here and nowhere else.
    assert topic.id is not None
    assert topic.owner_id == owner_id
    assert topic.label == "Billing & invoices"
    # Copied, not recomputed: re-embedding the label would put the topic
    # somewhere other than the cluster that motivated it.
    assert topic.centroid == (1.0, 0.0, 0.0)
    assert topic.embedding_model == MODEL
    assert topic.embedding_dimensions == DIMENSIONS
    assert topic.seed_question_count == 7
    assert not topic.is_deleted

    assert decided.status == SuggestionStatus.ACCEPTED.value
    assert decided.accepted_topic_id == topic.id


@pytest.mark.asyncio
async def test_accept_bumps_the_taxonomy_version_from_zero(topics_engine):
    owner_id = uuid4()
    assert await repository.current_taxonomy_version(owner_id) == 0

    first, _ = await repository.accept_topic_suggestion(
        (await _suggest(owner_id, label="Billing")).id, [owner_id]
    )
    second, _ = await repository.accept_topic_suggestion(
        (await _suggest(owner_id, label="Runbooks")).id, [owner_id]
    )

    assert first.taxonomy_version == 1
    assert second.taxonomy_version == 2
    assert await repository.current_taxonomy_version(owner_id) == 2


@pytest.mark.asyncio
async def test_the_accepted_topic_is_immediately_active_for_the_next_run(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)

    topic, _ = await repository.accept_topic_suggestion(suggestion.id, [owner_id])

    active = await repository.load_active_topics(owner_id)
    assert [record.id for record in active] == [topic.id]


@pytest.mark.asyncio
async def test_accepting_twice_is_refused_rather_than_minting_a_second_id(topics_engine):
    """Two ids for one theme would split its score trend in half."""
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)

    await repository.accept_topic_suggestion(suggestion.id, [owner_id])

    with pytest.raises(CoverageSuggestionNotPendingError):
        await repository.accept_topic_suggestion(suggestion.id, [owner_id])

    assert len(await repository.load_active_topics(owner_id)) == 1


@pytest.mark.asyncio
async def test_accept_404s_for_another_owner_rather_than_403(topics_engine):
    owner_id = uuid4()
    other_owner = uuid4()
    suggestion = await _suggest(owner_id)

    with pytest.raises(CoverageSuggestionNotFoundError):
        await repository.accept_topic_suggestion(suggestion.id, [other_owner])

    with pytest.raises(CoverageSuggestionNotFoundError):
        await repository.accept_topic_suggestion(uuid4(), [owner_id])

    # Nothing was minted for either owner.
    assert await repository.load_active_topics(owner_id) == []
    assert await repository.load_active_topics(other_owner) == []


@pytest.mark.asyncio
async def test_a_tenant_peer_accepts_into_the_suggestions_owner_scope(topics_engine):
    """Accepting must not fork the taxonomy into a scope nothing is scored against."""
    tenant_id = uuid4()
    peer_id = uuid4()
    suggestion = await _suggest(tenant_id)

    topic, _ = await repository.accept_topic_suggestion(suggestion.id, [peer_id, tenant_id])

    assert topic.owner_id == tenant_id
    assert [record.id for record in await repository.load_active_topics(tenant_id)] == [topic.id]
    assert await repository.load_active_topics(peer_id) == []


# --- dismiss -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_dismiss_settles_the_suggestion_without_touching_the_taxonomy(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)

    dismissed = await repository.dismiss_topic_suggestion(suggestion.id, [owner_id])

    assert dismissed.status == SuggestionStatus.DISMISSED.value
    assert dismissed.accepted_topic_id is None
    # Nothing about the taxonomy changed, so runs before and after this call were
    # scored against the same topics.
    assert await repository.current_taxonomy_version(owner_id) == 0
    assert await repository.load_active_topics(owner_id) == []


@pytest.mark.asyncio
async def test_a_dismissed_suggestion_keeps_suppressing_the_theme(topics_engine):
    """The row is the record of the decision — the re-proposal guard reads it."""
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)

    await repository.dismiss_topic_suggestion(suggestion.id, [owner_id])

    settled = await repository.load_settled_suggestions(owner_id)
    assert [record.id for record in settled] == [suggestion.id]
    assert await repository.load_pending_suggestions(owner_id) == []


@pytest.mark.asyncio
async def test_dismissing_a_decided_suggestion_is_a_conflict(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)
    await repository.dismiss_topic_suggestion(suggestion.id, [owner_id])

    with pytest.raises(CoverageSuggestionNotPendingError):
        await repository.dismiss_topic_suggestion(suggestion.id, [owner_id])

    accepted = await _suggest(owner_id, label="Runbooks")
    await repository.accept_topic_suggestion(accepted.id, [owner_id])
    with pytest.raises(CoverageSuggestionNotPendingError):
        await repository.dismiss_topic_suggestion(accepted.id, [owner_id])


@pytest.mark.asyncio
async def test_dismiss_404s_for_another_owner(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id)

    with pytest.raises(CoverageSuggestionNotFoundError):
        await repository.dismiss_topic_suggestion(suggestion.id, [uuid4()])

    assert [record.id for record in await repository.load_pending_suggestions(owner_id)] == [
        suggestion.id
    ]


# --- delete ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_is_soft_and_bumps_the_version(topics_engine):
    owner_id = uuid4()
    topic, _ = await repository.accept_topic_suggestion((await _suggest(owner_id)).id, [owner_id])
    assert topic.taxonomy_version == 1

    version = await repository.delete_topic(topic.id, [owner_id])

    assert version == 2
    # Gone from the run's input...
    assert await repository.load_active_topics(owner_id) == []
    # ...but the row survives, so a historical run's topic_id still resolves.
    stored = await repository.list_topics([owner_id], include_deleted=True)
    assert [record.id for record in stored] == [topic.id]
    assert stored[0].is_deleted
    assert stored[0].deleted_at is not None


@pytest.mark.asyncio
async def test_the_version_never_goes_backwards_across_deletes_and_accepts(topics_engine):
    """The monotonicity invariant, exercised as an interleaved sequence."""
    owner_id = uuid4()
    versions: list[int] = []

    first, _ = await repository.accept_topic_suggestion((await _suggest(owner_id)).id, [owner_id])
    versions.append(first.taxonomy_version)

    versions.append(await repository.delete_topic(first.id, [owner_id]))

    second, _ = await repository.accept_topic_suggestion(
        (await _suggest(owner_id, label="Runbooks")).id, [owner_id]
    )
    versions.append(second.taxonomy_version)

    versions.append(await repository.delete_topic(second.id, [owner_id]))

    assert versions == [1, 2, 3, 4]
    assert versions == sorted(versions)
    # The highest version is still readable with every topic deleted, which is
    # exactly what the soft delete is for.
    assert await repository.current_taxonomy_version(owner_id) == 4
    assert await repository.load_active_topics(owner_id) == []


@pytest.mark.asyncio
async def test_deleting_twice_does_not_inflate_the_version(topics_engine):
    owner_id = uuid4()
    topic, _ = await repository.accept_topic_suggestion((await _suggest(owner_id)).id, [owner_id])

    first = await repository.delete_topic(topic.id, [owner_id])
    second = await repository.delete_topic(topic.id, [owner_id])

    assert first == second == 2


@pytest.mark.asyncio
async def test_delete_404s_for_another_owner_rather_than_403(topics_engine):
    owner_id = uuid4()
    other_owner = uuid4()
    topic, _ = await repository.accept_topic_suggestion((await _suggest(owner_id)).id, [owner_id])

    with pytest.raises(CoverageTopicNotFoundError):
        await repository.delete_topic(topic.id, [other_owner])

    with pytest.raises(CoverageTopicNotFoundError):
        await repository.delete_topic(uuid4(), [owner_id])

    assert len(await repository.load_active_topics(owner_id)) == 1


# --- the sink is not a topic --------------------------------------------------


def test_the_sink_id_is_not_deletable():
    """``"other"`` is a wire literal, not a row: 422, not 404."""
    with pytest.raises(SinkTopicNotEditableError):
        repository.parse_topic_id(SINK_TOPIC_ID)

    with pytest.raises(SinkTopicNotEditableError):
        repository.parse_topic_id("Other")


def test_an_unparseable_topic_id_is_a_not_found():
    with pytest.raises(CoverageTopicNotFoundError):
        repository.parse_topic_id("not-a-uuid")

    with pytest.raises(CoverageTopicNotFoundError):
        repository.parse_topic_id("")


def test_parse_topic_id_passes_a_uuid_through():
    topic_id = uuid4()
    assert repository.parse_topic_id(topic_id) == topic_id
    assert repository.parse_topic_id(str(topic_id)) == topic_id


# --- listing ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_topics_hides_deleted_rows_unless_asked_and_is_owner_scoped(topics_engine):
    owner_id = uuid4()
    other_owner = uuid4()

    kept, _ = await repository.accept_topic_suggestion(
        (await _suggest(owner_id, label="Billing")).id, [owner_id]
    )
    removed, _ = await repository.accept_topic_suggestion(
        (await _suggest(owner_id, label="Runbooks")).id, [owner_id]
    )
    await repository.accept_topic_suggestion(
        (await _suggest(other_owner, label="Someone else's")).id, [other_owner]
    )
    await repository.delete_topic(removed.id, [owner_id])

    visible = await repository.list_topics([owner_id])
    assert [record.id for record in visible] == [kept.id]

    everything = await repository.list_topics([owner_id], include_deleted=True)
    assert {record.id for record in everything} == {kept.id, removed.id}
    # Oldest first, the same order the assignment matmul used.
    assert [record.label for record in everything] == ["Billing", "Runbooks"]


# --- the operations that deliberately do not exist ----------------------------


def test_there_is_no_rename_merge_or_split():
    """All three change what an existing topic id means while looking like an edit."""
    forbidden = [
        name
        for name in dir(repository)
        if any(word in name.lower() for word in ("rename", "merge", "split"))
    ]
    assert forbidden == []
