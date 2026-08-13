"""Guards on the topic lifecycle: create, delete, dismiss and nothing else.

What this file pins down is why that is a design and not an omission:

* **Posting a label mints the topic id**, and nothing else ever does. A suggestion
  carries no id, so the id comes into existence on a human decision and then never
  moves — which is what lets a topic's ``memory_score`` be a trend across runs.
* **One route, two paths.** ``POST /topics`` is also the accept path: a posted
  label close enough to a pending suggestion copies that suggestion's centroid
  **verbatim** and settles it, because re-embedding the label would put the topic
  somewhere other than the cluster that motivated it — and settling it is what
  stops the theme being proposed again. A label matching nothing becomes its own
  centroid.
* **Delete is soft, and idempotent.** The row survives, so a historical run's
  ``topic_id`` still resolves; its questions are never deleted and simply fall
  back to ``Uncategorized`` on the next run, because ``load_active_topics`` stops
  returning the topic. There is no version counter to keep monotone — the soft
  delete exists for exactly the one reason above.
* **No rename, no merge, no split.** Asserted as an absence: the module must not
  grow a function for any of them, since all three change what an existing id
  means while looking like an edit.
* **Owner scope on every id-keyed call, 404 and never 403.** A 403 would confirm
  that someone else's topic with that id exists.

SQLite over ``tmp_path`` holding only the two tables under test. The embedding
engine is a hand-written fake, so every similarity here is exact; no LLM, no
network.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.dedup import collapse_text_key
from cognee.modules.recall_coverage.exceptions import (
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    DuplicateTopicError,
    EmptyTopicLabelError,
)
from cognee.modules.recall_coverage.models import (
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.types import SuggestionStatus

repository = importlib.import_module("cognee.modules.recall_coverage.repository")
suggest = importlib.import_module("cognee.modules.recall_coverage.suggest")

MODEL = "openai/text-embedding-3-large"
DIMENSIONS = 3

# The label a suggestion carries, and the vector the fake engine embeds it to.
BILLING = "Billing & invoices"
BILLING_VECTOR = [1.0, 0.0, 0.0]
# Same theme, different words: cosine ~0.9995 to BILLING_VECTOR.
BILLING_REWORDED = "Invoices and billing"
UNRELATED = "Deploy rollbacks"

VECTORS: dict[str, list[float]] = {
    collapse_text_key(BILLING): BILLING_VECTOR,
    collapse_text_key(BILLING_REWORDED): [10.0, 0.3, 0.0],
    collapse_text_key(UNRELATED): [0.0, 0.0, 1.0],
}


class _FakeEmbeddingEngine:
    """Deterministic embeddings for the labels used in this module.

    Keyed on the collapse key rather than the exact string, because a posted label
    is embedded *before* the duplicate check runs — so ``"  deploy   ROLLBACKS "``
    reaches the engine as itself and must still resolve to a vector.
    """

    model = MODEL

    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 8

    async def embed_text(self, texts):
        return [list(VECTORS[collapse_text_key(text)]) for text in texts]


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
    monkeypatch.setattr(suggest, "get_embedding_engine", _FakeEmbeddingEngine)

    yield engine

    await engine.engine.dispose()


def _config(**overrides):
    from cognee.modules.recall_coverage.config import RecallCoverageConfig

    return RecallCoverageConfig(_env_file=None, **overrides)


def _draft(owner_id, label=BILLING, *, centroid=(1.0, 0.0, 0.0), agent_label="codex"):
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


async def _post(owner_id, label, **overrides):
    """What ``POST /topics`` calls."""
    return await suggest.create_topic_from_label(owner_id, label, config=_config(**overrides))


# --- creating a topic from a posted label ------------------------------------


@pytest.mark.asyncio
async def test_a_posted_label_mints_a_topic_whose_centroid_is_that_label(topics_engine):
    """With nothing to match, the topic's centroid *is* its name.

    Worth stating out loud: such a topic only attracts questions worded like its
    name until real traffic drifts towards it, which is why it honestly reports 0
    questions to begin with. The remedy is to delete it and accept a suggestion,
    not to widen the threshold.
    """
    owner_id = uuid4()

    topic, accepted = await _post(owner_id, UNRELATED)

    # The id is minted here and nowhere else.
    assert topic.id is not None
    assert topic.owner_id == owner_id
    assert topic.label == UNRELATED
    assert accepted is None
    # The live engine's fingerprint, so the next run can score against it.
    assert topic.embedding_model == MODEL
    assert topic.embedding_dimensions == DIMENSIONS
    assert topic.centroid == pytest.approx((0.0, 0.0, 1.0))
    assert not topic.is_deleted


@pytest.mark.asyncio
async def test_a_minted_topic_is_immediately_active_for_the_next_run(topics_engine):
    owner_id = uuid4()

    topic, _ = await _post(owner_id, UNRELATED)

    active = await repository.load_active_topics(owner_id)
    assert [record.id for record in active] == [topic.id]


@pytest.mark.asyncio
async def test_a_blank_label_is_refused(topics_engine):
    """A blank label is a topic that means nothing and can attract nothing."""
    owner_id = uuid4()

    for blank in ("", "   ", "\n"):
        with pytest.raises(EmptyTopicLabelError):
            await _post(owner_id, blank)

    assert await repository.load_active_topics(owner_id) == []


@pytest.mark.asyncio
async def test_a_duplicate_label_is_refused_rather_than_disabling_the_topic(topics_engine):
    """Two near-identical centroids cannot be separated by the assignment margin.

    Every question about the theme would then land in ``Uncategorized`` instead of
    in either topic — a duplicate label does not add a topic, it silently disables
    one. Casefold-exact and whitespace-collapsed, the same rule dedup uses.
    """
    owner_id = uuid4()
    await _post(owner_id, UNRELATED)

    with pytest.raises(DuplicateTopicError):
        await _post(owner_id, UNRELATED)
    with pytest.raises(DuplicateTopicError):
        await _post(owner_id, "  deploy   ROLLBACKS ")

    assert len(await repository.load_active_topics(owner_id)) == 1
    # Another owner's identical label is untouched: the taxonomy is per owner.
    other_topic, _ = await _post(uuid4(), UNRELATED)
    assert other_topic.label == UNRELATED


@pytest.mark.asyncio
async def test_a_deleted_labels_name_can_be_used_again(topics_engine):
    """The duplicate guard is over *active* topics; a new id visibly starts over."""
    owner_id = uuid4()
    first, _ = await _post(owner_id, UNRELATED)
    await repository.delete_topic(first.id, [owner_id])

    second, _ = await _post(owner_id, UNRELATED)

    assert second.id != first.id


# --- posting a label that matches a pending suggestion ------------------------


@pytest.mark.asyncio
async def test_a_matching_label_accepts_the_suggestion_and_copies_it_verbatim(topics_engine):
    """The accept path. The centroid is the cluster's, not the label's.

    Re-embedding the label would put the topic somewhere other than the cluster
    that motivated it, so the questions that produced the suggestion might not even
    be assigned to the topic the owner just accepted.
    """
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))

    topic, accepted = await _post(owner_id, BILLING)

    assert accepted is not None and accepted.id == suggestion.id
    assert topic.centroid == pytest.approx(tuple(BILLING_VECTOR))
    assert topic.embedding_model == MODEL
    assert topic.embedding_dimensions == DIMENSIONS
    # The label stored is the one the owner posted.
    assert topic.label == BILLING


@pytest.mark.asyncio
async def test_the_accepted_suggestion_is_settled_and_linked(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))

    topic, _ = await _post(owner_id, BILLING)

    settled = await repository.load_settled_suggestions(owner_id)
    assert [record.id for record in settled] == [suggestion.id]
    assert settled[0].status == SuggestionStatus.ACCEPTED.value
    assert settled[0].accepted_topic_id == topic.id
    # And it has left the review queue.
    assert await repository.load_pending_suggestions(owner_id) == []


@pytest.mark.asyncio
async def test_an_accepted_suggestion_stops_being_proposed_next_run(topics_engine):
    """The link is half of one mechanism: the re-proposal guard reads settled rows."""
    from cognee.modules.recall_coverage.embedding import EmbeddingFingerprint, normalize_rows

    owner_id = uuid4()
    await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))
    await _post(owner_id, BILLING)

    clusters = suggest.cluster_sink_questions(
        normalize_rows([[1.0, 0.02 * index, 0.0] for index in range(5)]),
        sink_cluster_threshold=0.80,
        min_questions_per_topic=5,
        max_suggestions_per_run=5,
    )
    kept = suggest.drop_reproposed(
        clusters,
        await repository.load_settled_suggestions(owner_id),
        fingerprint=EmbeddingFingerprint(model=MODEL, dimensions=DIMENSIONS),
        suggestion_dedup_threshold=0.90,
    )

    assert len(clusters) == 1
    assert kept == []


@pytest.mark.asyncio
async def test_a_reworded_label_still_accepts_the_suggestion(topics_engine):
    """The match is on the embedding, not the string: the UI's label may be edited."""
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))

    topic, accepted = await _post(owner_id, BILLING_REWORDED)

    assert accepted is not None and accepted.id == suggestion.id
    assert topic.label == BILLING_REWORDED
    assert topic.centroid == pytest.approx(tuple(BILLING_VECTOR))


@pytest.mark.asyncio
async def test_an_unrelated_label_mints_a_topic_and_leaves_the_queue_alone(topics_engine):
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))

    topic, accepted = await _post(owner_id, UNRELATED)

    assert accepted is None
    assert topic.centroid == pytest.approx((0.0, 0.0, 1.0))
    assert [record.id for record in await repository.load_pending_suggestions(owner_id)] == [
        suggestion.id
    ]


@pytest.mark.asyncio
async def test_a_pending_suggestion_under_a_stale_fingerprint_is_skipped_not_fatal(topics_engine):
    """A stale suggestion only weakens a match; it must not fail the request.

    Unlike a topic centroid — which is scored against, and therefore fails the run —
    the visible consequence here is a fresh topic instead of an accepted suggestion.
    """
    owner_id = uuid4()

    async with topics_engine.get_async_session() as session:
        session.add(
            RecallCoverageTopicSuggestion(
                owner_id=owner_id,
                label=BILLING,
                centroid=list(BILLING_VECTOR),
                embedding_model="openai/text-embedding-ada-002",
                embedding_dimensions=DIMENSIONS,
                question_count=5,
                cohesion=0.9,
                status=SuggestionStatus.PENDING.value,
            )
        )
        await session.commit()

    topic, accepted = await _post(owner_id, BILLING)

    assert accepted is None
    # The label became the centroid, under the live engine's fingerprint.
    assert topic.embedding_model == MODEL
    assert len(await repository.load_pending_suggestions(owner_id)) == 1


@pytest.mark.asyncio
async def test_another_owners_pending_suggestion_is_never_accepted(topics_engine):
    """The suggestion queue is owner-scoped, so a posted label cannot reach into it."""
    owner_id = uuid4()
    other_owner = uuid4()
    suggestion = await _suggest(other_owner, label=BILLING, centroid=tuple(BILLING_VECTOR))

    topic, accepted = await _post(owner_id, BILLING)

    assert accepted is None
    assert topic.owner_id == owner_id
    assert [record.id for record in await repository.load_pending_suggestions(other_owner)] == [
        suggestion.id
    ]


@pytest.mark.asyncio
async def test_accepting_an_already_decided_suggestion_mints_no_second_topic(topics_engine):
    """Two ids for one theme would split its score trend in half."""
    owner_id = uuid4()
    suggestion = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))

    with pytest.raises(CoverageSuggestionNotPendingError):
        await repository.create_topic(
            owner_id,
            BILLING,
            centroid=BILLING_VECTOR,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            accepted_suggestion_id=(
                await repository.dismiss_topic_suggestion(suggestion.id, [owner_id])
            ).id,
        )

    # The refusal is in the same transaction as the insert, so no topic landed.
    assert await repository.load_active_topics(owner_id) == []


@pytest.mark.asyncio
async def test_create_topic_404s_for_an_unknown_suggestion_rather_than_403(topics_engine):
    owner_id = uuid4()
    await _suggest(uuid4(), label=BILLING)

    with pytest.raises(CoverageSuggestionNotFoundError):
        await repository.create_topic(
            owner_id,
            BILLING,
            centroid=BILLING_VECTOR,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            accepted_suggestion_id=uuid4(),
        )

    assert await repository.load_active_topics(owner_id) == []


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

    accepted = await _suggest(owner_id, label=BILLING, centroid=tuple(BILLING_VECTOR))
    await _post(owner_id, BILLING)
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
async def test_delete_is_soft(topics_engine):
    """The row survives for one narrow reason: a historical run must resolve its ids."""
    owner_id = uuid4()
    topic, _ = await _post(owner_id, UNRELATED)

    assert await repository.delete_topic(topic.id, [owner_id]) is None

    # Gone from the run's input...
    assert await repository.load_active_topics(owner_id) == []
    # ...but the row survives, so a historical run's topic_id still resolves.
    stored = await repository.list_topics([owner_id], include_deleted=True)
    assert [record.id for record in stored] == [topic.id]
    assert stored[0].is_deleted
    assert stored[0].deleted_at is not None


@pytest.mark.asyncio
async def test_deleting_twice_is_idempotent(topics_engine):
    owner_id = uuid4()
    topic, _ = await _post(owner_id, UNRELATED)

    await repository.delete_topic(topic.id, [owner_id])
    await repository.delete_topic(topic.id, [owner_id])

    stored = await repository.list_topics([owner_id], include_deleted=True)
    assert len(stored) == 1
    assert stored[0].is_deleted


@pytest.mark.asyncio
async def test_delete_404s_for_another_owner_rather_than_403(topics_engine):
    owner_id = uuid4()
    other_owner = uuid4()
    topic, _ = await _post(owner_id, UNRELATED)

    with pytest.raises(CoverageTopicNotFoundError):
        await repository.delete_topic(topic.id, [other_owner])

    with pytest.raises(CoverageTopicNotFoundError):
        await repository.delete_topic(uuid4(), [owner_id])

    assert len(await repository.load_active_topics(owner_id)) == 1


# --- the sink is not a topic --------------------------------------------------


def test_the_sink_has_no_id_to_address_it_by():
    """``Uncategorized`` is reported with ``topic_id: null``, so it has no id at all.

    That is why there is no reserved literal to special-case here: the sink cannot
    be deleted by construction rather than by a rule, and a caller who tries is
    naming nothing.
    """
    from cognee.modules.recall_coverage import types

    assert not hasattr(types, "SINK_TOPIC_ID")

    for not_an_id in ("other", "Other", "Uncategorized", "null"):
        with pytest.raises(CoverageTopicNotFoundError):
            repository.parse_topic_id(not_an_id)


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

    kept, _ = await _post(owner_id, BILLING)
    removed, _ = await _post(owner_id, UNRELATED)
    await _post(other_owner, BILLING)
    await repository.delete_topic(removed.id, [owner_id])

    visible = await repository.list_topics([owner_id])
    assert [record.id for record in visible] == [kept.id]

    everything = await repository.list_topics([owner_id], include_deleted=True)
    assert {record.id for record in everything} == {kept.id, removed.id}
    # Oldest first, the same order the assignment matmul used.
    assert [record.label for record in everything] == [BILLING, UNRELATED]


@pytest.mark.asyncio
async def test_topic_question_counts_are_as_of_the_newest_complete_run(topics_engine):
    """A count now, not a lifetime total, and 0 for a topic nothing has landed in.

    A lifetime total over every run would multiply by how often the owner happens
    to run coverage; a live count over an in-flight run would move while it is
    being read.
    """
    from cognee.modules.recall_coverage.models import (
        RecallCoverageQuestion,
        RecallCoverageRun,
    )
    from cognee.modules.recall_coverage.types import RunStatus

    async with topics_engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[RecallCoverageRun.__table__, RecallCoverageQuestion.__table__],
        )

    owner_id = uuid4()
    billing, _ = await _post(owner_id, BILLING)
    empty, _ = await _post(owner_id, UNRELATED)

    async with topics_engine.get_async_session() as session:
        older = RecallCoverageRun(
            owner_id=owner_id,
            agent_label="all",
            status=RunStatus.COMPLETE.value,
            created_at=repository._utc_now().replace(year=2020),
        )
        newest = RecallCoverageRun(
            owner_id=owner_id, agent_label="all", status=RunStatus.COMPLETE.value
        )
        session.add_all([older, newest])
        await session.flush()

        session.add_all(
            [
                RecallCoverageQuestion(
                    run_id=newest.id, user_id=owner_id, question="a", topic_id=billing.id
                ),
                RecallCoverageQuestion(
                    run_id=newest.id, user_id=owner_id, question="b", topic_id=billing.id
                ),
                # Uncategorized is not a topic anyone can list, so it is excluded.
                RecallCoverageQuestion(
                    run_id=newest.id, user_id=owner_id, question="c", topic_id=None
                ),
                # An older run must not add to the count.
                RecallCoverageQuestion(
                    run_id=older.id, user_id=owner_id, question="d", topic_id=billing.id
                ),
            ]
        )
        await session.commit()

    counts = await repository.topic_question_counts([owner_id])

    assert counts == {billing.id: 2}
    # A topic that received nothing is absent, so the caller reports 0 rather than
    # hiding it.
    assert empty.id not in counts


# --- the operations that deliberately do not exist ----------------------------


def test_there_is_no_rename_merge_or_split():
    """All three change what an existing topic id means while looking like an edit."""
    forbidden = [
        name
        for name in dir(repository)
        if any(word in name.lower() for word in ("rename", "merge", "split"))
    ]
    assert forbidden == []


def test_there_is_no_taxonomy_version_counter():
    """It was deleted with the metrics: nothing joins on it, and delete is still soft."""
    assert not hasattr(repository, "current_taxonomy_version")
    assert "taxonomy_version" not in RecallCoverageTopic.__table__.columns
