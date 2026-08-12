"""Guards on recall-coverage topic suggestions — spec section 2, phase 2 steps 8-10.

Five invariants:

* **Density, not partitioning.** Only a group of at least
  ``min_questions_per_topic`` similar sink questions becomes a candidate. k-means
  (``cognee/modules/visualization/semantic_clusters.py::compute_clusters``) is
  deliberately not used: it returns exactly ``k`` partitions whether or not any of
  them is dense, so it would propose topics out of scattered noise.
* **``cohesion`` orders and is never scored.** It decides which candidates a run
  shows first, and appears in no aggregate.
* **The re-proposal guard is owner-scoped and label-blind.** A theme dismissed on
  the Codex run does not come back on the Claude Code run; another owner's
  dismissal has no effect at all; a *pending* suggestion suppresses nothing,
  because it is not a decision.
* **Cross-agent isolation.** Both labels' runs see the owner's topics; neither
  run reads or mutates the other's pending suggestions.
* **One LLM call per surviving cluster, and only for survivors.** Everything that
  can drop a candidate runs before the labelling call.

No test here reaches a provider: the LLM is patched and every vector is written
out by hand.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.assign import assign_topics
from cognee.modules.recall_coverage.embedding import EmbeddingFingerprint, normalize_rows
from cognee.modules.recall_coverage.models import (
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.suggest import (
    cluster_centroid,
    cluster_cohesion,
    cluster_sink_questions,
    drop_reproposed,
    generate_topic_label,
    suggest_topics,
    topic_label_model,
)
from cognee.modules.recall_coverage.types import CoverageParams, SuggestionStatus

repository = importlib.import_module("cognee.modules.recall_coverage.repository")
suggest = importlib.import_module("cognee.modules.recall_coverage.suggest")

MODEL = "openai/text-embedding-3-large"
FINGERPRINT = EmbeddingFingerprint(model=MODEL, dimensions=3)


def _params(**overrides) -> CoverageParams:
    """A params snapshot that ignores the developer's ``.env``."""
    from cognee.modules.recall_coverage.config import RecallCoverageConfig

    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _tight_cluster(count, axis, spread=0.02):
    """``count`` near-identical vectors around one axis (cosine > 0.999 to each other)."""
    rows = []
    for index in range(count):
        row = [0.0, 0.0, 0.0]
        row[axis] = 1.0
        row[(axis + 1) % 3] = spread * index
        rows.append(row)
    return rows


@pytest_asyncio.fixture
async def suggestion_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the topics and suggestions tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="topic_suggestions_test.db",
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


async def _store_suggestion(engine, *, owner_id, centroid, status, agent_label=None, label="X"):
    async with engine.get_async_session() as session:
        row = RecallCoverageTopicSuggestion(
            owner_id=owner_id,
            agent_label=agent_label,
            run_id=uuid4(),
            label=label,
            centroid=list(centroid),
            embedding_model=MODEL,
            embedding_dimensions=3,
            question_count=5,
            cohesion=0.9,
            status=status,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


async def _store_topic(engine, *, owner_id, centroid, label="Billing & invoices"):
    async with engine.get_async_session() as session:
        row = RecallCoverageTopic(
            owner_id=owner_id,
            label=label,
            centroid=list(centroid),
            embedding_model=MODEL,
            embedding_dimensions=3,
            seed_question_count=5,
            taxonomy_version=1,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


# --- Clustering -------------------------------------------------------------


def test_only_a_dense_cluster_becomes_a_candidate():
    """Four similar questions plus scattered ones propose nothing at min size 5."""
    vectors = normalize_rows(
        _tight_cluster(4, axis=0) + [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )

    clusters = cluster_sink_questions(
        vectors,
        sink_cluster_threshold=0.80,
        min_questions_per_topic=5,
        max_suggestions_per_run=5,
    )

    assert clusters == []


def test_a_dense_cluster_of_five_is_proposed_with_its_members():
    vectors = normalize_rows(_tight_cluster(5, axis=0) + [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    clusters = cluster_sink_questions(
        vectors,
        sink_cluster_threshold=0.80,
        min_questions_per_topic=5,
        max_suggestions_per_run=5,
    )

    assert len(clusters) == 1
    assert clusters[0].member_indices == (0, 1, 2, 3, 4)
    assert clusters[0].question_count == 5
    # The centroid is L2-normalized, like every stored centroid.
    assert np.linalg.norm(np.asarray(clusters[0].centroid)) == pytest.approx(1.0)


def test_candidates_are_ordered_by_cohesion_and_capped():
    """Tightest theme first, and never more than ``max_suggestions_per_run``."""
    loose = _tight_cluster(5, axis=0, spread=0.25)
    tight = _tight_cluster(5, axis=1, spread=0.01)
    vectors = normalize_rows(loose + tight)

    clusters = cluster_sink_questions(
        vectors,
        sink_cluster_threshold=0.80,
        min_questions_per_topic=5,
        max_suggestions_per_run=5,
    )
    assert len(clusters) == 2
    assert clusters[0].cohesion > clusters[1].cohesion
    assert clusters[0].member_indices == (5, 6, 7, 8, 9)

    capped = cluster_sink_questions(
        vectors,
        sink_cluster_threshold=0.80,
        min_questions_per_topic=5,
        max_suggestions_per_run=1,
    )
    assert len(capped) == 1
    assert capped[0].member_indices == (5, 6, 7, 8, 9)


def test_cohesion_is_the_mean_intra_cluster_cosine():
    """And it is only ever an ordering key — no aggregate reads it."""
    orthogonal = normalize_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert cluster_cohesion(orthogonal) == pytest.approx(0.0)

    identical = normalize_rows([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert cluster_cohesion(identical) == pytest.approx(1.0)

    # A lone member has no pair to average.
    assert cluster_cohesion(normalize_rows([[1.0, 0.0, 0.0]])) == 1.0


def test_a_degenerate_centroid_stays_zero_rather_than_being_faked():
    """Opposing members average to nothing, and nothing is what is stored."""
    opposed = normalize_rows([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert cluster_centroid(opposed) == (0.0, 0.0, 0.0)
    assert cluster_centroid(np.zeros((0, 0))) == ()


def test_no_sink_questions_proposes_nothing():
    assert (
        cluster_sink_questions(
            np.zeros((0, 0)),
            sink_cluster_threshold=0.80,
            min_questions_per_topic=5,
            max_suggestions_per_run=5,
        )
        == []
    )


def test_k_means_is_not_used():
    """``compute_clusters`` returns k partitions whether or not any of them is dense.

    The requirement is literally a dense cluster, so this module must reuse dedup's
    single-link grouping and never reach for the visualization module's k-means.
    """
    assert not hasattr(suggest, "compute_clusters")

    with open(suggest.__file__, encoding="utf-8") as handle:
        source = handle.read()

    assert "cognee.modules.visualization" not in source
    assert "group_by_similarity" in source


# --- Re-proposal guard ------------------------------------------------------


def _candidate(vectors, **kwargs):
    return cluster_sink_questions(
        vectors,
        sink_cluster_threshold=kwargs.get("sink_cluster_threshold", 0.80),
        min_questions_per_topic=kwargs.get("min_questions_per_topic", 5),
        max_suggestions_per_run=kwargs.get("max_suggestions_per_run", 5),
    )


def _settled(centroid, status, *, agent_label=None, owner_id=None, model=MODEL, dimensions=3):
    return repository.SuggestionRecord(
        id=uuid4(),
        owner_id=owner_id or uuid4(),
        label="Previously seen",
        centroid=tuple(centroid),
        embedding_model=model,
        embedding_dimensions=dimensions,
        question_count=5,
        cohesion=0.9,
        status=status,
        agent_label=agent_label,
    )


@pytest.mark.parametrize(
    "status", [SuggestionStatus.DISMISSED.value, SuggestionStatus.ACCEPTED.value]
)
def test_a_settled_suggestion_suppresses_the_same_theme(status):
    clusters = _candidate(normalize_rows(_tight_cluster(5, axis=0)))
    assert len(clusters) == 1

    kept = drop_reproposed(
        clusters,
        [_settled(clusters[0].centroid, status)],
        fingerprint=FINGERPRINT,
        suggestion_dedup_threshold=0.90,
    )

    assert kept == []


def test_a_dismissal_under_one_agent_suppresses_another_agents_run():
    """Dismissing is a statement about the taxonomy, not about one tool."""
    clusters = _candidate(normalize_rows(_tight_cluster(5, axis=0)))

    kept = drop_reproposed(
        clusters,
        [_settled(clusters[0].centroid, SuggestionStatus.DISMISSED.value, agent_label="codex")],
        fingerprint=FINGERPRINT,
        suggestion_dedup_threshold=0.90,
    )

    assert kept == []


def test_an_unrelated_settled_suggestion_suppresses_nothing():
    clusters = _candidate(normalize_rows(_tight_cluster(5, axis=0)))

    kept = drop_reproposed(
        clusters,
        [_settled((0.0, 1.0, 0.0), SuggestionStatus.DISMISSED.value)],
        fingerprint=FINGERPRINT,
        suggestion_dedup_threshold=0.90,
    )

    assert len(kept) == 1


def test_a_settled_suggestion_from_another_embedding_space_is_ignored_not_fatal():
    """A stale suggestion only weakens the filter; it cannot produce a wrong number."""
    clusters = _candidate(normalize_rows(_tight_cluster(5, axis=0)))

    kept = drop_reproposed(
        clusters,
        [
            _settled(
                clusters[0].centroid,
                SuggestionStatus.DISMISSED.value,
                model="openai/text-embedding-ada-002",
            )
        ],
        fingerprint=FINGERPRINT,
        suggestion_dedup_threshold=0.90,
    )

    assert len(kept) == 1


# --- Label generation -------------------------------------------------------


def test_the_label_prompt_files_exist_and_render():
    """``read_query_prompt`` returns None for a missing file instead of raising.

    Without this test a renamed or unpackaged prompt would silently send
    ``system_prompt=None`` to the provider and label topics from no instructions.
    """
    text_input, system_prompt = suggest._label_prompts(
        ["Where are the runbooks?", "How do I rotate credentials?"], max_chars=60
    )

    assert "Where are the runbooks?" in text_input
    assert "60" in text_input
    assert "noun phrase" in system_prompt


def test_the_label_length_limit_is_the_config_parameter():
    """``topic_label_max_chars`` reaches the response model, not a literal."""
    model = topic_label_model(24)

    assert model.model_fields["label"].metadata[0].max_length == 24
    # Cached per limit, so the structured-output schema identity is stable.
    assert topic_label_model(24) is model
    assert topic_label_model(60) is not model


@pytest.mark.asyncio
async def test_generate_topic_label_truncates_a_provider_that_ignores_the_schema():
    long_label = "Billing, invoices, dunning, refunds and every other money question"

    with patch.object(
        suggest.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(label=long_label),
    ):
        label = await generate_topic_label(["a", "b"], topic_label_max_chars=20)

    assert label == long_label[:20]


# --- The orchestrator -------------------------------------------------------


@pytest.mark.asyncio
async def test_suggest_topics_writes_one_pending_row_per_surviving_cluster(suggestion_engine):
    owner_id = uuid4()
    run_id = uuid4()
    texts = [f"question {index}" for index in range(5)] + ["unrelated"]
    vectors = normalize_rows(_tight_cluster(5, axis=0) + [[0.0, 1.0, 0.0]])

    with patch.object(
        suggest.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(label="Billing & invoices"),
    ) as mock_llm:
        written = await suggest_topics(
            owner_id,
            texts,
            vectors,
            params=_params(),
            fingerprint=FINGERPRINT,
            run_id=run_id,
            agent_label="claude-code",
        )

    # One call, for the one surviving cluster, carrying that cluster's texts.
    assert mock_llm.await_count == 1
    assert "question 0" in mock_llm.await_args.kwargs["text_input"]
    assert "unrelated" not in mock_llm.await_args.kwargs["text_input"]

    assert len(written) == 1
    suggestion = written[0]
    assert suggestion.status == SuggestionStatus.PENDING.value
    assert suggestion.label == "Billing & invoices"
    assert suggestion.owner_id == owner_id
    # Provenance, not scope.
    assert suggestion.agent_label == "claude-code"
    assert suggestion.run_id == run_id
    assert suggestion.question_count == 5
    assert suggestion.cohesion > 0.99
    assert suggestion.embedding_model == MODEL
    assert suggestion.embedding_dimensions == 3

    pending = await repository.load_pending_suggestions(owner_id)
    assert [row.id for row in pending] == [suggestion.id]


@pytest.mark.asyncio
async def test_suggest_topics_makes_no_llm_call_when_nothing_survives(suggestion_engine):
    """Everything that can drop a candidate runs before the labelling call."""
    owner_id = uuid4()
    vectors = normalize_rows(_tight_cluster(5, axis=0))
    clusters = _candidate(vectors)
    await _store_suggestion(
        suggestion_engine,
        owner_id=owner_id,
        centroid=clusters[0].centroid,
        status=SuggestionStatus.DISMISSED.value,
        agent_label="codex",
    )

    with patch.object(
        suggest.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
    ) as mock_llm:
        written = await suggest_topics(
            owner_id,
            [f"question {index}" for index in range(5)],
            vectors,
            params=_params(),
            fingerprint=FINGERPRINT,
            agent_label="claude-code",
        )

    assert written == []
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_label_drops_the_candidate_instead_of_failing_the_run(suggestion_engine):
    """Suggestions are advisory; losing one must not lose a run's judged rows."""
    owner_id = uuid4()
    vectors = normalize_rows(_tight_cluster(5, axis=0) + _tight_cluster(5, axis=1, spread=0.05))
    texts = [f"question {index}" for index in range(10)]

    with patch.object(
        suggest.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=[SimpleNamespace(label="Billing & invoices"), RuntimeError("provider down")],
    ):
        written = await suggest_topics(
            owner_id,
            texts,
            vectors,
            params=_params(),
            fingerprint=FINGERPRINT,
        )

    assert [row.label for row in written] == ["Billing & invoices"]


@pytest.mark.asyncio
async def test_an_empty_label_drops_the_candidate(suggestion_engine):
    owner_id = uuid4()
    vectors = normalize_rows(_tight_cluster(5, axis=0))

    with patch.object(
        suggest.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(label="   "),
    ):
        written = await suggest_topics(
            owner_id,
            [f"question {index}" for index in range(5)],
            vectors,
            params=_params(),
            fingerprint=FINGERPRINT,
        )

    assert written == []
    assert await repository.load_pending_suggestions(owner_id) == []


# --- Cross-agent isolation --------------------------------------------------


@pytest.mark.asyncio
async def test_two_agents_share_the_owners_topics_but_not_each_others_suggestions(
    suggestion_engine,
):
    """The isolation property the whole owner-scoped taxonomy rests on.

    One taxonomy is what makes "Codex 4.2 on Billing, Claude Code 2.1 on Billing"
    comparable, so both labels' runs must be scored against the same topics.
    Pending suggestions are the opposite: label A's run neither reads nor mutates
    label B's queue, and only *settled* decisions cross labels.
    """
    owner_id = uuid4()
    other_owner_id = uuid4()

    billing_centroid = tuple(normalize_rows([[1.0, 0.0, 0.0]])[0])
    await _store_topic(suggestion_engine, owner_id=owner_id, centroid=billing_centroid)
    # Another owner's topic and dismissal must be invisible here.
    await _store_topic(
        suggestion_engine, owner_id=other_owner_id, centroid=(0.0, 1.0, 0.0), label="Elsewhere"
    )

    incidents = normalize_rows(_tight_cluster(5, axis=1, spread=0.01))
    codex_pending_id = await _store_suggestion(
        suggestion_engine,
        owner_id=owner_id,
        centroid=cluster_centroid(incidents),
        status=SuggestionStatus.PENDING.value,
        agent_label="codex",
        label="Incidents (proposed by the codex run)",
    )
    await _store_suggestion(
        suggestion_engine,
        owner_id=other_owner_id,
        centroid=cluster_centroid(incidents),
        status=SuggestionStatus.DISMISSED.value,
        label="Someone else's dismissal",
    )

    topics = await repository.load_active_topics(owner_id)
    assert [topic.id for topic in topics] and all(topic.owner_id == owner_id for topic in topics)
    assert "Elsewhere" not in [topic.label for topic in topics]

    # Both labels' runs are scored against that same single topic.
    for agent_label in ("claude-code", "codex"):
        assignment = assign_topics(
            normalize_rows([[1.0, 0.02, 0.0]]),
            topics,
            fingerprint=FINGERPRINT,
            assignment_threshold=0.55,
            assignment_margin=0.05,
        )
        assert assignment.assignments[0].topic_id == topics[0].id, agent_label

    # A pending suggestion from the codex run does not suppress the same theme on
    # the claude-code run, and another owner's dismissal does not either.
    with patch.object(
        suggest.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(label="Incidents"),
    ):
        written = await suggest_topics(
            owner_id,
            [f"incident question {index}" for index in range(5)],
            incidents,
            params=_params(),
            fingerprint=FINGERPRINT,
            agent_label="claude-code",
        )

    assert [row.label for row in written] == ["Incidents"]
    assert written[0].agent_label == "claude-code"

    # The codex queue is untouched, and the new row joined it rather than
    # replacing it.
    pending = await repository.load_pending_suggestions(owner_id)
    assert {row.id for row in pending} == {codex_pending_id, written[0].id}
    assert other_owner_id not in {row.owner_id for row in pending}

    # Nothing was written into the other owner's scope.
    assert await repository.load_pending_suggestions(other_owner_id) == []
