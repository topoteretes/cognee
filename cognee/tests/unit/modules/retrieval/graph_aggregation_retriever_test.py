from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.graph_aggregation_retriever import (
    AggregationOperation,
    AggregationSpec,
    GraphAggregationRetriever,
    _match_key,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

# A small typed graph. Two entity types (Issue, Pull Request) and three entities, where
# every entity carries type == "Entity" (the trap: you cannot count by node type). The
# category lives on the is_a edge to an EntityType node.
TYPE_ISSUE = "EntityType:issue"
TYPE_PR = "EntityType:pull_request"

NODES = [
    (TYPE_ISSUE, {"name": "Issue", "type": "EntityType"}),
    (TYPE_PR, {"name": "Pull Request", "type": "EntityType"}),
    (
        "Entity:login-bug",
        {"name": "Login bug", "type": "Entity", "description": "an open login problem"},
    ),
    (
        "Entity:crash",
        {"name": "Crash", "type": "Entity", "description": "a closed crash report"},
    ),
    (
        "Entity:fix-login",
        {"name": "Fix login", "type": "Entity", "description": "an open pull request"},
    ),
]

EDGES = [
    ("Entity:login-bug", TYPE_ISSUE, "is_a", {}),
    ("Entity:crash", TYPE_ISSUE, "is_a", {}),
    ("Entity:fix-login", TYPE_PR, "is_a", {}),
    ("Entity:login-bug", "Entity:fix-login", "relates_to", {}),
    ("Entity:login-bug", "Entity:crash", "relates_to", {}),
]


def _neighborhood_for(node_ids, depth=1, edge_types=None):
    """Mimic the adapter: entities linked by is_a to the seed types, plus edges among them."""
    type_ids = set(node_ids)
    is_a_edges = [edge for edge in EDGES if edge[2] == "is_a" and edge[1] in type_ids]
    kept_ids = type_ids | {edge[0] for edge in is_a_edges}
    nodes = [node for node in NODES if node[0] in kept_ids]
    edges = [edge for edge in EDGES if edge[0] in kept_ids and edge[1] in kept_ids]
    return nodes, edges


def _make_graph_engine(is_empty=False, vocabulary=None):
    engine = AsyncMock()
    engine.is_empty = AsyncMock(return_value=is_empty)
    if vocabulary is None:
        vocabulary = [node for node in NODES if node[1]["type"] == "EntityType"]
    engine.get_filtered_graph_data = AsyncMock(return_value=(vocabulary, []))
    engine.get_neighborhood = AsyncMock(side_effect=_neighborhood_for)
    engine.get_graph_data = AsyncMock(return_value=(NODES, EDGES))
    return engine


def _patches(engine, spec):
    return (
        patch(
            "cognee.modules.retrieval.graph_aggregation_retriever.get_graph_engine",
            new=AsyncMock(return_value=engine),
        ),
        patch(
            "cognee.modules.retrieval.graph_aggregation_retriever.LLMGateway."
            "acreate_structured_output",
            new=AsyncMock(return_value=spec),
        ),
    )


async def _run(spec, engine=None):
    engine = engine or _make_graph_engine()
    graph_patch, llm_patch = _patches(engine, spec)
    retriever = GraphAggregationRetriever()
    with graph_patch, llm_patch:
        retrieved = await retriever.get_retrieved_objects("question")
        result = retriever._compute_result(retrieved)
        context = await retriever.get_context_from_objects(retrieved_objects=retrieved)
        completion = await retriever.get_completion_from_context(
            retrieved_objects=retrieved, context=context
        )
    # The structured result lives on retrieved_objects; context is the rendered string.
    return result, context, completion


def test_match_key_normalizes_case_and_plural():
    assert _match_key("Issues") == _match_key("issue")
    assert _match_key("Pull Requests") == _match_key("pull request")
    assert _match_key("Issue") != _match_key("Person")


@pytest.mark.asyncio
async def test_count_differentiates_issue_from_pull_request():
    """The core of the issue: a Pull Request must not pollute an 'issues' count."""
    spec = AggregationSpec(operation=AggregationOperation.COUNT, target_type="issues")
    result, context, completion = await _run(spec)

    assert result["status"] == "ok"
    assert result["operation"] == "count"
    assert result["target_types"] == ["Issue"]
    # Two Issue entities, not three: the Pull Request is excluded structurally.
    assert result["count"] == 2
    assert "2" in context
    assert completion == [context]


@pytest.mark.asyncio
async def test_context_is_searchresultpayload_compatible():
    """The context must satisfy SearchResultPayload's str|List[str] contract.

    A dict context raises a pydantic ValidationError when the payload is constructed,
    so this asserts get_context_from_objects returns a string the payload accepts.
    """
    spec = AggregationSpec(operation=AggregationOperation.COUNT, target_type="issues")
    result, context, completion = await _run(spec)

    assert isinstance(context, str)
    payload = SearchResultPayload(
        result_object={"count": result["count"]},
        context=context,
        completion=completion,
        search_type=SearchType.GRAPH_AGGREGATION,
    )
    assert payload.context == context
    assert payload.completion == completion


@pytest.mark.asyncio
async def test_count_applies_best_effort_qualifier_filter():
    spec = AggregationSpec(
        operation=AggregationOperation.COUNT, target_type="issue", filters=["open"]
    )
    result, _, _ = await _run(spec)

    assert result["count"] == 2
    # Only the "open login problem" Issue matches the "open" qualifier.
    assert result["filtered_count"] == 1
    assert result["filters_best_effort"] is True


@pytest.mark.asyncio
async def test_count_unknown_type_refuses_instead_of_guessing():
    spec = AggregationSpec(operation=AggregationOperation.COUNT, target_type="dragon")
    engine = _make_graph_engine()
    graph_patch, llm_patch = _patches(engine, spec)
    retriever = GraphAggregationRetriever()
    with (
        graph_patch,
        llm_patch,
        patch(
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            side_effect=RuntimeError("no embedding backend in test"),
        ),
    ):
        retrieved = await retriever.get_retrieved_objects("how many dragons?")
        result = retriever._compute_result(retrieved)
        context = await retriever.get_context_from_objects(retrieved_objects=retrieved)

    assert result["status"] == "unknown_type"
    assert result["requested"] == "dragon"
    assert result["available_types"] == ["Issue", "Pull Request"]
    assert "count" not in result
    assert "dragon" in context


@pytest.mark.asyncio
async def test_embedding_fallback_refuses_baseline_cluster():
    """A nonsense noun sits at the embedding baseline distance from every type (high
    score, near-tie). The fallback must refuse rather than resolve to the nearest type.

    Sentence-embedding models report a high baseline similarity between unrelated
    words, so an absolute threshold alone is not enough to reject a non-match.
    """
    retriever = GraphAggregationRetriever()
    vocab = [("t1", "Issue"), ("t2", "Bug"), ("t3", "Person")]
    # Unit vectors vs query [1, 0] yield cosines 0.65 / 0.62 / 0.58 -- a high-baseline
    # near-tie with no clear winner.
    vectors = [[1.0, 0.0], [0.65, 0.7599], [0.62, 0.7846], [0.58, 0.8146]]
    engine = AsyncMock()
    engine.embed_text = AsyncMock(return_value=vectors)
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
        return_value=engine,
    ):
        resolved = await retriever._resolve_types_embedding("dragon", vocab)

    assert resolved == []


@pytest.mark.asyncio
async def test_embedding_fallback_refuses_high_score_near_tie():
    """Even above the absolute threshold, a near-tie at the top means there is no clear
    winner, so the margin check must still refuse rather than pick arbitrarily."""
    retriever = GraphAggregationRetriever()
    vocab = [("t1", "Issue"), ("t2", "Bug")]
    # Unit vectors vs query [1, 0] yield cosines 0.85 and 0.83: both above the 0.8
    # threshold, but the 0.02 gap is under the 0.05 margin.
    vectors = [[1.0, 0.0], [0.85, 0.5267], [0.83, 0.5578]]
    engine = AsyncMock()
    engine.embed_text = AsyncMock(return_value=vectors)
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
        return_value=engine,
    ):
        resolved = await retriever._resolve_types_embedding("ticket", vocab)

    assert resolved == []


@pytest.mark.asyncio
async def test_embedding_fallback_accepts_clear_winner():
    """A genuine semantic match (high score, large margin over the runner-up) resolves."""
    retriever = GraphAggregationRetriever()
    vocab = [("t1", "Issue"), ("t2", "Person")]
    # "ticket" ~ Issue at cosine 0.92, far above Person at 0.50.
    vectors = [[1.0, 0.0], [0.92, 0.3919], [0.50, 0.8660]]
    engine = AsyncMock()
    engine.embed_text = AsyncMock(return_value=vectors)
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
        return_value=engine,
    ):
        resolved = await retriever._resolve_types_embedding("ticket", vocab)

    assert resolved == [("t1", "Issue")]


@pytest.mark.asyncio
async def test_group_by_count_groups_on_is_a_target():
    spec = AggregationSpec(operation=AggregationOperation.GROUP_BY_COUNT)
    result, _, _ = await _run(spec)

    assert result["status"] == "ok"
    assert result["groups"] == {"Issue": 2, "Pull Request": 1}
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_top_by_degree_ranks_entities_only():
    spec = AggregationSpec(operation=AggregationOperation.TOP_BY_DEGREE)
    result, context, completion = await _run(spec)

    assert result["status"] == "ok"
    ranking = result["ranking"]
    # EntityType nodes are never ranked as entities.
    assert all(entry["id"].startswith("Entity:") for entry in ranking)
    # "Login bug" links to two other entities (Crash, Fix login) -> most connected.
    # The is_a edge to its EntityType is not counted as connectivity.
    assert ranking[0]["name"] == "Login bug"
    assert ranking[0]["degree"] == 2
    assert "Login bug" in context


@pytest.mark.asyncio
async def test_top_by_degree_works_without_entity_types():
    """top_by_degree ranks raw connectivity and must not be refused when no EntityType
    nodes exist (the vocabulary guard only applies to count)."""
    spec = AggregationSpec(operation=AggregationOperation.TOP_BY_DEGREE)
    engine = _make_graph_engine(vocabulary=[])
    result, context, _ = await _run(spec, engine=engine)

    assert result["status"] == "ok"
    assert result["operation"] == "top_by_degree"
    assert result["ranking"][0]["name"] == "Login bug"
    assert "Login bug" in context


@pytest.mark.asyncio
async def test_empty_graph_returns_clear_message():
    spec = AggregationSpec(operation=AggregationOperation.COUNT, target_type="issue")
    engine = _make_graph_engine(is_empty=True)
    result, context, completion = await _run(spec, engine=engine)

    assert result["status"] == "empty_graph"
    assert "empty" in context.lower()
