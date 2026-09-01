import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from cognee.exceptions import CogneeValidationError
from cognee.modules.observability import capture
from cognee.modules.observability.capture import KIND_RETRIEVAL_CANDIDATES
from cognee.modules.retrieval.utils import brute_force_triplet_search as bfts_module
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
    get_memory_fragment,
    format_triplets,
)
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class MockScoredResult:
    """Mock class for vector search results."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


@pytest.mark.asyncio
async def test_brute_force_triplet_search_empty_query():
    """Test that empty query raises ValueError."""
    with pytest.raises(ValueError, match="The query must be a non-empty string."):
        await brute_force_triplet_search(query="")


@pytest.mark.asyncio
async def test_brute_force_triplet_search_none_query():
    """Test that None query raises ValueError."""
    with pytest.raises(ValueError, match="Must provide either 'query' or 'query_batch'."):
        await brute_force_triplet_search(query=None)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_negative_top_k():
    """Test that negative top_k raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be a positive integer."):
        await brute_force_triplet_search(query="test query", top_k=-1)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_zero_top_k():
    """Test that zero top_k raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be a positive integer."):
        await brute_force_triplet_search(query="test query", top_k=0)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_wide_search_limit_global_search():
    """Test that wide_search_limit is applied for global search (node_name=None)."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(
            query="test",
            node_name=None,  # Global search
            wide_search_top_k=75,
        )

        for call in mock_vector_engine.search.call_args_list:
            assert call[1]["limit"] == 75


@pytest.mark.asyncio
async def test_brute_force_triplet_search_wide_search_limit_filtered_search():
    """Test that wide_search_limit is None for filtered search (node_name provided)."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(
            query="test",
            node_name=["Node1"],
            wide_search_top_k=50,
        )

        for call in mock_vector_engine.search.call_args_list:
            assert call[1]["limit"] is None


@pytest.mark.asyncio
async def test_brute_force_triplet_search_wide_search_default():
    """Test that wide_search_top_k defaults to 100."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        for call in mock_vector_engine.search.call_args_list:
            assert call[1]["limit"] == 100


@pytest.mark.asyncio
async def test_brute_force_triplet_search_default_collections():
    """Test that default collections are used when none provided."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test")

        expected_collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
            "DltRow_text",
            "EdgeType_relationship_name",
        ]

        call_collections = [
            call[1]["collection_name"] for call in mock_vector_engine.search.call_args_list
        ]
        assert call_collections == expected_collections


@pytest.mark.asyncio
async def test_brute_force_triplet_search_custom_collections():
    """Test that custom collections are used when provided."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    custom_collections = ["CustomCol1", "CustomCol2"]

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test", collections=custom_collections)

        call_collections = [
            call[1]["collection_name"] for call in mock_vector_engine.search.call_args_list
        ]
        assert set(call_collections) == set(custom_collections) | {"EdgeType_relationship_name"}


@pytest.mark.asyncio
async def test_brute_force_triplet_search_always_includes_edge_collection():
    """Test that EdgeType_relationship_name is always searched even when not in collections."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    collections_without_edge = ["Entity_name", "TextSummary_text"]

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test", collections=collections_without_edge)

        call_collections = [
            call[1]["collection_name"] for call in mock_vector_engine.search.call_args_list
        ]
        assert "EdgeType_relationship_name" in call_collections
        assert set(call_collections) == set(collections_without_edge) | {
            "EdgeType_relationship_name"
        }


@pytest.mark.asyncio
async def test_brute_force_triplet_search_does_not_mutate_caller_collections():
    """Regression: the caller's collections list must not be mutated.

    The edge collection is appended to a local copy, not to the list the caller
    passed in (e.g. a context provider's persistent, shared ``self.collections``).
    The same list is reused across two calls to mimic a caller that runs many
    searches with one configured list — it must never grow or accumulate
    duplicates.
    """
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    caller_collections = ["Entity_name", "TextSummary_text"]
    snapshot = list(caller_collections)

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test", collections=caller_collections)
        await brute_force_triplet_search(query="test", collections=caller_collections)

    # The edge collection is still searched (added to the internal copy)...
    searched = {call[1]["collection_name"] for call in mock_vector_engine.search.call_args_list}
    assert "EdgeType_relationship_name" in searched
    # ...but the caller's own list is left untouched across repeated calls.
    assert caller_collections == snapshot


@pytest.mark.asyncio
async def test_brute_force_triplet_search_caller_collections_with_edge_not_duplicated():
    """If the caller already includes the edge collection, the list is neither
    mutated nor given a duplicate entry."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    caller_collections = ["Entity_name", "EdgeType_relationship_name"]
    snapshot = list(caller_collections)

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test", collections=caller_collections)

    assert caller_collections == snapshot


@pytest.mark.asyncio
async def test_triplet_context_provider_does_not_mutate_configured_collections():
    """End-to-end regression for issue #3481.

    TripletSearchContextProvider keeps a single ``self.collections`` and passes the
    same list into one brute_force_triplet_search() per entity. Running a context
    search across multiple entities must not mutate that configured list.
    """
    from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
        TripletSearchContextProvider,
    )

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    class _Entity:
        def __init__(self, name):
            self.name = name

    provider = TripletSearchContextProvider(collections=["Entity_name"])

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.context_providers."
            "TripletSearchContextProvider.get_memory_fragment",
            new=AsyncMock(return_value=CogneeGraph()),
        ),
    ):
        await provider.get_context([_Entity("Alice"), _Entity("Bob")], query="how are they related")

    # The provider's configured collections list is unchanged after the search.
    assert provider.collections == ["Entity_name"]


@pytest.mark.asyncio
async def test_brute_force_triplet_search_all_collections_empty():
    """Test that empty list is returned when all collections return no results."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        results = await brute_force_triplet_search(query="test")
        assert results == []


# Tests for query embedding


@pytest.mark.asyncio
async def test_brute_force_triplet_search_embeds_query():
    """Test that query is embedded before searching."""
    query_text = "test query"
    expected_vector = [0.1, 0.2, 0.3]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[expected_vector])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query=query_text)

        mock_vector_engine.embedding_engine.embed_text.assert_called_once_with([query_text])

        for call in mock_vector_engine.search.call_args_list:
            assert call[1]["query_vector"] == expected_vector


@pytest.mark.asyncio
async def test_brute_force_triplet_search_extracts_node_ids_global_search():
    """Test that node IDs are extracted from search results for global search."""
    scored_results = [
        MockScoredResult("node1", 0.95),
        MockScoredResult("node2", 0.87),
        MockScoredResult("node3", 0.92),
    ]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=scored_results)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert set(call_kwargs["relevant_ids_to_filter"]) == {"node1", "node2", "node3"}


@pytest.mark.asyncio
async def test_brute_force_triplet_search_reuses_provided_fragment():
    """Test that provided memory fragment is reused instead of creating new one."""
    provided_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("n1", 0.95)])

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment"
        ) as mock_get_fragment,
    ):
        await brute_force_triplet_search(
            query="test",
            memory_fragment=provided_fragment,
            node_name=["node"],
        )

        mock_get_fragment.assert_not_called()


@pytest.mark.asyncio
async def test_brute_force_triplet_search_creates_fragment_when_not_provided():
    """Test that memory fragment is created when not provided."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("n1", 0.95)])

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment,
    ):
        await brute_force_triplet_search(query="test", node_name=["node"])

        mock_get_fragment.assert_called_once()


@pytest.mark.asyncio
async def test_brute_force_triplet_search_passes_top_k_to_importance_calculation():
    """Test that custom top_k is passed to importance calculation."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("n1", 0.95)])

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        custom_top_k = 15
        await brute_force_triplet_search(query="test", top_k=custom_top_k, node_name=["n"])

        mock_fragment.calculate_top_triplet_importances.assert_called_once_with(
            k=custom_top_k, query_list_length=None, feedback_influence=0.0
        )


@pytest.mark.asyncio
async def test_brute_force_triplet_search_applies_personal_weights():
    """personal_weights are handed to the fragment after distance mapping."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("n1", 0.95)])

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )
    # apply_personal_weights is a plain (sync) method on CogneeGraph.
    mock_fragment.apply_personal_weights = MagicMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        await brute_force_triplet_search(
            query="test", node_name=["node"], personal_weights={"n1": 0.9}
        )

    mock_fragment.apply_personal_weights.assert_called_once_with({"n1": 0.9})
    mock_fragment.map_vector_distances_to_graph_nodes.assert_awaited_once()
    mock_fragment.map_vector_distances_to_graph_edges.assert_awaited_once()
    mock_fragment.calculate_top_triplet_importances.assert_awaited_once()


@pytest.mark.asyncio
async def test_brute_force_triplet_search_skips_personal_weights_when_absent():
    """Without personal_weights the fragment is never touched — byte-identical path."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("n1", 0.95)])

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )
    mock_fragment.apply_personal_weights = MagicMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        await brute_force_triplet_search(query="test", node_name=["node"])

    mock_fragment.apply_personal_weights.assert_not_called()


@pytest.mark.asyncio
async def test_get_memory_fragment_projects_feedback_weight_only_when_feedback_influence_enabled():
    """Test that feedback_weight properties are projected only when feedback_influence > 0."""
    mock_graph_engine = AsyncMock()
    mock_fragment = MagicMock(spec=CogneeGraph)
    mock_fragment.project_graph_from_db = AsyncMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.CogneeGraph",
            return_value=mock_fragment,
        ),
    ):
        await get_memory_fragment(feedback_influence=0.0)
        kwargs_without_feedback = mock_fragment.project_graph_from_db.call_args.kwargs
        assert "feedback_weight" not in kwargs_without_feedback["node_properties_to_project"]
        assert "feedback_weight" not in kwargs_without_feedback["edge_properties_to_project"]

        await get_memory_fragment(feedback_influence=0.2)
        kwargs_with_feedback = mock_fragment.project_graph_from_db.call_args.kwargs
        assert "feedback_weight" in kwargs_with_feedback["node_properties_to_project"]
        assert "feedback_weight" in kwargs_with_feedback["edge_properties_to_project"]
        assert kwargs_with_feedback["feedback_influence"] == 0.2


@pytest.mark.asyncio
async def test_brute_force_triplet_search_invalid_feedback_influence_raises():
    """Test feedback_influence value range validation."""
    with pytest.raises(
        CogneeValidationError, match="feedback_influence must be in range \\[0, 1\\]"
    ):
        await brute_force_triplet_search(query="test query", feedback_influence=1.5)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_negative_feedback_influence_raises():
    """Test feedback_influence lower bound validation."""
    with pytest.raises(
        CogneeValidationError, match="feedback_influence must be in range \\[0, 1\\]"
    ):
        await brute_force_triplet_search(query="test query", feedback_influence=-0.1)


@pytest.mark.asyncio
async def test_get_memory_fragment_returns_empty_graph_on_entity_not_found():
    """Test that get_memory_fragment returns empty graph when entity not found (line 85)."""
    mock_graph_engine = AsyncMock()

    # Create a mock fragment that will raise EntityNotFoundError when project_graph_from_db is called
    mock_fragment = MagicMock(spec=CogneeGraph)
    mock_fragment.project_graph_from_db = AsyncMock(
        side_effect=EntityNotFoundError("Entity not found")
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.CogneeGraph",
            return_value=mock_fragment,
        ),
    ):
        result = await get_memory_fragment()

        # Fragment should be returned even though EntityNotFoundError was raised (pass statement on line 85)
        assert result == mock_fragment
        mock_fragment.project_graph_from_db.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_memory_fragment_returns_empty_graph_on_error():
    """Test that get_memory_fragment returns empty graph on generic error."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.project_graph_from_db = AsyncMock(side_effect=Exception("Generic error"))

    with patch(
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
        return_value=mock_graph_engine,
    ):
        fragment = await get_memory_fragment()

        assert isinstance(fragment, CogneeGraph)
        assert len(fragment.nodes) == 0


@pytest.mark.asyncio
async def test_brute_force_triplet_search_deduplicates_node_ids():
    """Test that duplicate node IDs across collections are deduplicated."""

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [
                MockScoredResult("node1", 0.95),
                MockScoredResult("node2", 0.87),
            ]
        elif collection_name == "TextSummary_text":
            return [
                MockScoredResult("node1", 0.90),
                MockScoredResult("node3", 0.92),
            ]
        else:
            return []

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert set(call_kwargs["relevant_ids_to_filter"]) == {"node1", "node2", "node3"}
        assert len(call_kwargs["relevant_ids_to_filter"]) == 3


@pytest.mark.asyncio
async def test_brute_force_triplet_search_excludes_edge_collection():
    """Test that EdgeType_relationship_name collection is excluded from ID extraction."""

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [MockScoredResult("node1", 0.95)]
        elif collection_name == "EdgeType_relationship_name":
            return [MockScoredResult("edge1", 0.88)]
        else:
            return []

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(
            query="test",
            node_name=None,
            collections=["Entity_name", "EdgeType_relationship_name"],
        )

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert call_kwargs["relevant_ids_to_filter"] == ["node1"]


@pytest.mark.asyncio
async def test_brute_force_triplet_search_skips_nodes_without_ids():
    """Test that nodes without ID attribute are skipped."""

    class ScoredResultNoId:
        """Mock result without id attribute."""

        def __init__(self, score):
            self.score = score

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [
                MockScoredResult("node1", 0.95),
                ScoredResultNoId(0.90),
                MockScoredResult("node2", 0.87),
            ]
        else:
            return []

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert set(call_kwargs["relevant_ids_to_filter"]) == {"node1", "node2"}


@pytest.mark.asyncio
async def test_brute_force_triplet_search_handles_tuple_results():
    """Test that both list and tuple results are handled correctly."""

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return (
                MockScoredResult("node1", 0.95),
                MockScoredResult("node2", 0.87),
            )
        else:
            return []

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert set(call_kwargs["relevant_ids_to_filter"]) == {"node1", "node2"}


@pytest.mark.asyncio
async def test_brute_force_triplet_search_mixed_empty_collections():
    """Test ID extraction with mixed empty and non-empty collections."""

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [MockScoredResult("node1", 0.95)]
        elif collection_name == "TextSummary_text":
            return []
        elif collection_name == "EntityType_name":
            return [MockScoredResult("node2", 0.92)]
        else:
            return []

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment_fn,
    ):
        await brute_force_triplet_search(query="test", node_name=None)

        call_kwargs = mock_get_fragment_fn.call_args[1]
        assert set(call_kwargs["relevant_ids_to_filter"]) == {"node1", "node2"}


def test_format_triplets():
    """Test format_triplets function."""
    mock_edge = MagicMock()
    mock_node1 = MagicMock()
    mock_node2 = MagicMock()

    mock_node1.attributes = {"name": "Node1", "type": "Entity", "id": "n1"}
    mock_node2.attributes = {"name": "Node2", "type": "Entity", "id": "n2"}
    mock_edge.attributes = {"relationship_name": "relates_to", "edge_text": "connects"}

    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2

    result = format_triplets([mock_edge])

    assert isinstance(result, str)
    assert "Node1" in result
    assert "Node2" in result
    assert "relates_to" in result
    assert "connects" in result


def test_format_triplets_with_none_values():
    """Test format_triplets filters out None values."""
    mock_edge = MagicMock()
    mock_node1 = MagicMock()
    mock_node2 = MagicMock()

    mock_node1.attributes = {"name": "Node1", "type": None, "id": "n1"}
    mock_node2.attributes = {"name": "Node2", "type": "Entity", "id": None}
    mock_edge.attributes = {"relationship_name": "relates_to", "edge_text": None}

    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2

    result = format_triplets([mock_edge])

    assert "Node1" in result
    assert "Node2" in result
    assert "relates_to" in result
    assert "None" not in result or result.count("None") == 0


def test_format_triplets_with_nested_dict():
    """Test format_triplets handles nested dict attributes (lines 23-35)."""
    mock_edge = MagicMock()
    mock_node1 = MagicMock()
    mock_node2 = MagicMock()

    mock_node1.attributes = {"name": "Node1", "metadata": {"type": "Entity", "id": "n1"}}
    mock_node2.attributes = {"name": "Node2", "metadata": {"type": "Entity", "id": "n2"}}
    mock_edge.attributes = {"relationship_name": "relates_to"}

    mock_edge.node1 = mock_node1
    mock_edge.node2 = mock_node2

    result = format_triplets([mock_edge])

    assert isinstance(result, str)
    assert "Node1" in result
    assert "Node2" in result
    assert "relates_to" in result


@pytest.mark.asyncio
async def test_brute_force_triplet_search_vector_engine_init_error():
    """Test brute_force_triplet_search handles vector engine initialization error (lines 145-147)."""
    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async"
        ) as mock_get_vector_engine,
    ):
        mock_get_vector_engine.side_effect = Exception("Initialization error")

        with pytest.raises(RuntimeError, match="Initialization error"):
            await brute_force_triplet_search(query="test query")


@pytest.mark.asyncio
async def test_brute_force_triplet_search_collection_not_found_error():
    """Test brute_force_triplet_search handles CollectionNotFoundError in search (lines 156-157)."""
    mock_vector_engine = AsyncMock()
    mock_embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine = mock_embedding_engine
    mock_embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    mock_vector_engine.search = AsyncMock(
        side_effect=[
            CollectionNotFoundError("Collection not found"),
            [],
            [],
        ]
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=CogneeGraph(),
        ),
    ):
        result = await brute_force_triplet_search(
            query="test query", collections=["missing_collection", "existing_collection"]
        )

    assert result == []


@pytest.mark.asyncio
async def test_brute_force_triplet_search_generic_exception():
    """Test brute_force_triplet_search handles generic exceptions (lines 209-217)."""
    mock_vector_engine = AsyncMock()
    mock_embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine = mock_embedding_engine
    mock_embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    mock_vector_engine.search = AsyncMock(side_effect=Exception("Generic error"))

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
    ):
        with pytest.raises(Exception, match="Generic error"):
            await brute_force_triplet_search(query="test query")


@pytest.mark.asyncio
async def test_brute_force_triplet_search_with_node_name_sets_relevant_ids_to_none():
    """Test brute_force_triplet_search sets relevant_ids_to_filter to None when node_name is provided (line 191)."""
    mock_vector_engine = AsyncMock()
    mock_embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine = mock_embedding_engine
    mock_embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    mock_result = MockScoredResult(id="node1", score=0.8, payload={"id": "node1"})
    mock_vector_engine.search = AsyncMock(return_value=[mock_result])

    mock_fragment = AsyncMock()
    mock_fragment.map_vector_distances_to_graph_nodes = AsyncMock()
    mock_fragment.map_vector_distances_to_graph_edges = AsyncMock()
    mock_fragment.calculate_top_triplet_importances = AsyncMock(return_value=[])

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment,
    ):
        await brute_force_triplet_search(query="test query", node_name=["Node1"])

        assert mock_get_fragment.called
        call_kwargs = mock_get_fragment.call_args.kwargs if mock_get_fragment.call_args else {}
        assert call_kwargs.get("relevant_ids_to_filter") is None


@pytest.mark.asyncio
async def test_brute_force_triplet_search_collection_not_found_at_top_level():
    """Test brute_force_triplet_search handles CollectionNotFoundError at top level (line 210)."""
    mock_vector_engine = AsyncMock()
    mock_embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine = mock_embedding_engine
    mock_embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    mock_result = MockScoredResult(id="node1", score=0.8, payload={"id": "node1"})
    mock_vector_engine.search = AsyncMock(return_value=[mock_result])

    mock_fragment = AsyncMock()
    mock_fragment.map_vector_distances_to_graph_nodes = AsyncMock()
    mock_fragment.map_vector_distances_to_graph_edges = AsyncMock()
    mock_fragment.calculate_top_triplet_importances = AsyncMock(
        side_effect=CollectionNotFoundError("Collection not found")
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        result = await brute_force_triplet_search(query="test query")

    assert result == []


@pytest.mark.asyncio
async def test_brute_force_triplet_search_single_query_regression():
    """Test that single-query mode maintains legacy behavior (flat list, ID filtering)."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("node1", 0.95)])

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment,
    ):
        result = await brute_force_triplet_search(
            query="q1", query_batch=None, wide_search_top_k=10, node_name=None
        )

        assert isinstance(result, list)
        assert not (result and isinstance(result[0], list))
        mock_get_fragment.assert_called_once()
        call_kwargs = mock_get_fragment.call_args[1]
        assert call_kwargs["relevant_ids_to_filter"] is not None


@pytest.mark.asyncio
async def test_brute_force_triplet_search_batch_wiring_happy_path():
    """Test that batch mode returns list-of-lists and skips ID filtering."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.batch_search = AsyncMock(
        return_value=[
            [MockScoredResult("node1", 0.95)],
            [MockScoredResult("node2", 0.87)],
        ]
    )

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[[], []]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment,
    ):
        result = await brute_force_triplet_search(query_batch=["q1", "q2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        mock_get_fragment.assert_called_once()
        call_kwargs = mock_get_fragment.call_args[1]
        assert call_kwargs["relevant_ids_to_filter"] is None


@pytest.mark.asyncio
async def test_brute_force_triplet_search_shape_propagation_to_graph():
    """Test that query_list_length is passed through to graph mapping methods."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.batch_search = AsyncMock(
        return_value=[
            [MockScoredResult("node1", 0.95)],
            [MockScoredResult("node2", 0.87)],
        ]
    )

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[[], []]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        await brute_force_triplet_search(query_batch=["q1", "q2"])

        mock_fragment.map_vector_distances_to_graph_nodes.assert_called_once()
        node_call_kwargs = mock_fragment.map_vector_distances_to_graph_nodes.call_args[1]
        assert "query_list_length" in node_call_kwargs
        assert node_call_kwargs["query_list_length"] == 2

        mock_fragment.map_vector_distances_to_graph_edges.assert_called_once()
        edge_call_kwargs = mock_fragment.map_vector_distances_to_graph_edges.call_args[1]
        assert "query_list_length" in edge_call_kwargs
        assert edge_call_kwargs["query_list_length"] == 2

        mock_fragment.calculate_top_triplet_importances.assert_called_once()
        importance_call_kwargs = mock_fragment.calculate_top_triplet_importances.call_args[1]
        assert "query_list_length" in importance_call_kwargs
        assert importance_call_kwargs["query_list_length"] == 2


@pytest.mark.asyncio
async def test_brute_force_triplet_search_batch_path_comprehensive():
    """Test batch mode: returns list-of-lists, skips ID filtering, passes None for wide_search_limit."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()

    def batch_search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [
                [MockScoredResult("node1", 0.95)],
                [MockScoredResult("node2", 0.87)],
            ]
        elif collection_name == "EdgeType_relationship_name":
            return [
                [MockScoredResult("edge1", 0.92)],
                [MockScoredResult("edge2", 0.88)],
            ]
        return [[], []]

    mock_vector_engine.batch_search = AsyncMock(side_effect=batch_search_side_effect)

    mock_fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[[], []]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ) as mock_get_fragment,
    ):
        result = await brute_force_triplet_search(
            query_batch=["q1", "q2"], collections=["Entity_name", "EdgeType_relationship_name"]
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

        mock_get_fragment.assert_called_once()
        fragment_call_kwargs = mock_get_fragment.call_args[1]
        assert fragment_call_kwargs["relevant_ids_to_filter"] is None

        batch_search_calls = mock_vector_engine.batch_search.call_args_list
        assert len(batch_search_calls) > 0
        for call in batch_search_calls:
            assert call[1]["limit"] is None


@pytest.mark.asyncio
async def test_brute_force_triplet_search_batch_error_fallback():
    """Test that CollectionNotFoundError in batch mode returns [[], []] matching batch length."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.batch_search = AsyncMock(
        side_effect=CollectionNotFoundError("Collection not found")
    )

    with patch(
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        result = await brute_force_triplet_search(query_batch=["q1", "q2"])

        assert result == [[], []]
        assert len(result) == 2


@pytest.mark.asyncio
async def test_cognee_graph_mapping_batch_shapes():
    """Test that CogneeGraph mapping methods accept list-of-lists with query_list_length set."""
    from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge

    graph = CogneeGraph()
    node1 = Node("node1", {"name": "Node1"})
    node2 = Node("node2", {"name": "Node2"})
    graph.add_node(node1)
    graph.add_node(node2)

    edge = Edge(node1, node2, attributes={"edge_text": "relates_to"})
    graph.add_edge(edge)

    node_distances_batch = {
        "Entity_name": [
            [MockScoredResult("node1", 0.95)],
            [MockScoredResult("node2", 0.87)],
        ]
    }

    edge_1_text = "relates_to"
    edge_2_text = "relates_to"
    edge_distances_batch = [
        [MockScoredResult(EdgeType.id_for(edge_1_text), 0.92, payload={"text": edge_1_text})],
        [MockScoredResult(EdgeType.id_for(edge_2_text), 0.88, payload={"text": edge_2_text})],
    ]

    await graph.map_vector_distances_to_graph_nodes(
        node_distances=node_distances_batch, query_list_length=2
    )
    await graph.map_vector_distances_to_graph_edges(
        edge_distances=edge_distances_batch, query_list_length=2
    )

    assert node1.attributes.get("vector_distance") == [0.95, 6.5]
    assert node2.attributes.get("vector_distance") == [6.5, 0.87]
    assert edge.attributes.get("vector_distance") == [0.92, 0.88]


# ---------------------------------------------------------------------------
# Eval capture (SDK-529): retrieval.candidates events and bounding-setting notes.
# The vector engine and the memory fragment are faked exactly as above; the
# fragment returns real CogneeGraph Edge objects so the flat payload is built
# from the same attribute paths production uses. No LLM, no database.
# ---------------------------------------------------------------------------


def _capture_vector_engine(single=None, batch=None):
    """A vector engine whose per-collection results are given as dicts."""
    engine = AsyncMock()
    engine.embedding_engine = AsyncMock()
    engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    if single is not None:
        engine.search = AsyncMock(side_effect=lambda **kw: single.get(kw["collection_name"], []))
    if batch is not None:
        engine.batch_search = AsyncMock(
            side_effect=lambda **kw: batch.get(
                kw["collection_name"], [[] for _ in kw["query_texts"]]
            )
        )
    return engine


def _capture_edge(source, target, rel, distances):
    """A real Edge whose source, edge, and target carry per-query vector distances."""
    node1 = Node(source, {"name": source})
    node2 = Node(target, {"name": target})
    edge = Edge(node1, node2, attributes={"relationship_type": rel})
    for element, element_distances in zip((node1, edge, node2), distances):
        element.attributes["vector_distance"] = list(element_distances)
    return edge


def _capture_fragment(results):
    return AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=results),
    )


async def _capture_search(engine, fragment, **kwargs):
    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=fragment,
        ),
    ):
        return await brute_force_triplet_search(**kwargs)


def _candidate_events(sink):
    return [record for record in sink.records if record["kind"] == KIND_RETRIEVAL_CANDIDATES]


def _assert_flat(value):
    """Every leaf is a JSON scalar: no ScoredResult, Node, Edge, or other object leaks."""
    if isinstance(value, dict):
        for item in value.values():
            _assert_flat(item)
    elif isinstance(value, list):
        for item in value:
            _assert_flat(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), repr(value)


@pytest.mark.asyncio
async def test_capture_off_builds_no_retrieval_payload(monkeypatch, capture_reset):
    """(a) With capture off the search never constructs a payload or buffers an event."""
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    builder = MagicMock(side_effect=AssertionError("payload built while capture is off"))
    monkeypatch.setattr(bfts_module, "_retrieval_candidates_payload", builder)

    edge = _capture_edge("a", "b", "knows", ([0.1], [0.2], [0.3]))
    engine = _capture_vector_engine(single={"Entity_name": [MockScoredResult("a", 0.1)]})

    results = await _capture_search(engine, _capture_fragment([edge]), query="q", node_name=None)

    assert results == [edge]
    assert capture.is_active() is False
    builder.assert_not_called()
    assert not capture.hook._buffer


@pytest.mark.asyncio
async def test_sampled_out_search_records_notes_without_event(monkeypatch, fake_capture_sink):
    """(b) A run sampled out still records the bounding settings; no event, no payload."""
    capture.hook._configure(sample_rate=0.0)
    builder = MagicMock(side_effect=AssertionError("payload built for a sampled-out run"))
    monkeypatch.setattr(bfts_module, "_retrieval_candidates_payload", builder)

    edge = _capture_edge("a", "b", "knows", ([0.1], [0.2], [0.3]))
    engine = _capture_vector_engine(single={"Entity_name": [MockScoredResult("a", 0.1)]})

    with capture.run_scope(uuid4(), kind="operation") as scope:
        assert scope.sampled is False
        results = await _capture_search(
            engine,
            _capture_fragment([edge]),
            query="q",
            node_name=None,
            top_k=3,
            wide_search_top_k=7,
            neighborhood_seed_top_k=4,
            feedback_influence=0.2,
            collections=["Entity_name"],
        )

    assert results == [edge]
    assert scope.fields == {
        "retrieval.top_k": 3,
        "retrieval.wide_search_top_k": 7,
        "retrieval.neighborhood_seed_top_k": 4,
        "retrieval.feedback_influence": 0.2,
        "retrieval.mode": "single",
        "retrieval.collections": ["Entity_name", "EdgeType_relationship_name"],
    }
    builder.assert_not_called()
    await capture.drain()
    assert _candidate_events(fake_capture_sink) == []


@pytest.mark.asyncio
async def test_sampled_in_search_emits_flat_capped_candidates(fake_capture_sink):
    """(c) One flat event: scalar-only pool/top_k, pool capped at 500, cut_size == len(top_k)."""
    entity_rows = [MockScoredResult(f"n{i}", i / 1000) for i in range(600)]
    edge_rows = [MockScoredResult(f"e{i}", 0.5 + i / 100) for i in range(3)]
    engine = _capture_vector_engine(
        single={"Entity_name": entity_rows, "EdgeType_relationship_name": edge_rows}
    )
    edges = [
        _capture_edge("a", "b", "knows", ([0.1], [0.2], [0.3])),
        _capture_edge("b", "c", "likes", ([0.4], [0.5], [0.6])),
    ]

    results = await _capture_search(
        engine, _capture_fragment(edges), query="q", node_name=None, top_k=2
    )
    assert results == edges

    await capture.drain()
    [event] = _candidate_events(fake_capture_sink)
    assert event["stage"] == "brute_force_triplet_search"
    payload = event["payload"]
    assert set(payload) == {
        "query_index",
        "pool",
        "top_k",
        "pool_size",
        "pool_truncated",
        "cut_size",
    }
    _assert_flat(payload)
    assert payload["query_index"] == 0

    # Pool: 600 entity + 3 edge candidates seen, 500 kept.
    assert payload["pool_size"] == 603
    assert payload["pool_truncated"] is True
    assert len(payload["pool"]) == 500
    for entry in payload["pool"]:
        assert set(entry) == {"id", "collection", "score"}
        assert isinstance(entry["id"], str)
        assert isinstance(entry["collection"], str)
        assert isinstance(entry["score"], float)
    assert payload["pool"][0] == {"id": "n0", "collection": "Entity_name", "score": 0.0}
    # The cap keeps the head of every collection, not just the first 500 seen.
    edge_ids = [e["id"] for e in payload["pool"] if e["collection"] == "EdgeType_relationship_name"]
    assert edge_ids == ["e0", "e1", "e2"]

    # Cut: plain ids and the summed raw vector distance for this query.
    assert payload["cut_size"] == len(payload["top_k"]) == 2
    for entry in payload["top_k"]:
        assert set(entry) == {"source", "target", "rel", "score"}
    assert payload["top_k"][0] == {
        "source": "a",
        "target": "b",
        "rel": "knows",
        "score": pytest.approx(0.6),
    }
    assert payload["top_k"][1] == {
        "source": "b",
        "target": "c",
        "rel": "likes",
        "score": pytest.approx(1.5),
    }


@pytest.mark.asyncio
async def test_batch_search_emits_one_candidates_event_per_query(fake_capture_sink):
    """(d) Batch mode: one event per query index, each with its own pool and cut."""
    batch = {
        "Entity_name": [
            [MockScoredResult("n1", 0.1), MockScoredResult("n2", 0.2)],
            [MockScoredResult("n3", 0.3)],
        ],
        "EdgeType_relationship_name": [[MockScoredResult("e1", 0.4)], []],
    }
    engine = _capture_vector_engine(batch=batch)
    edge_a = _capture_edge("a", "b", "knows", ([0.1, 1.0], [0.2, 2.0], [0.3, 3.0]))
    edge_b = _capture_edge("b", "c", "likes", ([0.4, 4.0], [0.5, 5.0], [0.6, 6.0]))
    run_id = uuid4()

    with capture.run_scope(run_id, kind="operation") as scope:
        assert scope.sampled is True
        results = await _capture_search(
            engine,
            _capture_fragment([[edge_a], [edge_a, edge_b]]),
            query_batch=["q1", "q2"],
            collections=["Entity_name"],
        )

    assert results == [[edge_a], [edge_a, edge_b]]
    assert scope.fields["retrieval.mode"] == "batch"

    await capture.drain()
    events = _candidate_events(fake_capture_sink)
    assert [event["run_id"] for event in events] == [str(run_id), str(run_id)]
    first, second = (event["payload"] for event in events)
    _assert_flat(first)
    _assert_flat(second)

    assert first["query_index"] == 0
    assert first["pool"] == [
        {"id": "n1", "collection": "Entity_name", "score": 0.1},
        {"id": "n2", "collection": "Entity_name", "score": 0.2},
        {"id": "e1", "collection": "EdgeType_relationship_name", "score": 0.4},
    ]
    assert first["pool_size"] == 3
    assert first["pool_truncated"] is False
    assert first["top_k"] == [
        {"source": "a", "target": "b", "rel": "knows", "score": pytest.approx(0.6)}
    ]
    assert first["cut_size"] == 1

    assert second["query_index"] == 1
    assert second["pool"] == [{"id": "n3", "collection": "Entity_name", "score": 0.3}]
    assert second["pool_size"] == 1
    assert second["cut_size"] == 2
    # Scores are taken at this query's index of each element's distance list.
    assert second["top_k"][0]["score"] == pytest.approx(6.0)
    assert second["top_k"][1] == {
        "source": "b",
        "target": "c",
        "rel": "likes",
        "score": pytest.approx(15.0),
    }


@pytest.mark.asyncio
async def test_candidates_capture_failure_never_breaks_the_search(monkeypatch, fake_capture_sink):
    """A failing payload build is swallowed: the search returns normally, no event."""
    monkeypatch.setattr(
        bfts_module, "_retrieval_candidates_payload", MagicMock(side_effect=RuntimeError("boom"))
    )
    edge = _capture_edge("a", "b", "knows", ([0.1], [0.2], [0.3]))
    engine = _capture_vector_engine(single={"Entity_name": [MockScoredResult("a", 0.1)]})

    results = await _capture_search(engine, _capture_fragment([edge]), query="q", node_name=None)

    assert results == [edge]
    await capture.drain()
    assert _candidate_events(fake_capture_sink) == []


@pytest.mark.asyncio
async def test_missing_distances_fall_back_to_the_triplet_distance_penalty(fake_capture_sink):
    """Every ``score`` fallback path, which the well-formed fixtures never exercise.

    The penalty machinery exists because triplet endpoints frequently carry no
    distance for a given query, so a regression that hardcoded the default or
    returned 0.0 would silently corrupt the ranking-input score in every captured
    record. Covered here: a missing ``vector_distance`` key, a list too short for
    this query index, a non-numeric entry, and a pool row with no ``.score``.
    """
    edge = _capture_edge("a", "b", "knows", ([0.5], [1.5], [2.5]))
    del edge.node2.attributes["vector_distance"]  # missing key -> penalty
    edge.attributes["vector_distance"] = []  # too short for index 0 -> penalty

    short = _capture_edge("c", "d", "likes", ([0.25], [0.25], [0.25]))
    short.node1.attributes["vector_distance"] = ["not-a-number"]  # TypeError/ValueError -> penalty

    class _NoScore:
        id = "no-score"

    engine = _capture_vector_engine(single={"Entity_name": [_NoScore()]})

    results = await _capture_search(
        engine,
        _capture_fragment([edge, short]),
        query="q",
        node_name=None,
        triplet_distance_penalty=9.0,
    )
    assert results == [edge, short]

    await capture.drain()
    [event] = _candidate_events(fake_capture_sink)
    payload = event["payload"]

    # node1 keeps its 0.5; edge and node2 both fall back to the explicit 9.0.
    assert payload["top_k"][0]["score"] == pytest.approx(0.5 + 9.0 + 9.0)
    # node1's entry is non-numeric -> penalty; the other two are real.
    assert payload["top_k"][1]["score"] == pytest.approx(9.0 + 0.25 + 0.25)
    # A pool row with no .score falls back to the same penalty.
    assert payload["pool"] == [{"id": "no-score", "collection": "Entity_name", "score": 9.0}]


@pytest.mark.asyncio
async def test_the_default_penalty_is_used_when_none_is_passed(fake_capture_sink):
    """``triplet_distance_penalty=None`` means the 6.5 retrieval default, not 0.0."""
    edge = _capture_edge("a", "b", "knows", ([0.5], [1.5], [2.5]))
    del edge.node2.attributes["vector_distance"]
    engine = _capture_vector_engine(single={"Entity_name": [MockScoredResult("a", 0.1)]})

    await _capture_search(
        engine,
        _capture_fragment([edge]),
        query="q",
        node_name=None,
        triplet_distance_penalty=None,
    )

    await capture.drain()
    [event] = _candidate_events(fake_capture_sink)
    assert event["payload"]["top_k"][0]["score"] == pytest.approx(0.5 + 1.5 + 6.5)
