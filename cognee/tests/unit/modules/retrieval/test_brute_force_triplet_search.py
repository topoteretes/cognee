import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
    get_memory_fragment,
)
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError


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
    with pytest.raises(ValueError, match="The query must be a non-empty string."):
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        await brute_force_triplet_search(query="test")

        expected_collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
async def test_brute_force_triplet_search_all_collections_empty():
    """Test that empty list is returned when all collections return no results."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=mock_fragment,
        ),
    ):
        custom_top_k = 15
        await brute_force_triplet_search(query="test", top_k=custom_top_k, node_name=["n"])

        mock_fragment.calculate_top_triplet_importances.assert_called_once_with(k=custom_top_k)


@pytest.mark.asyncio
async def test_get_memory_fragment_returns_empty_graph_on_entity_not_found():
    """Test that get_memory_fragment returns empty graph when entity not found."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.project_graph_from_db = AsyncMock(
        side_effect=EntityNotFoundError("Entity not found")
    )

    with patch(
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
        return_value=mock_graph_engine,
    ):
        fragment = await get_memory_fragment()

        assert isinstance(fragment, CogneeGraph)
        assert len(fragment.nodes) == 0


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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine",
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
