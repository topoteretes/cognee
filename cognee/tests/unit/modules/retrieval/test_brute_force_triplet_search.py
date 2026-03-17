import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
    get_memory_fragment,
    format_triplets,
)
from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            k=custom_top_k, query_list_length=None
        )


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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine"
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine",
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
        [MockScoredResult(generate_edge_id(edge_1_text), 0.92, payload={"text": edge_1_text})],
        [MockScoredResult(generate_edge_id(edge_2_text), 0.88, payload={"text": edge_2_text})],
    ]

    await graph.map_vector_distances_to_graph_nodes(
        node_distances=node_distances_batch, query_list_length=2
    )
    await graph.map_vector_distances_to_graph_edges(
        edge_distances=edge_distances_batch, query_list_length=2
    )

    assert node1.attributes.get("vector_distance") == [0.95, 3.5]
    assert node2.attributes.get("vector_distance") == [3.5, 0.87]
    assert edge.attributes.get("vector_distance") == [0.92, 0.88]
