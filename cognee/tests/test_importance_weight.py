import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine


class MockEdge:
    def __init__(self, node1, node2, relationship):
        self.node1 = node1
        self.node2 = node2
        self.relationship = relationship
        self.attributes = {}


class MockNode:
    def __init__(self, node_id, node_type, importance_weight=0.5):
        self.id = node_id
        self.type = node_type
        self.attributes = {"importance_weight": importance_weight}


@pytest.fixture
def mock_graph_engine():
    with patch('cognee.infrastructure.databases.graph.get_graph_engine') as mock:
        mock.return_value = AsyncMock()
        mock.return_value.is_empty.return_value = False
        yield mock.return_value


@pytest.fixture
def mock_vector_engine():
    with patch('cognee.infrastructure.databases.vector.get_vector_engine') as mock:
        mock.return_value = MagicMock()
        mock.return_value.embedding_engine.embed_text.return_value = [[0.1] * 768]  # 模拟嵌入向量
        mock.return_value.search.return_value = []
        yield mock.return_value


@pytest.fixture
def mock_memory_fragment():
    node1 = MockNode("node1", "Entity", importance_weight=0.8)
    node2 = MockNode("node2", "Entity", importance_weight=0.9)
    node3 = MockNode("node3", "Entity", importance_weight=0.3)
    node4 = MockNode("node4", "Entity", importance_weight=0.7)

    edge1 = MockEdge(node1, node2, "related_to")
    edge2 = MockEdge(node3, node4, "related_to")

    class MockMemoryFragment:
        async def calculate_top_triplet_importances(self, k):
            return [edge1, edge2]

        async def map_vector_distances_to_graph_nodes(self, node_distances):
            pass

        async def map_vector_distances_to_graph_edges(self, vector_engine, query_vector, edge_distances):
            pass

    return MockMemoryFragment()


@pytest.mark.asyncio
async def test_importance_weight_in_scoring(mock_vector_engine, mock_graph_engine, mock_memory_fragment):
    with patch('cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment',
               return_value=mock_memory_fragment):
        query = "query test"

        retriever = GraphCompletionRetriever(top_k=2)
        triplets = await retriever.get_triplets(query)
        assert len(triplets) == 2

        first_edge = triplets[0]
        second_edge = triplets[1]

        first_avg_weight = (first_edge.node1.attributes["importance_weight"] +
                            first_edge.node2.attributes["importance_weight"]) / 2

        assert abs(first_avg_weight - 0.85) < 0.01

        second_avg_weight = (second_edge.node1.attributes["importance_weight"] +
                             second_edge.node2.attributes["importance_weight"]) / 2
        assert abs(second_avg_weight - 0.5) < 0.01


@pytest.mark.asyncio
async def test_importance_weight_default_value():
    node1 = MockNode("node1", "Entity")
    node2 = MockNode("node2", "Entity")

    node1.attributes.pop("importance_weight", None)
    node2.attributes.pop("importance_weight", None)

    edge = MockEdge(node1, node2, "related_to")

    class MockMemoryFragment:
        async def calculate_top_triplet_importances(self, k):
            return [edge]

        async def map_vector_distances_to_graph_nodes(self, node_distances):
            pass

        async def map_vector_distances_to_graph_edges(self, vector_engine, query_vector, edge_distances):
            pass

    with patch('cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment',
               return_value=MockMemoryFragment()):
        retriever = GraphCompletionRetriever()
        triplets = await retriever.get_triplets("query test")

        assert len(triplets) == 1
        assert triplets[0] == edge

        assert "importance_weight" not in triplets[0].node1.attributes
        assert "importance_weight" not in triplets[0].node2.attributes


@pytest.mark.asyncio
async def test_importance_weight_edge_cases():
    node1 = MockNode("node1", "Entity", importance_weight=0.0)
    node2 = MockNode("node2", "Entity", importance_weight=1.0)

    edge = MockEdge(node1, node2, "related_to")

    class MockMemoryFragment:
        async def calculate_top_triplet_importances(self, k):
            return [edge]

        async def map_vector_distances_to_graph_nodes(self, node_distances):
            pass

        async def map_vector_distances_to_graph_edges(self, vector_engine, query_vector, edge_distances):
            pass

    with patch('cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment',
               return_value=MockMemoryFragment()):
        retriever = GraphCompletionRetriever()
        triplets = await retriever.get_triplets("query test")

        assert len(triplets) == 1
        assert triplets[0].node1.attributes["importance_weight"] == 0.0
        assert triplets[0].node2.attributes["importance_weight"] == 1.0

        avg_weight = (0.0 + 1.0) / 2
        assert abs(avg_weight - 0.5) < 0.01


@pytest.fixture
def mock_memory_fragment_for_ranking():
    node_a1 = MockNode("A1", "Entity", importance_weight=1.0)
    node_a2 = MockNode("A2", "Entity", importance_weight=1.0)
    edge_a = MockEdge(node_a1, node_a2, "high_weight", score=0.9)

    node_b1 = MockNode("B1", "Entity", importance_weight=0.1)
    node_b2 = MockNode("B2", "Entity", importance_weight=0.1)
    edge_b = MockEdge(node_b1, node_b2, "low_weight", score=0.5)

    expected_ranking = [edge_a, edge_b]

    class MockMemoryFragment:
        async def calculate_top_triplet_importances(self, k):
            return expected_ranking

        async def map_vector_distances_to_graph_nodes(self, node_distances):
            pass

        async def map_vector_distances_to_graph_edges(self, vector_engine, query_vector, edge_distances):
            pass

    return MockMemoryFragment()


@pytest.mark.asyncio
async def test_importance_weight_ranking_override(mock_vector_engine, mock_graph_engine,
                                                  mock_memory_fragment_for_ranking):
    with patch('cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment',
               return_value=mock_memory_fragment_for_ranking):
        query = "ranking test"
        retriever = GraphCompletionRetriever(top_k=2)
        triplets = await retriever.get_triplets(query)

        assert len(triplets) == 2

        assert triplets[0].node1.attributes["importance_weight"] == 1.0
        assert triplets[0].relationship == "high_weight"

        assert triplets[1].node1.attributes["importance_weight"] == 0.1
        assert triplets[1].relationship == "low_weight"

        assert triplets[0].score > triplets[1].score
        assert abs(triplets[0].score - 0.9) < 0.01
        assert abs(triplets[1].score - 0.5) < 0.01