from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.hybrid.results import ScoredResult

class MockHit:
    def __init__(self, id_val, score):
        self.id = id_val
        self.score = score
        self.payload = {"id": id_val, "name": id_val}

@pytest.mark.asyncio
async def test_retrieve_entities_and_facts_truth_reranking():
    retriever = HybridRetriever(
        entities_top_k=3,
        facts_top_k=3,
        use_truth_weight=True,
    )
    retriever._unified_engine = MagicMock()
    retriever._unified_engine.vector = MagicMock()

    # We mock search_collection to return entity hits
    # The original rank is:
    # 0. "stale" (index 0, original_score = 1.0)
    # 1. "poor" (index 1, original_score = 0.5)
    # 2. "perfect" (index 2, original_score = 0.333)
    entity_hits = [MockHit("stale", 0.9), MockHit("poor", 0.8), MockHit("perfect", 0.7)]
    edge_hits = [MockHit("edge_stale", 0.9), MockHit("edge_poor", 0.8), MockHit("edge_perfect", 0.7)]

    # Mock get_node_truth_state
    # "stale": epoch mismatch -> no reranking, final_score = 1.0
    # "poor": epoch match, bad alignment -> truth_factor = 0.75, final_score = 0.5 * 0.75 = 0.375
    # "perfect": epoch match, perfect alignment -> truth_factor = 1.25, final_score = 0.333 * 1.25 = 0.416
    truth_state_data = {
        "stale": {"truth_alignment": [1.0], "truth_epoch": 1},
        "poor": {"truth_alignment": [0.0], "truth_epoch": 3},
        "perfect": {"truth_alignment": [1.0], "truth_epoch": 3},
        "edge_stale": {"truth_alignment": [1.0], "truth_epoch": 1},
        "edge_poor": {"truth_alignment": [0.0], "truth_epoch": 3},
        "edge_perfect": {"truth_alignment": [1.0], "truth_epoch": 3},
    }

    retriever._unified_engine.graph.get_node_truth_state = AsyncMock(return_value=truth_state_data)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_collection",
            AsyncMock(side_effect=[entity_hits, edge_hits]),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_entities",
            AsyncMock(side_effect=lambda graph, hits, max_edges, edge_ranks: [getattr(hit, "id") for hit in hits]),
        ),
        patch.object(
            retriever,
            "_select_facts",
            side_effect=lambda edge_hits, entities: [getattr(hit, "id") for hit in edge_hits],
        ),
    ):
        entities, facts = await retriever._retrieve_entities_and_facts(
            query="test query",
            query_vector=[1.0],
            use_truth_weight=True,
            q_coords=[1.0],
            current_truth_epoch=3,
        )

    # After reranking, order of (stale, poor, perfect) should be:
    # 1. stale (final_score = 1.0)
    # 2. perfect (final_score = 0.416)
    # 3. poor (final_score = 0.375)
    # So order of entities should be: ["stale", "perfect", "poor"]
    assert entities == ["stale", "perfect", "poor"]
    assert facts == ["edge_stale", "edge_perfect", "edge_poor"]
