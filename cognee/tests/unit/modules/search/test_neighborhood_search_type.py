import pytest

import cognee.modules.retrieval.neighborhood_retriever as nr_module
from cognee.modules.retrieval.neighborhood_retriever import NeighborhoodRetriever
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType


@pytest.mark.asyncio
async def test_neighborhood_search_type_registry():
    retriever = await get_search_type_retriever_instance(
        query_type=SearchType.NEIGHBORHOOD,
        query_text="Acme",
        retriever_specific_config={
            "depth": 2,
            "seed_top_k": 3,
            "edge_types": ["works_at"],
        },
    )

    assert isinstance(retriever, NeighborhoodRetriever)
    assert retriever.neighborhood_depth == 2
    assert retriever.neighborhood_seed_top_k == 3
    assert retriever.edge_types == ["works_at"]


@pytest.mark.asyncio
async def test_neighborhood_retriever_passes_neighborhood_args(monkeypatch):
    seen = {}

    async def fake_brute_force(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(nr_module, "brute_force_triplet_search", fake_brute_force)

    retriever = NeighborhoodRetriever(depth=2, seed_top_k=3, edge_types=["works_at"])
    retriever._get_vector_index_collections = lambda: []
    retriever.top_k = 5
    retriever.node_type = None
    retriever.node_name = None
    retriever.node_name_filter_operator = "OR"
    retriever.wide_search_top_k = 100
    retriever.triplet_distance_penalty = 6.5
    retriever.feedback_influence = 0.0

    await retriever.get_triplets(query="Acme")

    assert seen["neighborhood_depth"] == 2
    assert seen["neighborhood_seed_top_k"] == 3
    assert seen["edge_types"] == ["works_at"]
