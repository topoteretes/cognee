"""BROAD search type: graph completion cut by the LLM's context window (SDK-324)."""

from types import SimpleNamespace

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import (
    ALL_TRIPLETS,
    BroadRetriever,
    resolve_context_window,
)
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType

QUERY = "Who has the most issues assigned?"


@pytest.fixture
def llm_config(monkeypatch):
    config = SimpleNamespace(llm_model="gpt-4o-mini", llm_max_completion_tokens=1000)
    monkeypatch.setattr(broad_retriever, "get_llm_context_config", lambda: config)
    return config


def _chunk_triplets(count: int, words_per_chunk: int = 200) -> list[Edge]:
    """Best-first triplets, each bringing in its own chunk-sized node."""
    document = Node("doc", {"name": "document"})
    return [
        Edge(
            Node(f"chunk-{i}", {"text": f"row {i} " + "assignee " * words_per_chunk}),
            document,
            {"relationship_name": "is_part_of"},
        )
        for i in range(count)
    ]


def test_context_window_comes_from_the_configured_model(llm_config):
    assert resolve_context_window() == 128000


def test_explicit_context_window_wins(llm_config):
    assert resolve_context_window(4096) == 4096


def test_unknown_model_fails_fast(llm_config):
    llm_config.llm_model = "not-a-real-provider/not-a-real-model"
    with pytest.raises(ValueError, match="context_window_tokens"):
        resolve_context_window()


def test_budget_leaves_room_for_prompt_output_and_margin(llm_config):
    retriever = BroadRetriever(context_window_tokens=10000)
    budget = retriever.context_token_budget(QUERY)
    # 1000 output tokens + 5% margin (500) + a non-empty prompt shell.
    assert 0 < budget < 10000 - 1000 - 500


def test_window_too_small_for_the_prompt_fails_fast(llm_config):
    retriever = BroadRetriever(context_window_tokens=1000)
    with pytest.raises(ValueError, match="no room"):
        retriever.context_token_budget(QUERY)


@pytest.mark.asyncio
async def test_keeps_the_longest_best_first_prefix_that_fits(llm_config):
    retriever = BroadRetriever(context_window_tokens=4000)
    triplets = _chunk_triplets(40)

    kept = await retriever.fit_to_window(QUERY, triplets)

    budget = retriever.context_token_budget(QUERY)
    assert 0 < len(kept) < len(triplets)
    assert kept == triplets[: len(kept)]
    assert retriever.count_tokens(await retriever.resolve_edges_to_text(kept)) <= budget
    one_more = triplets[: len(kept) + 1]
    assert retriever.count_tokens(await retriever.resolve_edges_to_text(one_more)) > budget


@pytest.mark.asyncio
async def test_everything_is_kept_when_the_graph_fits(llm_config):
    retriever = BroadRetriever(context_window_tokens=200000)
    triplets = _chunk_triplets(40)

    assert await retriever.fit_to_window(QUERY, triplets) == triplets


@pytest.mark.asyncio
async def test_retrieval_is_cut_per_query(llm_config, monkeypatch):
    triplets = _chunk_triplets(40)

    async def rank_everything(self, query=None, query_batch=None):
        return [triplets, triplets[:3]] if query_batch else triplets

    monkeypatch.setattr(
        broad_retriever.GraphCompletionRetriever, "get_retrieved_objects", rank_everything
    )
    retriever = BroadRetriever(context_window_tokens=4000)

    single = await retriever.get_retrieved_objects(query=QUERY)
    batch = await retriever.get_retrieved_objects(query_batch=[QUERY, QUERY])

    assert 0 < len(single) < len(triplets)
    assert batch == [single, triplets[:3]]


@pytest.mark.asyncio
async def test_search_ignores_top_k_and_ranks_the_whole_graph():
    retriever = await get_search_type_retriever_instance(
        SearchType.BROAD,
        QUERY,
        top_k=3,
        retriever_specific_config={"context_window_tokens": 50000},
    )

    assert isinstance(retriever, BroadRetriever)
    assert retriever.top_k == ALL_TRIPLETS
    assert retriever.wide_search_top_k is None
    assert retriever.context_window_tokens == 50000
