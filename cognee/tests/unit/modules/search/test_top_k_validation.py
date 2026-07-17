import pytest

from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType


# ── invalid top_k values are rejected at the factory level ────────────────────


@pytest.mark.asyncio
async def test_top_k_zero_raises_query_validation_error():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    with pytest.raises(QueryValidationError, match="top_k must be a positive integer"):
        await mod.get_search_type_retriever_instance(SearchType.CHUNKS, query_text="test", top_k=0)


@pytest.mark.asyncio
async def test_top_k_negative_one_raises_query_validation_error():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    with pytest.raises(QueryValidationError, match="top_k must be a positive integer"):
        await mod.get_search_type_retriever_instance(SearchType.CHUNKS, query_text="test", top_k=-1)


@pytest.mark.asyncio
async def test_top_k_large_negative_raises_query_validation_error():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    with pytest.raises(QueryValidationError, match="top_k must be a positive integer"):
        await mod.get_search_type_retriever_instance(
            SearchType.SUMMARIES, query_text="test", top_k=-100
        )


# ── guard fires before any retriever is instantiated ─────────────────────────


@pytest.mark.asyncio
async def test_top_k_zero_raises_before_retriever_instantiated(monkeypatch):
    """The QueryValidationError is raised in the factory, not inside the retriever.
    We verify this by asserting the guard fires even for graph-based search types."""
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    with pytest.raises(QueryValidationError, match="top_k must be a positive integer"):
        await mod.get_search_type_retriever_instance(
            SearchType.GRAPH_COMPLETION, query_text="test", top_k=0
        )


# ── valid top_k values pass through ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_positive_top_k_does_not_raise():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.chunks_retriever import ChunksRetriever

    retriever = await mod.get_search_type_retriever_instance(
        SearchType.CHUNKS, query_text="test", top_k=5
    )
    assert isinstance(retriever, ChunksRetriever)
    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_top_k_none_does_not_raise():
    """None is allowed — individual retrievers fall back to their own defaults."""
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.summaries_retriever import SummariesRetriever

    retriever = await mod.get_search_type_retriever_instance(
        SearchType.SUMMARIES, query_text="test", top_k=None
    )
    assert isinstance(retriever, SummariesRetriever)
