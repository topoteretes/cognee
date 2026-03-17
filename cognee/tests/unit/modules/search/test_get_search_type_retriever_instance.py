import pytest

from cognee.modules.search.exceptions import UnsupportedSearchTypeError
from cognee.modules.search.types import SearchType


class _DummyCommunityRetriever:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_completion(self, *args, **kwargs):
        return {"kind": "completion", "init": self.kwargs, "args": args, "kwargs": kwargs}

    def get_context(self, *args, **kwargs):
        return {"kind": "context", "init": self.kwargs, "args": args, "kwargs": kwargs}


@pytest.mark.asyncio
async def test_feeling_lucky_delegates_to_select_search_type(monkeypatch):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.chunks_retriever import ChunksRetriever

    async def _fake_select_search_type(query_text: str):
        assert query_text == "hello"
        return SearchType.CHUNKS

    monkeypatch.setattr(mod, "select_search_type", _fake_select_search_type)

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.FEELING_LUCKY, query_text="hello"
    )
    assert isinstance(retriever_instance, ChunksRetriever)


@pytest.mark.asyncio
async def test_disallowed_cypher_search_types_raise(monkeypatch):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "false")

    with pytest.raises(UnsupportedSearchTypeError, match="disabled"):
        await mod.get_search_type_retriever_instance(
            SearchType.CYPHER, query_text="MATCH (n) RETURN n"
        )

    with pytest.raises(UnsupportedSearchTypeError, match="disabled"):
        await mod.get_search_type_retriever_instance(
            SearchType.NATURAL_LANGUAGE, query_text="Find nodes"
        )


@pytest.mark.asyncio
async def test_allowed_cypher_search_types_return_tools(monkeypatch):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.cypher_search_retriever import CypherSearchRetriever

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.CYPHER, query_text="q"
    )
    assert isinstance(retriever_instance, CypherSearchRetriever)


@pytest.mark.asyncio
async def test_registered_community_retriever_is_used(monkeypatch):
    """
    Integration point: community retrievers are loaded from the registry module and should
    override the default mapping when present.
    """
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval import registered_community_retrievers as registry

    monkeypatch.setattr(
        registry,
        "registered_community_retrievers",
        {SearchType.SUMMARIES: _DummyCommunityRetriever},
    )

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.SUMMARIES, query_text="q", top_k=7
    )

    assert isinstance(retriever_instance, _DummyCommunityRetriever)


@pytest.mark.asyncio
async def test_unknown_query_type_raises_unsupported():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    with pytest.raises(UnsupportedSearchTypeError, match="UNKNOWN_TYPE"):
        await mod.get_search_type_retriever_instance("UNKNOWN_TYPE", query_text="q")


@pytest.mark.asyncio
async def test_default_mapping_passes_top_k_to_retrievers():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.summaries_retriever import SummariesRetriever

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.SUMMARIES, query_text="q", top_k=4
    )
    assert isinstance(retriever_instance, SummariesRetriever)


@pytest.mark.asyncio
async def test_chunks_lexical_returns_jaccard_tools():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.CHUNKS_LEXICAL, query_text="q", top_k=3
    )
    assert isinstance(retriever_instance, JaccardChunksRetriever)


@pytest.mark.asyncio
async def test_coding_rules_uses_node_name_as_rules_nodeset_name():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.coding_rules_retriever import CodingRulesRetriever

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.CODING_RULES, query_text="q", node_name=[]
    )
    assert isinstance(retriever_instance, CodingRulesRetriever)
