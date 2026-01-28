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
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.chunks_retriever import ChunksRetriever

    async def _fake_select_search_type(query_text: str):
        assert query_text == "hello"
        return SearchType.CHUNKS

    monkeypatch.setattr(mod, "select_search_type", _fake_select_search_type)

    tools = await mod.get_search_type_tools(SearchType.FEELING_LUCKY, query_text="hello")

    assert len(tools) == 2
    assert all(callable(t) for t in tools)
    assert tools[0].__name__ == "get_completion"
    assert tools[1].__name__ == "get_context"
    assert tools[0].__self__.__class__ is ChunksRetriever
    assert tools[1].__self__.__class__ is ChunksRetriever


@pytest.mark.asyncio
async def test_disallowed_cypher_search_types_raise(monkeypatch):
    import cognee.modules.search.methods.get_search_type_tools as mod

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "false")

    with pytest.raises(UnsupportedSearchTypeError, match="disabled"):
        await mod.get_search_type_tools(SearchType.CYPHER, query_text="MATCH (n) RETURN n")

    with pytest.raises(UnsupportedSearchTypeError, match="disabled"):
        await mod.get_search_type_tools(SearchType.NATURAL_LANGUAGE, query_text="Find nodes")


@pytest.mark.asyncio
async def test_allowed_cypher_search_types_return_tools(monkeypatch):
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.cypher_search_retriever import CypherSearchRetriever

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")

    tools = await mod.get_search_type_tools(SearchType.CYPHER, query_text="q")
    assert len(tools) == 2
    assert tools[0].__name__ == "get_completion"
    assert tools[1].__name__ == "get_context"
    assert tools[0].__self__.__class__ is CypherSearchRetriever
    assert tools[1].__self__.__class__ is CypherSearchRetriever


@pytest.mark.asyncio
async def test_registered_community_retriever_is_used(monkeypatch):
    """
    Integration point: community retrievers are loaded from the registry module and should
    override the default mapping when present.
    """
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval import registered_community_retrievers as registry

    monkeypatch.setattr(
        registry,
        "registered_community_retrievers",
        {SearchType.SUMMARIES: _DummyCommunityRetriever},
    )

    tools = await mod.get_search_type_tools(SearchType.SUMMARIES, query_text="q", top_k=7)

    assert len(tools) == 2
    assert tools[0].__self__.__class__ is _DummyCommunityRetriever
    assert tools[0].__self__.kwargs["top_k"] == 7
    assert tools[1].__self__.__class__ is _DummyCommunityRetriever
    assert tools[1].__self__.kwargs["top_k"] == 7


@pytest.mark.asyncio
async def test_unknown_query_type_raises_unsupported():
    import cognee.modules.search.methods.get_search_type_tools as mod

    with pytest.raises(UnsupportedSearchTypeError, match="UNKNOWN_TYPE"):
        await mod.get_search_type_tools("UNKNOWN_TYPE", query_text="q")


@pytest.mark.asyncio
async def test_default_mapping_passes_top_k_to_retrievers():
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.summaries_retriever import SummariesRetriever

    tools = await mod.get_search_type_tools(SearchType.SUMMARIES, query_text="q", top_k=4)
    assert len(tools) == 2
    assert tools[0].__self__.__class__ is SummariesRetriever
    assert tools[1].__self__.__class__ is SummariesRetriever
    assert tools[0].__self__.top_k == 4
    assert tools[1].__self__.top_k == 4


@pytest.mark.asyncio
async def test_chunks_lexical_returns_jaccard_tools():
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.jaccard_retrival import JaccardChunksRetriever

    tools = await mod.get_search_type_tools(SearchType.CHUNKS_LEXICAL, query_text="q", top_k=3)
    assert len(tools) == 2
    assert tools[0].__self__.__class__ is JaccardChunksRetriever
    assert tools[1].__self__.__class__ is JaccardChunksRetriever
    assert tools[0].__self__ is tools[1].__self__


@pytest.mark.asyncio
async def test_coding_rules_uses_node_name_as_rules_nodeset_name():
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.coding_rules_retriever import CodingRulesRetriever

    tools = await mod.get_search_type_tools(SearchType.CODING_RULES, query_text="q", node_name=[])
    assert len(tools) == 1
    assert tools[0].__name__ == "get_existing_rules"
    assert tools[0].__self__.__class__ is CodingRulesRetriever

    assert tools[0].__self__.rules_nodeset_name == ["coding_agent_rules"]


@pytest.mark.asyncio
async def test_feedback_uses_last_k():
    import cognee.modules.search.methods.get_search_type_tools as mod
    from cognee.modules.retrieval.user_qa_feedback import UserQAFeedback

    tools = await mod.get_search_type_tools(SearchType.FEEDBACK, query_text="q", last_k=11)
    assert len(tools) == 1
    assert tools[0].__name__ == "add_feedback"
    assert tools[0].__self__.__class__ is UserQAFeedback
    assert tools[0].__self__.last_k == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_type, expected_class_name, expected_method_names",
    [
        (SearchType.CHUNKS, "ChunksRetriever", ("get_completion", "get_context")),
        (SearchType.RAG_COMPLETION, "CompletionRetriever", ("get_completion", "get_context")),
        (SearchType.TRIPLET_COMPLETION, "TripletRetriever", ("get_completion", "get_context")),
        (
            SearchType.TRIPLET_COMPLETION_CACHE,
            "CacheTripletRetriever",
            ("get_completion", "get_context"),
        ),
        (
            SearchType.GRAPH_COMPLETION,
            "GraphCompletionRetriever",
            ("get_completion", "get_context"),
        ),
        (
            SearchType.GRAPH_COMPLETION_COT,
            "GraphCompletionCotRetriever",
            ("get_completion", "get_context"),
        ),
        (
            SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
            "GraphCompletionContextExtensionRetriever",
            ("get_completion", "get_context"),
        ),
        (
            SearchType.GRAPH_SUMMARY_COMPLETION,
            "GraphSummaryCompletionRetriever",
            ("get_completion", "get_context"),
        ),
        (SearchType.TEMPORAL, "TemporalRetriever", ("get_completion", "get_context")),
        (
            SearchType.NATURAL_LANGUAGE,
            "NaturalLanguageRetriever",
            ("get_completion", "get_context"),
        ),
    ],
)
async def test_tool_construction_for_supported_search_types(
    monkeypatch, query_type, expected_class_name, expected_method_names
):
    import cognee.modules.search.methods.get_search_type_tools as mod

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")

    tools = await mod.get_search_type_tools(query_type, query_text="q")

    assert len(tools) == 2
    assert tools[0].__name__ == expected_method_names[0]
    assert tools[1].__name__ == expected_method_names[1]
    assert tools[0].__self__.__class__.__name__ == expected_class_name
    assert tools[1].__self__.__class__.__name__ == expected_class_name


@pytest.mark.asyncio
async def test_some_completion_tools_are_callable_without_backends(monkeypatch):
    """
    "Making search tools" should include that the returned callables are usable.
    For retrievers that accept an explicit `context`, we can call get_completion without touching
    DB/LLM backends.
    """
    import cognee.modules.search.methods.get_search_type_tools as mod

    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")

    for query_type in [
        SearchType.CHUNKS,
        SearchType.SUMMARIES,
        SearchType.CYPHER,
        SearchType.NATURAL_LANGUAGE,
    ]:
        tools = await mod.get_search_type_tools(query_type, query_text="q")
        completion = tools[0]
        result = await completion("q", context=["ok"])
        assert result == ["ok"]
