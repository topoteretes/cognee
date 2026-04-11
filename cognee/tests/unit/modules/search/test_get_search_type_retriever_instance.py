import pytest

from cognee.modules.search.exceptions import UnsupportedSearchTypeError
from cognee.modules.search.types import SearchType
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_decomposition_retriever import (
    DecompositionMode,
    GraphCompletionDecompositionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_type", "expected_class"),
    [
        (SearchType.GRAPH_COMPLETION, GraphCompletionRetriever),
        (SearchType.GRAPH_COMPLETION_DECOMPOSITION, GraphCompletionDecompositionRetriever),
        (SearchType.GRAPH_COMPLETION_COT, GraphCompletionCotRetriever),
        (
            SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
            GraphCompletionContextExtensionRetriever,
        ),
        (SearchType.GRAPH_SUMMARY_COMPLETION, GraphSummaryCompletionRetriever),
        (SearchType.TEMPORAL, TemporalRetriever),
    ],
)
async def test_graph_search_retrievers_receive_feedback_influence(search_type, expected_class):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    retriever_instance = await mod.get_search_type_retriever_instance(
        search_type,
        query_text="q",
        feedback_influence=0.4,
    )

    assert isinstance(retriever_instance, expected_class)
    assert retriever_instance.feedback_influence == 0.4


@pytest.mark.asyncio
async def test_graph_search_retrievers_default_triplet_penalty_is_updated():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    for search_type in [
        SearchType.GRAPH_COMPLETION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.TEMPORAL,
    ]:
        retriever_instance = await mod.get_search_type_retriever_instance(
            search_type, query_text="q"
        )
        assert retriever_instance.triplet_distance_penalty == 6.5


@pytest.mark.asyncio
async def test_graph_completion_decomposition_uses_retriever_specific_mode():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        query_text="q",
        retriever_specific_config={"decomposition_mode": "combined_triplets_context"},
    )

    assert isinstance(retriever_instance, GraphCompletionDecompositionRetriever)
    assert retriever_instance.decomposition_mode is DecompositionMode.COMBINED_TRIPLETS_CONTEXT


@pytest.mark.asyncio
async def test_graph_completion_decomposition_defaults_to_answer_per_subquery():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    retriever_instance = await mod.get_search_type_retriever_instance(
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        query_text="q",
    )

    assert isinstance(retriever_instance, GraphCompletionDecompositionRetriever)
    assert retriever_instance.decomposition_mode is DecompositionMode.ANSWER_PER_SUBQUERY
