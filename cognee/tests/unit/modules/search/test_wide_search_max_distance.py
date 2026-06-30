import pytest
from cognee.modules.search.types import SearchType
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_decomposition_retriever import GraphCompletionDecompositionRetriever
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import GraphCompletionContextExtensionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import GraphSummaryCompletionRetriever
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_type", "expected_class"),
    [
        (SearchType.GRAPH_COMPLETION, GraphCompletionRetriever),
        (SearchType.GRAPH_COMPLETION_DECOMPOSITION, GraphCompletionDecompositionRetriever),
        (SearchType.GRAPH_COMPLETION_COT, GraphCompletionCotRetriever),
        (SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION, GraphCompletionContextExtensionRetriever),
        (SearchType.GRAPH_SUMMARY_COMPLETION, GraphSummaryCompletionRetriever),
        (SearchType.TEMPORAL, TemporalRetriever),
    ],
)
async def test_graph_search_retrievers_accept_wide_search_max_distance(search_type, expected_class):
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    retriever_instance = await mod.get_search_type_retriever_instance(
        search_type,
        query_text="q",
        wide_search_max_distance=0.8,
    )
    assert isinstance(retriever_instance, expected_class)
    assert retriever_instance.wide_search_max_distance == 0.8
