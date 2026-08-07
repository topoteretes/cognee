import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_search import can_run_as_concurrent_turn
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType


def supports(retriever_type: type, **overrides):
    inputs = {
        "original_search_type": SearchType.RAG_COMPLETION,
        "retriever_type": retriever_type,
        "session_available": True,
        "is_batch": False,
        "only_context": False,
    }
    inputs.update(overrides)
    return can_run_as_concurrent_turn(**inputs)


@pytest.mark.parametrize(
    "retriever_type",
    [
        CompletionRetriever,
        GraphCompletionRetriever,
        GraphSummaryCompletionRetriever,
        HybridRetriever,
        TripletRetriever,
    ],
)
def test_latency_mode_supports_only_designated_retrievers(retriever_type):
    assert supports(retriever_type) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"original_search_type": SearchType.FEELING_LUCKY},
        {"session_available": False},
        {"is_batch": True},
        {"only_context": True},
    ],
)
def test_latency_mode_falls_back_to_accuracy(overrides):
    assert supports(GraphCompletionRetriever, **overrides) is False


def test_latency_mode_accepts_direct_retriever_calls_without_search_type():
    assert supports(GraphCompletionRetriever, original_search_type=None) is True


def test_latency_mode_rejects_subclasses_and_unrelated_retrievers():
    class ExtendedGraphRetriever(GraphCompletionRetriever):
        pass

    assert supports(ExtendedGraphRetriever) is False
    assert supports(object) is False
