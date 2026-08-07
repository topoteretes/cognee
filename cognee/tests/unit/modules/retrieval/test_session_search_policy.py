import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_search import (
    ACCURACY_OPTIMIZED,
    LATENCY_OPTIMIZED,
    resolve_session_search_mode,
)
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType


def resolve(retriever_type: type, **overrides):
    inputs = {
        "configured_mode": LATENCY_OPTIMIZED,
        "original_search_type": SearchType.RAG_COMPLETION,
        "retriever_type": retriever_type,
        "session_available": True,
        "is_batch": False,
        "only_context": False,
    }
    inputs.update(overrides)
    return resolve_session_search_mode(**inputs)


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
    assert resolve(retriever_type) == LATENCY_OPTIMIZED


@pytest.mark.parametrize(
    "overrides",
    [
        {"configured_mode": ACCURACY_OPTIMIZED},
        {"original_search_type": SearchType.FEELING_LUCKY},
        {"session_available": False},
        {"is_batch": True},
        {"only_context": True},
    ],
)
def test_latency_mode_falls_back_to_accuracy(overrides):
    assert resolve(GraphCompletionRetriever, **overrides) == ACCURACY_OPTIMIZED


def test_latency_mode_accepts_direct_retriever_calls_without_search_type():
    assert resolve(GraphCompletionRetriever, original_search_type=None) == LATENCY_OPTIMIZED


def test_latency_mode_rejects_subclasses_and_unrelated_retrievers():
    class ExtendedGraphRetriever(GraphCompletionRetriever):
        pass

    assert resolve(ExtendedGraphRetriever) == ACCURACY_OPTIMIZED
    assert resolve(object) == ACCURACY_OPTIMIZED
