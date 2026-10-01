import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_aware_completion import can_run_as_turn
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType


def is_eligible_for_turn(retriever_type: type, **overrides):
    inputs = {
        "original_search_type": SearchType.RAG_COMPLETION,
        "retriever_type": retriever_type,
        "session_available": True,
        "is_batch": False,
        "only_context": False,
    }
    inputs.update(overrides)
    return can_run_as_turn(**inputs)


@pytest.mark.parametrize(
    "retriever_type",
    [
        CompletionRetriever,
        GraphCompletionRetriever,
        HybridRetriever,
        TripletRetriever,
    ],
)
def test_only_designated_retriever_types_are_eligible(retriever_type):
    assert is_eligible_for_turn(retriever_type) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"original_search_type": SearchType.FEELING_LUCKY},
        {"session_available": False},
        {"is_batch": True},
        {"only_context": True},
    ],
)
def test_unsupported_call_shapes_are_not_eligible(overrides):
    assert is_eligible_for_turn(GraphCompletionRetriever, **overrides) is False


def test_direct_retriever_calls_without_search_type_are_eligible():
    assert is_eligible_for_turn(GraphCompletionRetriever, original_search_type=None) is True


def test_retriever_subclasses_are_not_eligible():
    from cognee.modules.retrieval.graph_summary_completion_retriever import (
        GraphSummaryCompletionRetriever,
    )

    class ExtendedGraphRetriever(GraphCompletionRetriever):
        pass

    assert is_eligible_for_turn(ExtendedGraphRetriever) is False
    assert is_eligible_for_turn(GraphSummaryCompletionRetriever) is False
    assert is_eligible_for_turn(object) is False
