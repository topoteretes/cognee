from functools import lru_cache
from typing import Literal

from cognee.modules.search.types import SearchType

SessionSearchMode = Literal["accuracy_optimized", "latency_optimized"]

ACCURACY_OPTIMIZED: SessionSearchMode = "accuracy_optimized"
LATENCY_OPTIMIZED: SessionSearchMode = "latency_optimized"


@lru_cache(maxsize=1)
def _latency_retriever_types() -> frozenset[type]:
    from cognee.modules.retrieval.completion_retriever import CompletionRetriever
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
    from cognee.modules.retrieval.triplet_retriever import TripletRetriever

    return frozenset(
        {
            CompletionRetriever,
            GraphCompletionRetriever,
            HybridRetriever,
            TripletRetriever,
        }
    )


def resolve_session_search_mode(
    configured_mode: SessionSearchMode,
    *,
    original_search_type: SearchType | None,
    retriever_type: type,
    session_available: bool,
    is_batch: bool,
    only_context: bool,
    structured_output_supported: bool,
) -> SessionSearchMode:
    """Resolve the effective session-search mode without changing runtime state."""
    if configured_mode != LATENCY_OPTIMIZED:
        return ACCURACY_OPTIMIZED

    if (
        original_search_type is SearchType.FEELING_LUCKY
        or not session_available
        or is_batch
        or only_context
        or not structured_output_supported
        or retriever_type not in _latency_retriever_types()
    ):
        return ACCURACY_OPTIMIZED

    return LATENCY_OPTIMIZED
