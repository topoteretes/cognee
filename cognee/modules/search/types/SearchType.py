import warnings
from enum import Enum


class SearchType(str, Enum):
    SUMMARIES = "SUMMARIES"
    CHUNKS = "CHUNKS"
    RAG_COMPLETION = "RAG_COMPLETION"
    HYBRID_COMPLETION = "HYBRID_COMPLETION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    GRAPH_COMPLETION_DECOMPOSITION = "GRAPH_COMPLETION_DECOMPOSITION"
    GRAPH_SUMMARY_COMPLETION = "GRAPH_SUMMARY_COMPLETION"
    CYPHER = "CYPHER"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    GRAPH_COMPLETION_COT = "GRAPH_COMPLETION_COT"
    GRAPH_COMPLETION_CONTEXT_EXTENSION = "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    FEELING_LUCKY = "FEELING_LUCKY"
    TEMPORAL = "TEMPORAL"
    CODING_RULES = "CODING_RULES"
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"
    AGENTIC_COMPLETION = "AGENTIC_COMPLETION"
    CODE = "CODE"
    GRAPH_REPORT = "GRAPH_REPORT"
    BROAD = "BROAD"
    SKILLS = "SKILLS"
    # Deprecated (SDK-1115): triplet embeddings were removed, so GRAPH_COMPLETION
    # answers it. Removed in the next release.
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"


# Deprecated search type -> the type that serves it until it is removed.
DEPRECATED_SEARCH_TYPES: dict[SearchType, SearchType] = {
    SearchType.TRIPLET_COMPLETION: SearchType.GRAPH_COMPLETION,
}


def resolve_deprecated_search_type(
    query_type: SearchType | None, *, stacklevel: int = 2
) -> SearchType | None:
    """Return the replacement for a deprecated search type, warning once per call site.

    ``stacklevel`` is what the calling API function would pass to its own
    ``warnings.warn`` to point at its caller: 2 for an undecorated function,
    one more per wrapping decorator frame.
    """
    replacement = DEPRECATED_SEARCH_TYPES.get(query_type) if query_type is not None else None
    if replacement is None:
        return query_type
    warnings.warn(
        f"SearchType.{query_type.name} is deprecated and will be removed in the next release; "
        f"it now runs SearchType.{replacement.name}. Use SearchType.{replacement.name} instead.",
        FutureWarning,
        stacklevel=stacklevel + 1,
    )
    return replacement
