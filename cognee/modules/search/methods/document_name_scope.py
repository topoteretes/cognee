"""Shared rules for search types that can preserve a named-document scope.

``node_name`` normally means a node-set filter: the retrievers that support it
look for matching ``belongs_to_set`` tags. A ``TextDocument`` is a source
document, though, and its chunks carry the source's name in their vector-store
payload rather than in those node-set tags. Treating a document name as a
node-set therefore silently returns no chunks for CHUNKS and RAG_COMPLETION.

The scope is passed to vector search as a document-name predicate, so the
database applies it while retrieving the nearest neighbors instead of loading
an unscoped result set and filtering that set in Python. This keeps the work
bounded by the vector backend's filtered search and the requested result count;
it avoids an additional scan over every chunk in the corpus. The backend
predicate is combined with any node-set predicate, preserving callers that
provide both kinds of scope.

Feeling-lucky selection needs a separate allowlist because the selected
retriever is not known until runtime. A document-name scope may only reach
retrievers that preserve that scope. In particular, hybrid completion is
allowed for automatic selection because its deferral path sends a custom node
type to graph completion; explicit hybrid search remains unsupported for this
scope and is rejected by the retriever factory.
"""

from cognee.modules.search.types import SearchType

DOCUMENT_NAME_FILTER_SEARCH_TYPES = frozenset(
    {
        SearchType.CHUNKS,
        SearchType.RAG_COMPLETION,
        SearchType.GRAPH_COMPLETION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.AGENTIC_COMPLETION,
    }
)

FEELING_LUCKY_DOCUMENT_NAME_SEARCH_TYPES = DOCUMENT_NAME_FILTER_SEARCH_TYPES | {
    SearchType.HYBRID_COMPLETION,
}


def is_document_name_scope(node_type: type | None, node_name: list[str] | None) -> bool:
    """True when ``node_name`` is intended to name source TextDocument nodes."""
    return bool(node_name) and getattr(node_type, "__name__", None) == "TextDocument"
