"""What ``node_name`` means for each search type, and which types can honour it.

``node_name`` is a filter whose meaning is set by ``node_type``:

- ``NodeSet`` (the default): the node-set tags the data was ingested with. Chunk
  searches apply it to the ``belongs_to_set`` of every vector row.
- A ``Document`` subclass: the source documents the chunks were cut from. Chunk
  searches apply it to the ``document_name`` every chunk row carries; graph
  completion projects the document nodes and their neighbours.
- Any other type, or ``None``: graph nodes of that type by name. Only the
  graph-completion family can honour it, by projecting the named nodes.

A search type handed a filter it cannot apply must refuse it. Dropping the filter
returns unscoped results and matching it against the wrong field returns none,
and neither tells the caller what happened.
"""

from enum import Enum

from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType


class NodeNameScope(str, Enum):
    NODE_SET = "node_set"
    DOCUMENT = "document"
    GRAPH = "graph"


_GRAPH_COMPLETION_FAMILY = frozenset(
    {
        SearchType.GRAPH_COMPLETION,
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.AGENTIC_COMPLETION,
    }
)
_EVERY_SCOPE = frozenset(NodeNameScope)

# The scopes each search type honours. A type missing here ignores ``node_name``
# and is refused when one is given. HYBRID_COMPLETION lists every scope because
# a scope other than a node set is served by deferring to GRAPH_COMPLETION
# (see hybrid_deferral.request_deferral_reason).
SUPPORTED_SCOPES: dict[SearchType, frozenset[NodeNameScope]] = {
    SearchType.CHUNKS: frozenset({NodeNameScope.NODE_SET, NodeNameScope.DOCUMENT}),
    SearchType.RAG_COMPLETION: frozenset({NodeNameScope.NODE_SET, NodeNameScope.DOCUMENT}),
    SearchType.HYBRID_COMPLETION: _EVERY_SCOPE,
    SearchType.TEMPORAL: frozenset({NodeNameScope.NODE_SET}),
    SearchType.CODING_RULES: frozenset({NodeNameScope.NODE_SET}),
    **{search_type: _EVERY_SCOPE for search_type in _GRAPH_COMPLETION_FAMILY},
}


def node_name_scope(node_type: type | None, node_name: list[str] | None) -> NodeNameScope | None:
    """The scope ``node_name`` asks for, or None when no filter is set."""
    if not node_name:
        return None
    if node_type is NodeSet:
        return NodeNameScope.NODE_SET
    if isinstance(node_type, type) and issubclass(node_type, Document):
        return NodeNameScope.DOCUMENT
    return NodeNameScope.GRAPH


def search_types_honouring(scope: NodeNameScope) -> frozenset[SearchType]:
    return frozenset(
        search_type for search_type, scopes in SUPPORTED_SCOPES.items() if scope in scopes
    )


def lucky_search_types(
    node_type: type | None, node_name: list[str] | None
) -> frozenset[SearchType] | None:
    """The types FEELING_LUCKY may pick for this request, or None when any type will do."""
    scope = node_name_scope(node_type, node_name)
    return None if scope is None else search_types_honouring(scope)


def reject_unsupported_node_name(
    query_type: SearchType,
    node_type: type | None,
    node_name: list[str] | None,
    node_name_filter_operator: str,
) -> NodeNameScope | None:
    """Raise unless ``query_type`` can apply the filter ``node_type`` / ``node_name`` ask for.

    Returns the scope so the caller can route the names to the right field.
    """
    scope = node_name_scope(node_type, node_name)
    if scope is None:
        return None

    if scope not in SUPPORTED_SCOPES.get(query_type, frozenset()):
        type_name = getattr(node_type, "__name__", repr(node_type))
        honouring = ", ".join(sorted(t.value for t in search_types_honouring(scope)))
        raise QueryValidationError(
            message=(
                f"{query_type.value} does not apply node_name with node_type={type_name}. "
                f"Search types that do: {honouring}."
            )
        )

    if (
        scope is NodeNameScope.DOCUMENT
        and node_name_filter_operator == "AND"
        and len(set(node_name or ())) > 1
    ):
        raise QueryValidationError(
            message=(
                "node_name_filter_operator='AND' cannot apply to several document names: "
                "a chunk belongs to exactly one document."
            )
        )

    return scope
