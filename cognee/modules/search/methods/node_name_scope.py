"""What ``node_name`` means for each search type, and which types can honour it.

``node_name`` is a filter whose meaning is set by ``node_type``:

- ``NodeSet`` (the default): the node-set tags the data was ingested with. Chunk
  searches apply it to the ``belongs_to_set`` of every vector row.
- Any other graph node type: the named nodes of that type and their one-hop
  neighbours, projected by the graph. Graph completion ranks that projection;
  chunk searches score the chunks in it (a document's chunks, an entity's
  mentions), see ``scoped_chunk_search``.

A search type handed a filter it cannot apply must refuse it. Dropping the filter
returns unscoped results and matching it against the wrong field returns none,
and neither tells the caller what happened.
"""

from enum import Enum

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType


class NodeNameScope(str, Enum):
    NODE_SET = "node_set"
    GRAPH = "graph"


_EVERY_SCOPE = frozenset(NodeNameScope)

# The scopes each search type honours. A type missing here ignores ``node_name``
# and is refused when one is given. HYBRID_COMPLETION lists every scope because
# a graph scope is served by deferring to GRAPH_COMPLETION
# (see hybrid_deferral.request_deferral_reason).
SUPPORTED_SCOPES: dict[SearchType, frozenset[NodeNameScope]] = {
    SearchType.CHUNKS: _EVERY_SCOPE,
    SearchType.RAG_COMPLETION: _EVERY_SCOPE,
    SearchType.HYBRID_COMPLETION: _EVERY_SCOPE,
    SearchType.TEMPORAL: frozenset({NodeNameScope.NODE_SET}),
    SearchType.CODING_RULES: frozenset({NodeNameScope.NODE_SET}),
    SearchType.GRAPH_COMPLETION: _EVERY_SCOPE,
    SearchType.GRAPH_COMPLETION_COT: _EVERY_SCOPE,
    SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: _EVERY_SCOPE,
    SearchType.GRAPH_COMPLETION_DECOMPOSITION: _EVERY_SCOPE,
    SearchType.GRAPH_SUMMARY_COMPLETION: _EVERY_SCOPE,
    SearchType.AGENTIC_COMPLETION: _EVERY_SCOPE,
}


def node_name_scope(node_type: type | None, node_name: list[str] | None) -> NodeNameScope | None:
    """The scope ``node_name`` asks for, or None when no filter is set.

    ``node_type=None`` with names is refused: the graph projection needs the type,
    and without it the names would be dropped on the floor.
    """
    if not node_name:
        return None
    if node_type is NodeSet:
        return NodeNameScope.NODE_SET
    if node_type is None:
        raise QueryValidationError(
            message=(
                "node_name needs a node_type: NodeSet for node-set tags, or the graph node "
                "type to scope to (a Document class, Entity, ...)."
            )
        )
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
) -> NodeNameScope | None:
    """Raise unless ``query_type`` can apply the filter ``node_type`` / ``node_name`` ask for."""
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
    return scope
