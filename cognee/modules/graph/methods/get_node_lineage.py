"""Read the provenance lineage of a graph node.

The provenance lineage layer (see ``cognee.tasks.storage.provenance_lineage``)
writes ``derived_from`` and ``in_dataset`` edges during ingestion. This module
reads those edges back to answer two questions:

* ``get_source_lineage(node_id)``: what produced this node. It returns the source
  documents the node was derived from and the datasets those documents belong to.
* ``get_derived_nodes(source_id)``: what was produced from this source. Given a
  Document it returns the nodes derived from it, and given a Dataset it returns
  the documents in it.

Both read from the graph engine's ``get_connections``, which returns each edge as
a ``(node, edge, neighbor)`` tuple. The edge properties carry ``source_node_id``
and ``target_node_id``, so the direction of each edge is known regardless of the
order the backend returns the endpoints in.
"""

from typing import Any, Dict, List, Optional

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.tasks.storage.provenance_lineage import (
    DERIVED_FROM_RELATIONSHIP,
    IN_DATASET_RELATIONSHIP,
)


def _node_id(node: Optional[Dict[str, Any]]) -> str:
    return str((node or {}).get("id"))


def _neighbors_by_relationship(
    connections: List[Any],
    node_id: str,
    relationship_name: str,
    outgoing: bool,
) -> List[Dict[str, Any]]:
    """Return the neighbor nodes reached over edges of one relationship type.

    ``connections`` is the output of ``get_connections(node_id)``: a list of
    ``(node, edge, neighbor)`` tuples. When ``outgoing`` is True this keeps edges
    that start at ``node_id`` (node is the source), and when False it keeps edges
    that end at ``node_id`` (node is the target). The neighbor returned is always
    the endpoint that is not ``node_id``.
    """
    node_key = str(node_id)
    neighbors = []
    for source, edge, target in connections:
        if not isinstance(edge, dict) or edge.get("relationship_name") != relationship_name:
            continue
        edge_source = str(edge.get("source_node_id", _node_id(source)))
        edge_target = str(edge.get("target_node_id", _node_id(target)))
        if outgoing and edge_source == node_key:
            neighbors.append(target if _node_id(target) != node_key else source)
        elif not outgoing and edge_target == node_key:
            neighbors.append(source if _node_id(source) != node_key else target)
    return neighbors


def _dedupe_by_id(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for node in nodes:
        key = _node_id(node)
        if key not in seen:
            seen.add(key)
            unique.append(node)
    return unique


async def get_source_lineage(node_id: Any, graph_engine=None) -> Dict[str, List[Dict[str, Any]]]:
    """Return the documents and datasets a node was produced from.

    The result is ``{"documents": [...], "datasets": [...]}`` where each entry is
    a node record with ``id``, ``name`` and ``type``. Datasets are collected by
    following the ``in_dataset`` edge from each source document.
    """
    engine = graph_engine or await get_graph_engine()

    node_connections = await engine.get_connections(str(node_id))
    documents = _dedupe_by_id(
        _neighbors_by_relationship(
            node_connections, node_id, DERIVED_FROM_RELATIONSHIP, outgoing=True
        )
    )

    datasets: List[Dict[str, Any]] = []
    for document in documents:
        document_id = _node_id(document)
        document_connections = await engine.get_connections(document_id)
        datasets.extend(
            _neighbors_by_relationship(
                document_connections, document_id, IN_DATASET_RELATIONSHIP, outgoing=True
            )
        )

    return {"documents": documents, "datasets": _dedupe_by_id(datasets)}


async def get_derived_nodes(source_id: Any, graph_engine=None) -> List[Dict[str, Any]]:
    """Return the nodes produced from a source (a Document or a Dataset).

    For a Document this is the nodes that have a ``derived_from`` edge to it. For
    a Dataset this is the documents that have an ``in_dataset`` edge to it. Both
    are included so the function works for either kind of source.
    """
    engine = graph_engine or await get_graph_engine()

    connections = await engine.get_connections(str(source_id))
    derived = _neighbors_by_relationship(
        connections, source_id, DERIVED_FROM_RELATIONSHIP, outgoing=False
    )
    documents = _neighbors_by_relationship(
        connections, source_id, IN_DATASET_RELATIONSHIP, outgoing=False
    )
    return _dedupe_by_id(derived + documents)
