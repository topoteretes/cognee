"""Provenance lineage layer: materialize source lineage as first-class graph edges.

Provenance is already captured in three separate places today: ``source_*``
fields stamped on every DataPoint (lossy — first write wins), the relational
``nodes``/``edges`` ledger (``data_id``/``dataset_id`` on every row), and domain
edges such as ``is_part_of`` / ``made_from`` (schema-specific, and the chain
stops at the Document — a Dataset is not a graph node). None of these is a single,
uniform, in-graph contract that answers "what produced this node" for *every*
node type.

This module adds that contract. For each ingested data item it emits, on top of
what already exists:

* one ``<node> -derived_from-> Document`` edge for every extracted content node,
  so every node type (including ones with no domain source edge, e.g.
  ``EntityType``, or custom-model nodes) is uniformly traceable to its source
  Document. Because an extracted node that recurs across data items is written
  once per data item (its id is deterministic), a *merged* node accumulates one
  ``derived_from`` edge per contributing Document — i.e. many-to-many provenance,
  the case the lossy ``source_*`` fields get wrong.
* one deduplicated ``DatasetNode`` plus a ``Document -in_dataset-> Dataset`` edge,
  completing the chain up to the dataset.

The layer is on by default and configurable via ``PROVENANCE_LINEAGE`` /
``PROVENANCE_LINEAGE_DEPTH``. Lineage edges carry a ``provenance=True`` property
so they form a queryable, reserved contract that can be traversed/filtered
generically, without relying on a namespaced relationship name (relationship
names are used as native edge types by some backends, e.g. Neo4j, so a plain
underscore name is the safe choice).

Overhead is bounded: the only new node is the DatasetNode (one per dataset, not
embedded — it declares no ``index_fields``), and every lineage edge shares a
constant ``edge_text`` equal to its relationship name, so the N ``derived_from``
edges collapse to a single embedded ``EdgeType`` in ``index_graph_edges`` rather
than N distinct ones.
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.DatasetNode import DatasetNode
from cognee.modules.engine.utils import generate_node_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("provenance_lineage")

# Relationship name for a ``<node> -> Document`` provenance edge.
DERIVED_FROM_RELATIONSHIP = "derived_from"

# Relationship name for the ``Document -> Dataset`` provenance edge. Kept
# underscore style (not, e.g., "prov:in") so it is safe as a native relationship
# type across all graph backends. The reserved-provenance contract is expressed
# by the ``provenance`` edge property, not by the relationship name.
IN_DATASET_RELATIONSHIP = "in_dataset"

# Property flag marking an edge as part of the provenance lineage layer.
PROVENANCE_EDGE_FLAG = "provenance"

# Graph node ``type`` values that are structural (not extracted content) and so
# must not be given a ``derived_from`` edge to a Document. ``DatasetNode`` is the
# dataset tier itself; ``NodeSet`` is a cross-document tag, so anchoring it to a
# single Document would be misleading.
_STRUCTURAL_TYPES = frozenset({"DatasetNode", "NodeSet"})

# Depth controls how far up the synthetic lineage is materialized:
#   "document" — only ``<node> -derived_from-> Document`` edges.
#   "dataset"  — also ``Document -in_dataset-> Dataset`` (the default, full chain).
VALID_DEPTHS = ("document", "dataset")
DEFAULT_DEPTH = "dataset"


class ProvenanceConfig(BaseSettings):
    # PROVENANCE_LINEAGE — when True (default) the base pipeline materializes the
    # source lineage subgraph. False reproduces the pre-lineage behavior exactly.
    provenance_lineage: bool = True
    # PROVENANCE_LINEAGE_DEPTH — "document" or "dataset" (default). See VALID_DEPTHS.
    provenance_lineage_depth: str = DEFAULT_DEPTH

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_provenance_config() -> ProvenanceConfig:
    return ProvenanceConfig()


def _normalize_depth(depth: Optional[str]) -> str:
    """Return a valid depth, falling back to the default for unknown values."""
    if depth in VALID_DEPTHS:
        return depth
    logger.warning("Unknown PROVENANCE_LINEAGE_DEPTH %r; falling back to %r.", depth, DEFAULT_DEPTH)
    return DEFAULT_DEPTH


def _provenance_edge(
    source_id: Any, target_id: Any, relationship_name: str
) -> Tuple[Any, Any, str, dict]:
    """Build a provenance edge tuple in ``(source, target, name, properties)`` shape.

    ``edge_text`` is set to the relationship name (a small constant set) so that
    ``index_graph_edges`` collapses all edges of one relationship into a single
    embedded ``EdgeType`` instead of one per edge.
    """
    properties = {
        "source_node_id": source_id,
        "target_node_id": target_id,
        "relationship_name": relationship_name,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "edge_text": relationship_name,
        PROVENANCE_EDGE_FLAG: True,
    }
    return (source_id, target_id, relationship_name, properties)


def dataset_lineage_node_id(dataset_id: Any):
    """Deterministic id for a dataset's lineage node.

    Derived from the dataset id so a single DatasetNode is shared across every
    data item in the dataset (one Dataset node, not one per document).
    """
    return generate_node_id(f"DatasetNode:{dataset_id}")


def _is_structural(node: DataPoint) -> bool:
    return getattr(node, "type", None) in _STRUCTURAL_TYPES


def build_source_lineage(
    nodes: List[DataPoint],
    data_item: Any,
) -> List[Tuple[Any, Any, str, dict]]:
    """Build ``<node> -derived_from-> Document`` edges for every content node.

    The Document graph node id equals ``data_item.id`` (cognify constructs
    ``Document(id=data_item.id)``). Every node in this batch belongs to that
    Document, so each content node gets one ``derived_from`` edge to it. The
    Document node itself and structural nodes (Dataset/NodeSet) are skipped.

    Returns an empty list when the Document node is not present in ``nodes``
    (nothing to anchor to).
    """
    if data_item is None:
        return []
    document_id = getattr(data_item, "id", None)
    if document_id is None:
        return []

    node_ids = {str(node.id) for node in nodes}
    if str(document_id) not in node_ids:
        return []

    document_key = str(document_id)
    edges = []
    for node in nodes:
        if str(node.id) == document_key or _is_structural(node):
            continue
        edges.append(_provenance_edge(node.id, document_id, DERIVED_FROM_RELATIONSHIP))
    return edges


def build_dataset_lineage(
    nodes: List[DataPoint],
    dataset: Any,
    data_item: Any,
) -> Tuple[List[DataPoint], List[Tuple[Any, Any, str, dict]]]:
    """Build the Dataset tier of the provenance lineage subgraph.

    Returns ``(extra_nodes, extra_edges)``: one deduplicated ``DatasetNode`` plus
    a ``Document -in_dataset-> Dataset`` edge. Emitted only when the Document node
    (id ``== data_item.id``) is present in ``nodes``, to avoid a dangling
    endpoint (e.g. custom pipelines that do not materialize a Document node).
    """
    if dataset is None or data_item is None:
        return [], []

    dataset_id = getattr(dataset, "id", None)
    document_id = getattr(data_item, "id", None)
    if dataset_id is None or document_id is None:
        return [], []

    node_ids = {str(node.id) for node in nodes}
    if str(document_id) not in node_ids:
        return [], []

    node_id = dataset_lineage_node_id(dataset_id)
    dataset_name = getattr(dataset, "name", None) or str(dataset_id)
    dataset_node = DatasetNode(id=node_id, name=dataset_name)
    edge = _provenance_edge(document_id, node_id, IN_DATASET_RELATIONSHIP)

    return [dataset_node], [edge]


def build_provenance_lineage(
    nodes: List[DataPoint],
    dataset: Any,
    data_item: Any,
    config: Optional[ProvenanceConfig] = None,
) -> Tuple[List[DataPoint], List[Tuple[Any, Any, str, dict]]]:
    """Assemble the full provenance lineage layer for one ``add_data_points`` batch.

    Returns ``(extra_nodes, extra_edges)`` to append to the graph write before the
    upsert, so the layer is written and forget-tracked through the existing ledger
    path. Honors ``PROVENANCE_LINEAGE_DEPTH``.
    """
    config = config or get_provenance_config()
    if not config.provenance_lineage:
        return [], []

    depth = _normalize_depth(config.provenance_lineage_depth)

    extra_nodes: List[DataPoint] = []
    extra_edges: List[Tuple[Any, Any, str, dict]] = []

    # <node> -derived_from-> Document (both depths).
    extra_edges.extend(build_source_lineage(nodes, data_item))

    # Document -in_dataset-> Dataset (dataset depth only).
    if depth == "dataset":
        dataset_nodes, dataset_edges = build_dataset_lineage(nodes, dataset, data_item)
        extra_nodes.extend(dataset_nodes)
        extra_edges.extend(dataset_edges)

    return extra_nodes, extra_edges
