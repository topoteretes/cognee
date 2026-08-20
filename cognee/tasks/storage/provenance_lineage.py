"""Provenance lineage layer.

This layer records where each graph node came from by adding explicit lineage
edges during ingestion:

    <node> -derived_from-> Document -in_dataset-> Dataset

Cognee already stamps ``source_*`` fields on data points and stores
``data_id``/``dataset_id`` on the relational node and edge rows, but those fields
only keep the first source and a Dataset is not represented as a graph node. This
module fills both gaps. Every extracted node gets a ``derived_from`` edge to its
source Document, and each Document gets an ``in_dataset`` edge to a single shared
Dataset node. Because a node that recurs across data items is written once per
data item, a merged node ends up with one ``derived_from`` edge per contributing
Document, which is the many-to-many case the ``source_*`` fields cannot express.

The layer is on by default and is controlled by ``PROVENANCE_LINEAGE`` and
``PROVENANCE_LINEAGE_DEPTH``. Every lineage edge carries a ``provenance=True``
property so callers can traverse or filter the layer without knowing the domain
schema. The relationship names use underscores rather than a namespaced form
(for example ``prov:in``) because some backends, such as Neo4j, use the
relationship name as a native edge type.

The overhead is small. The only new node is the Dataset node (one per dataset,
and it is not embedded because it declares no ``index_fields``). Every lineage
edge shares a constant ``edge_text`` equal to its relationship name, so the
``derived_from`` edges collapse to a single embedded ``EdgeType`` in
``index_graph_edges`` instead of one per edge.
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

# Relationship name for a <node> -> Document provenance edge.
DERIVED_FROM_RELATIONSHIP = "derived_from"

# Relationship name for the Document -> Dataset provenance edge.
IN_DATASET_RELATIONSHIP = "in_dataset"

# Edge property that marks an edge as part of the provenance lineage layer.
PROVENANCE_EDGE_FLAG = "provenance"

# Graph node types that are structural rather than extracted content. They do not
# get a derived_from edge to a Document. DatasetNode is the dataset tier itself,
# and NodeSet is a tag that spans documents, so anchoring it to one document would
# be misleading.
_STRUCTURAL_TYPES = frozenset({"DatasetNode", "NodeSet"})

# Depth controls how far up the lineage is built. "document" adds only the
# <node> -> Document edges. "dataset" also adds the Document -> Dataset edge and
# is the default.
VALID_DEPTHS = ("document", "dataset")
DEFAULT_DEPTH = "dataset"


class ProvenanceConfig(BaseSettings):
    # When True (the default) the base pipeline builds the lineage layer. Set to
    # False to reproduce the pre-lineage graph exactly.
    provenance_lineage: bool = True
    # How far up the lineage is built: "document" or "dataset" (the default).
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
    """Build a provenance edge tuple of (source, target, name, properties).

    edge_text is set to the relationship name so that index_graph_edges, which
    embeds distinct edge_text values, collapses all edges of one relationship
    into a single EdgeType instead of one per edge.
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
    """Return the deterministic id for a dataset's lineage node.

    The id is derived from the dataset id so a single Dataset node is shared
    across every data item in the dataset.
    """
    return generate_node_id(f"DatasetNode:{dataset_id}")


def _is_structural(node: DataPoint) -> bool:
    return getattr(node, "type", None) in _STRUCTURAL_TYPES


def build_source_lineage(
    nodes: List[DataPoint],
    data_item: Any,
) -> List[Tuple[Any, Any, str, dict]]:
    """Build a derived_from edge from every content node to its source Document.

    The Document node id equals ``data_item.id`` because cognify constructs the
    Document with ``id=data_item.id``. Every node in this batch belongs to that
    Document, so each content node gets one derived_from edge to it. The Document
    node itself and structural nodes such as Dataset and NodeSet are skipped.

    Returns an empty list when the Document node is not present in ``nodes``,
    which avoids creating an edge with a missing endpoint.
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
    """Build the dataset tier: a Dataset node and a Document -> Dataset edge.

    Returns ``(extra_nodes, extra_edges)`` with one shared Dataset node and one
    ``in_dataset`` edge. The edge is produced only when the Document node
    (id ``== data_item.id``) is present in ``nodes``, which avoids creating an
    edge with a missing endpoint for pipelines that do not materialize a Document.
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
    """Build the full lineage layer for one add_data_points batch.

    Returns ``(extra_nodes, extra_edges)`` to append before the upsert, so the
    edges are written and tracked for deletion through the existing ledger path.
    The result respects ``PROVENANCE_LINEAGE_DEPTH``.
    """
    config = config or get_provenance_config()
    if not config.provenance_lineage:
        return [], []

    depth = _normalize_depth(config.provenance_lineage_depth)

    extra_nodes: List[DataPoint] = []
    extra_edges: List[Tuple[Any, Any, str, dict]] = []

    extra_edges.extend(build_source_lineage(nodes, data_item))

    if depth == "dataset":
        dataset_nodes, dataset_edges = build_dataset_lineage(nodes, dataset, data_item)
        extra_nodes.extend(dataset_nodes)
        extra_edges.extend(dataset_edges)

    return extra_nodes, extra_edges
