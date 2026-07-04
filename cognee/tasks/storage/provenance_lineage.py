"""Provenance lineage layer: materialize source lineage as first-class graph edges.

Provenance is already captured in three separate places today: ``source_*``
fields stamped on every DataPoint, the relational ``nodes``/``edges`` ledger
(``data_id``/``dataset_id`` on every row), and domain edges such as
``is_part_of`` / ``made_from``. Those give a traversable lineage from an
extracted node up to its Document, but the chain stops there: a Dataset is not a
graph node, so there is no in-graph path from a Document to the dataset it
belongs to.

This module adds that missing Dataset tier. For each ingested data item it emits
one deduplicated ``DatasetNode`` plus a ``Document -in_dataset-> Dataset`` edge,
so every node can be traced up to its dataset by graph traversal (not only by a
relational lookup).

The layer is on by default and configurable via ``PROVENANCE_LINEAGE`` /
``PROVENANCE_LINEAGE_DEPTH``. Lineage edges carry a ``provenance=True`` property
so they form a queryable, reserved contract without relying on a namespaced
relationship name (relationship names are used as native edge types by some
backends, e.g. Neo4j, so a plain underscore name is the safe choice).
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.DatasetNode import DatasetNode
from cognee.modules.engine.utils import generate_node_id

# Relationship name for the Document -> Dataset lineage edge. Kept underscore
# style (not, e.g., "prov:in") so it is safe as a native relationship type
# across all graph backends. The reserved-provenance contract is expressed by
# the ``provenance`` edge property below, not by the relationship name.
IN_DATASET_RELATIONSHIP = "in_dataset"

# Property flag marking an edge as part of the provenance lineage layer, so it
# can be traversed/filtered generically ("what produced this / what depends on
# this source") without knowing the domain schema.
PROVENANCE_EDGE_FLAG = "provenance"

VALID_DEPTHS = ("chunk", "document", "dataset")


class ProvenanceConfig(BaseSettings):
    # PROVENANCE_LINEAGE — when True (default) the base pipeline materializes the
    # source lineage subgraph. False reproduces the pre-lineage behavior exactly.
    provenance_lineage: bool = True
    # PROVENANCE_LINEAGE_DEPTH — how far up the lineage is materialized. "dataset"
    # (default) adds the Document -> Dataset tier. Bounds ingest overhead.
    provenance_lineage_depth: str = "dataset"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_provenance_config() -> ProvenanceConfig:
    return ProvenanceConfig()


def dataset_lineage_node_id(dataset_id: Any):
    """Deterministic id for a dataset's lineage node.

    Derived from the dataset id so a single DatasetNode is shared across every
    data item in the dataset (one Dataset node, not one per document).
    """
    return generate_node_id(f"DatasetNode:{dataset_id}")


def build_dataset_lineage(
    nodes: List[DataPoint],
    dataset: Any,
    data_item: Any,
) -> Tuple[List[DataPoint], List[Tuple[Any, Any, str, dict]]]:
    """Build the Dataset tier of the provenance lineage subgraph.

    Returns ``(extra_nodes, extra_edges)`` to append before the graph write: one
    deduplicated ``DatasetNode`` plus a ``Document -in_dataset-> Dataset`` edge.

    The Document graph node id equals ``data_item.id`` (cognify constructs
    ``Document(id=data_item.id)``), so the edge source is ``data_item.id``. The
    edge is emitted only when that Document node is actually present in
    ``nodes``, to avoid a dangling endpoint (e.g. custom pipelines that do not
    materialize a Document node in this batch).
    """
    if dataset is None or data_item is None:
        return [], []

    dataset_id = getattr(dataset, "id", None)
    document_id = getattr(data_item, "id", None)
    if dataset_id is None or document_id is None:
        return [], []

    node_ids = {str(node.id) for node in nodes}
    if str(document_id) not in node_ids:
        # No Document node materialized in this batch: nothing to anchor to.
        return [], []

    node_id = dataset_lineage_node_id(dataset_id)
    dataset_name = getattr(dataset, "name", None) or str(dataset_id)
    dataset_node = DatasetNode(id=node_id, name=dataset_name)

    edge_properties = {
        "source_node_id": document_id,
        "target_node_id": node_id,
        "relationship_name": IN_DATASET_RELATIONSHIP,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        PROVENANCE_EDGE_FLAG: True,
    }
    edge = (document_id, node_id, IN_DATASET_RELATIONSHIP, edge_properties)

    return [dataset_node], [edge]
