"""Identities of graph elements removed by a data-item delete.

Session invalidation compares these against the ``used_graph_element_ids``
recorded on cached session answers, so both id spaces must match what
retrievers store: stringified graph node ids for nodes, and the deterministic
``edge_object_id`` (``generate_edge_object_id(source, target, relationship)``)
for edges.
"""

from dataclasses import dataclass, field

from cognee.modules.engine.utils import generate_edge_object_id


@dataclass
class DeletedGraphElements:
    """Node/edge identities hard-deleted from the graph by a delete operation."""

    node_ids: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)

    def merge(self, other: "DeletedGraphElements") -> None:
        self.node_ids |= other.node_ids
        self.edge_ids |= other.edge_ids

    @classmethod
    def from_ledger_rows(cls, nodes: list, edges: list) -> "DeletedGraphElements":
        """Build from relational-ledger Node/Edge rows (slug = graph node id)."""
        return cls(
            node_ids={str(node.slug) for node in nodes},
            edge_ids={
                str(
                    generate_edge_object_id(
                        str(edge.source_node_id),
                        str(edge.destination_node_id),
                        edge.relationship_name,
                    )
                )
                for edge in edges
            },
        )

    @classmethod
    def from_source_ref_removal(cls, result) -> "DeletedGraphElements":
        """Build from a graph-provenance ``SourceRefRemovalResult``."""
        return cls(
            node_ids={str(node_id) for node_id in result.deleted_node_ids},
            edge_ids={
                str(
                    generate_edge_object_id(
                        str(edge.source_id), str(edge.target_id), edge.relationship_name
                    )
                )
                for edge in result.deleted_edges
            },
        )
