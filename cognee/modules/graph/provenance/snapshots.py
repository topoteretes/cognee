"""Snapshot dataclasses describing graph artifacts slated for deletion.

These are the graph-native replacements for the relational ledger rows that
``delete``/``rollback`` read today. Part 1 (storage primitives) builds these by
reading provenance off the graph and vector stores; Part 2 (delete/rollback
wiring) consumes them to issue the actual graph + vector deletes. They carry
exactly the fields the current ``delete_from_graph_and_vector`` flow needs:
the graph node/edge identity, the vector collections to clean, and the
provenance refs that decide whether an artifact is uniquely owned (safe to hard
delete) or still shared.

All three are frozen so they can be deduplicated in sets, mirroring the
slug-based dedup the ledger flow performs before issuing deletes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeIdentity:
    """The minimal identity of a graph edge, independent of the ledger.

    ``(source_node_id, relationship_name, target_node_id)`` uniquely identifies
    an edge in the graph. This is what triplet ids and ``EdgeType`` retrieval
    ids are derived from in the current delete flow.
    """

    source_node_id: str
    relationship_name: str
    target_node_id: str


@dataclass(frozen=True)
class NodeDeleteData:
    """Everything needed to delete one graph node from the graph + vector stores.

    Replaces a relational ``nodes`` row for deletion purposes.

    Attributes:
        node_id: The graph node id (the ledger's ``slug``).
        node_type: The node's type (e.g. "Entity", "NodeSet"); combined with an
            indexed field to form a vector collection name (``f"{type}_{field}"``).
        label: The node's human-readable label, if any. Needed to strip orphaned
            ``NodeSet`` tags after a delete.
        indexed_fields: The embeddable property names; each maps to a vector
            collection that must be cleaned for this node.
        source_refs: Source refs (see refs.make_source_ref) the node belongs to.
            A node is safe to hard-delete only when the delete removes its last
            remaining source ref.
        source_run_refs: Source-run refs (see refs.make_source_run_ref) that
            touched the node. Used by rollback to scope removal to one run.
        dataset_ids: Dataset ids (stringified) the node belongs to. Used by
            dataset-scoped delete to decide survival: a node shared across
            datasets survives a single-dataset delete.
    """

    node_id: str
    node_type: str
    label: str | None = None
    indexed_fields: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    source_run_refs: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeDeleteData:
    """Everything needed to delete one graph edge from the graph + vector stores.

    Replaces a relational ``edges`` row for deletion purposes.

    Attributes:
        identity: The edge's graph identity (source, relationship, target).
        edge_retrieval_text: The text used to derive the edge's ``EdgeType`` and
            triplet vector ids. Falls back to ``identity.relationship_name`` when
            no distinct retrieval text was stored.
        source_refs: Source refs the edge belongs to (see NodeDeleteData).
        source_run_refs: Source-run refs that touched the edge (see NodeDeleteData).
        dataset_ids: Dataset ids (stringified) the edge belongs to (see NodeDeleteData).
    """

    identity: EdgeIdentity
    edge_retrieval_text: str | None = None
    source_refs: tuple[str, ...] = ()
    source_run_refs: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()


__all__ = ["EdgeIdentity", "NodeDeleteData", "EdgeDeleteData"]
