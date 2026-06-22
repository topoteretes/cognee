"""Result type returned by graph-native delete/rollback operations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProvenanceDeleteResult:
    """Summary of a graph-native delete or rollback.

    Returned by ``GraphVectorStoreInterface`` operations so callers can log and
    assert on what was removed without re-reading the stores. ``nodes_deleted``
    / ``edges_deleted`` count artifacts hard-deleted (their last ref removed);
    ``nodes_detached`` / ``edges_detached`` count artifacts that survived because
    another source ref or run still keeps them alive (only the targeted ref was
    stripped).
    """

    nodes_deleted: int = 0
    edges_deleted: int = 0
    nodes_detached: int = 0
    edges_detached: int = 0


__all__ = ["ProvenanceDeleteResult"]
