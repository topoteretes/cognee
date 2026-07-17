"""Structural fingerprinting for graph-topology entity resolution (issue #3630, Approach D).

A node's structural fingerprint is its typed neighborhood signature: the set of
(relationship_name, neighbor_id) pairs it participates in. Two nodes with
different textual names but a highly overlapping fingerprint are likely the
same real-world entity (e.g. "Apple" and "Apple Inc." both connect to
"Tim Cook" via CEO and "iPhone" via MAKES).

We use *typed* Jaccard similarity rather than plain neighbor-set overlap,
because two nodes can share a neighbor through different relationship types
(e.g. Apple --MAKES--> iPhone vs Apple --COMPETES_WITH--> iPhone), which
would false-positive under untyped overlap.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Tuple, Iterable, Any


@dataclass(frozen=True)
class StructuralFingerprint:
    """Typed neighborhood signature for a node.

    Attributes
    ----------
    node_id : str
        Identity of the node this fingerprint describes.
    neighbor_ids : FrozenSet[str]
        Ids of nodes this node connects to (either direction).
    typed_edges : FrozenSet[Tuple[str, str]]
        Set of (relationship_name, neighbor_id) pairs.
    """

    node_id: str
    neighbor_ids: FrozenSet[str] = field(default_factory=frozenset)
    typed_edges: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    @property
    def degree(self) -> int:
        return len(self.neighbor_ids)


def build_fingerprint(node_id: str, edges: Iterable[Tuple[str, str, str]]) -> StructuralFingerprint:
    """Build a StructuralFingerprint for `node_id` from an iterable of edges.

    Parameters
    ----------
    node_id : str
        The node to build a fingerprint for.
    edges : Iterable[Tuple[str, str, str]]
        Iterable of (source_id, target_id, relationship_name) tuples. Edges
        may be undirected in origin (i.e. either endpoint may equal node_id).

    Returns
    -------
    StructuralFingerprint
    """
    neighbor_ids = set()
    typed_edges = set()

    for source_id, target_id, relationship_name in edges:
        if source_id == node_id and target_id != node_id:
            neighbor_ids.add(target_id)
            typed_edges.add((relationship_name, target_id))
        elif target_id == node_id and source_id != node_id:
            neighbor_ids.add(source_id)
            typed_edges.add((relationship_name, source_id))

    return StructuralFingerprint(
        node_id=node_id,
        neighbor_ids=frozenset(neighbor_ids),
        typed_edges=frozenset(typed_edges),
    )


def structural_similarity(fp_a: StructuralFingerprint, fp_b: StructuralFingerprint) -> float:
    """Typed Jaccard similarity between two structural fingerprints.

    Returns a float in [0.0, 1.0]. 0.0 if both fingerprints are empty
    (no shared typed edges, no evidence either way).
    """
    if not fp_a.typed_edges and not fp_b.typed_edges:
        return 0.0

    intersection = fp_a.typed_edges & fp_b.typed_edges
    union = fp_a.typed_edges | fp_b.typed_edges

    if not union:
        return 0.0

    return len(intersection) / len(union)