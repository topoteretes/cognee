"""Embedding-similarity fuzzy duplicate resolution — issue #3628, Approach B.

Two entities extracted under different surface names — "OpenAI" and "OpenAI Inc."
— are often the same real-world thing when their names embed close together in
vector space. The deterministic id-based dedup (``deduplicate_nodes_and_edges``)
only collapses names that normalise to the same string (case / spaces /
apostrophes); it misses these semantic variants. This module finds them within an
ingestion batch and links each pair with a ``merged_into`` edge (duplicate ->
canonical), so the duplication is recorded in the graph itself —
provenance-preserving, non-destructive (both nodes and all their edges are kept),
and reversible (drop the ``merged_into`` edge to undo).

Detection reuses cognee's own embeddings: embed the batch's Entity names, cluster
them by cosine similarity above ``similarity_threshold`` (Union-Find, so a chain
"a~b, b~c" groups a, b and c together), and link every non-canonical member to
the first-appearing member of its cluster. Deterministic given the embeddings, so
it stays reproducible under the mocked-embedding CI harness (#3601), with no LLM
call.

It extends the existing dedup path — called from ``add_data_points`` right after
``deduplicate_nodes_and_edges`` — rather than forking a new pipeline, and is
opt-in via ``add_data_points(..., fuzzy_entity_dedup=True)``. Consuming
``merged_into`` at query time (collapsing duplicates during retrieval) and
matching against entities already persisted in earlier batches/runs are
deliberately left to a follow-up, once the sibling approaches (A–E) are compared
on a shared evaluation fixture.
"""

from typing import Any

import numpy as np

from cognee.infrastructure.engine import DataPoint

# Graph edges flow through the pipeline as
# (source_node_id, target_node_id, relationship_name, properties) tuples.
Edge = tuple[Any, Any, str, dict]

# The synthetic relationship that records a fuzzy merge.
MERGED_INTO_RELATIONSHIP = "merged_into"

# Names whose embeddings are at least this cosine-similar are treated as the same
# entity. Calibrated for OpenAI text-embedding-3-large on the issue #3628 fixture
# (precision stays 1.0 down to ~0.45; recall trades off as the threshold rises).
# A different embedding model may want a different value, hence the override.
FUZZY_DEDUP_THRESHOLD = 0.72


def _is_entity(node: Any) -> bool:
    """Only ``Entity`` nodes with a name take part — cheap blocking that keeps
    chunks, types and other structural nodes out of the candidate set."""
    return type(node).__name__ == "Entity" and bool(getattr(node, "name", None))


def _cosine_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """All-pairs cosine similarity of ``vectors`` as an NxN matrix."""
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for a zero vector
    normalized = matrix / norms
    return normalized @ normalized.T


class _UnionFind:
    """Union-Find that always keeps the smallest index as a cluster's root, so
    the root is the cluster's first-appearing member."""

    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # Smaller index becomes the root -> first-appearing node is canonical.
        if root_a < root_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_a] = root_b


async def resolve_fuzzy_duplicate_entities(
    nodes: list[DataPoint],
    vector_engine,
    *,
    similarity_threshold: float = FUZZY_DEDUP_THRESHOLD,
) -> list[Edge]:
    """Link fuzzy-duplicate Entity nodes in a batch with ``merged_into`` edges.

    Args:
        nodes: The batch's nodes (``DataPoint``s), after identity dedup.
        vector_engine: Supplies ``embed_data`` — cognee's own embedding engine.
        similarity_threshold: Minimum cosine similarity between two entity names
            for them to count as duplicates.

    Returns:
        New ``merged_into`` edges (duplicate -> canonical), one per non-canonical
        cluster member, each carrying ``similarity_score`` and ``resolution`` in
        its properties. Empty when fewer than two entity names cross the
        threshold. ``nodes`` is never mutated.
    """
    # Narrowed to name-bearing Entity nodes by _is_entity; typed loosely because
    # `name` lives on the Entity subclass, not the DataPoint base.
    entities: list[Any] = [node for node in nodes if _is_entity(node)]
    if len(entities) < 2:
        return []

    vectors = await vector_engine.embed_data([entity.name for entity in entities])
    similarity = _cosine_similarity_matrix(vectors)

    union_find = _UnionFind(len(entities))
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if similarity[i][j] >= similarity_threshold:
                union_find.union(i, j)

    merge_edges: list[Edge] = []
    for member in range(len(entities)):
        canonical = union_find.find(member)
        if canonical == member:
            continue  # cluster root (first-appearing member) is the canonical node
        merge_edges.append(
            (
                entities[member].id,
                entities[canonical].id,
                MERGED_INTO_RELATIONSHIP,
                {
                    "similarity_score": round(float(similarity[canonical][member]), 4),
                    "resolution": "embedding_similarity",
                },
            )
        )
    return merge_edges
