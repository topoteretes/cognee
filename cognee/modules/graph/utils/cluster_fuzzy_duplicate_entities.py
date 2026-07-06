"""Embedding-similarity fuzzy dedup for Entity nodes (issue #3628, Approach B).

Runs as a pre-processing step *before* ``deduplicate_nodes_and_edges`` in
``add_data_points``. It finds Entity nodes whose ``name`` embeddings are near
each other (semantic variants like "OpenAI" / "OpenAI Inc.") and collapses them
onto a single canonical node by rewriting ids — so the existing id-based
dedup/upsert path merges them for free, with no graph surgery.

Canonical rule (decision (a)): the **first-appearing** node wins.
  * In-batch cluster  → canonical is the earliest node in ``nodes``; duplicates
    get their ``.id`` rewritten to it and stay in the list (downstream dedup
    collapses them). Provenance goes on the canonical's ``metadata["merged_from"]``.
  * Cross-batch hit   → canonical is the pre-existing graph node (earliest of
    all). The batch duplicate is **removed** from ``nodes`` (never upserted, so
    it can't overwrite the existing node) and its edges are remapped. Provenance
    goes into the returned ``merge_records`` + a log line (the canonical isn't in
    this batch, so there's nothing to attach ``merged_from`` to).
"""

from typing import Any

import numpy as np

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("cluster_fuzzy_duplicate_entities")

# Entity name embeddings live in the "{type}_{field}" collection.
ENTITY_NAME_COLLECTION = "Entity_name"


def _is_entity(node: DataPoint) -> bool:
    return type(node).__name__ == "Entity" and bool(getattr(node, "name", None))


def _cosine_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for a zero vector
    normalized = matrix / norms
    return normalized @ normalized.T


class _UnionFind:
    """Union-Find that always keeps the *smallest* index as the root, so a
    cluster's root is its first-appearing member."""

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


def _remap(value: Any, id_map: dict[str, Any]) -> Any:
    """Replace an id with its canonical, preserving str/UUID element type."""
    canonical = id_map.get(str(value))
    if canonical is None:
        return value
    return str(canonical) if isinstance(value, str) else canonical


async def cluster_fuzzy_duplicate_entities(
    nodes: list[DataPoint],
    edges: list[tuple],
    vector_engine,
    similarity_threshold: float = 0.72,
    top_k: int = 5,
) -> tuple[list[DataPoint], list[tuple], list[dict]]:
    """Merge fuzzy-duplicate Entity nodes by rewriting ids to a canonical node.

    Returns the (possibly shrunk) ``nodes``, the id-remapped ``edges``, and a
    list of ``merge_records`` describing every merge for auditing.
    """
    merge_records: list[dict] = []

    entity_indices = [i for i, node in enumerate(nodes) if _is_entity(nodes[i])]
    if len(entity_indices) == 0:
        return nodes, edges, merge_records

    entities = [nodes[i] for i in entity_indices]
    names = [entity.name for entity in entities]
    batch_ids = {str(entity.id) for entity in entities}

    # 1 + 2. In-batch clustering via all-pairs cosine over this batch's names.
    vectors = await vector_engine.embed_data(names)
    sim = _cosine_similarity_matrix(vectors)

    uf = _UnionFind(len(entities))
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if sim[i][j] >= similarity_threshold:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(entities)):
        clusters.setdefault(uf.find(i), []).append(i)

    # 3. Cross-batch: search each cluster representative against existing graph
    #    entities. First ingest has no collection yet -> treat as no hits.
    rep_indices = sorted(clusters.keys())  # cluster roots = first-appearing members
    cross_hit: dict[int, Any] = {}  # rep index -> (existing_id, existing_name, similarity)
    try:
        search_results = await vector_engine.batch_search(
            ENTITY_NAME_COLLECTION,
            query_texts=[entities[r].name for r in rep_indices],
            limit=top_k,
            include_payload=True,
        )
    except CollectionNotFoundError:
        search_results = None

    if search_results:
        for rep, results in zip(rep_indices, search_results):
            for scored in results or []:
                if str(scored.id) in batch_ids:
                    continue  # never match our own not-yet-indexed batch nodes
                similarity = 1.0 - scored.score  # built-in adapters: score = cosine distance
                if similarity >= similarity_threshold:
                    existing_name = (scored.payload or {}).get("name")
                    cross_hit[rep] = (scored.id, existing_name, similarity)
                    break  # top result over threshold wins

    id_map: dict[str, Any] = {}
    remove_ids: set[str] = set()

    for root, members in clusters.items():
        if root in cross_hit:
            existing_id, existing_name, cross_sim = cross_hit[root]
            # Whole cluster is a duplicate of a pre-existing node: map every
            # member onto it and drop them from the batch (don't overwrite it).
            for m in members:
                node = entities[m]
                id_map[str(node.id)] = existing_id
                remove_ids.add(str(node.id))
                member_sim = cross_sim if m == root else float(sim[root][m])
                merge_records.append(
                    {
                        "duplicate_id": str(node.id),
                        "duplicate_name": node.name,
                        # Losing property value, kept for conflict-resolution audit:
                        # canonical (first-appearing) wins, this value is discarded.
                        "duplicate_description": getattr(node, "description", None),
                        "canonical_id": str(existing_id),
                        "canonical_name": existing_name,
                        "similarity": member_sim,
                        "method": "embedding_similarity",
                        "threshold": similarity_threshold,
                        "scope": "cross_batch",
                    }
                )
                logger.info(
                    "Fuzzy-merged entity into existing graph node",
                    extra={
                        "duplicate_id": str(node.id),
                        "duplicate_name": node.name,
                        "canonical_id": str(existing_id),
                        "similarity": member_sim,
                    },
                )
            continue

        if len(members) == 1:
            continue  # lone entity, nothing to merge

        # In-batch cluster: earliest member is canonical; rewrite the rest.
        canonical = entities[root]
        merged_from = list(canonical.metadata.get("merged_from", []))
        for m in members:
            if m == root:
                continue
            node = entities[m]
            original_id = str(node.id)
            member_sim = float(sim[root][m])
            id_map[original_id] = canonical.id
            node.id = canonical.id  # dedup will collapse the now-identical ids
            record = {
                "duplicate_id": original_id,
                "duplicate_name": node.name,
                # Losing property value (canonical keeps its own), kept for audit.
                "duplicate_description": getattr(node, "description", None),
                "canonical_id": str(canonical.id),
                "canonical_name": canonical.name,
                "similarity": member_sim,
                "method": "embedding_similarity",
                "threshold": similarity_threshold,
                "scope": "in_batch",
            }
            merge_records.append(record)
            merged_from.append(
                {
                    "duplicate_id": record["duplicate_id"],
                    "duplicate_name": record["duplicate_name"],
                    "duplicate_description": record["duplicate_description"],
                    "similarity": member_sim,
                    "method": "embedding_similarity",
                    "threshold": similarity_threshold,
                }
            )
        if merged_from:
            canonical.metadata["merged_from"] = merged_from  # type: ignore[typeddict-unknown-key]

    if not id_map:
        return nodes, edges, merge_records

    # 6. Rewrite edges: tuple endpoints AND the ids embedded in the properties dict.
    remapped_edges: list[tuple] = []
    for edge in edges:
        source, target, relationship, *rest = edge
        properties = rest[0] if rest else None
        if isinstance(properties, dict):
            properties = dict(properties)
            if "source_node_id" in properties:
                properties["source_node_id"] = _remap(properties["source_node_id"], id_map)
            if "target_node_id" in properties:
                properties["target_node_id"] = _remap(properties["target_node_id"], id_map)
        new_edge = (_remap(source, id_map), _remap(target, id_map), relationship)
        if rest:
            new_edge = new_edge + (properties, *rest[1:])
        remapped_edges.append(new_edge)

    # Drop cross-batch duplicates so they never reach upsert.
    if remove_ids:
        nodes = [node for node in nodes if str(node.id) not in remove_ids]

    return nodes, remapped_edges, merge_records
