"""Detect and merge semantically near-duplicate ``Entity`` nodes.

These two tasks back the ``consolidate_entities`` memify pipeline:

* :func:`detect_entity_duplicates` (extraction) loads every ``Entity`` node,
  embeds its name, and clusters near-duplicates by cosine similarity and/or
  normalized-name equality.
* :func:`merge_entity_duplicates` (enrichment) collapses each cluster into one
  canonical node: it re-points every edge from the duplicates onto the
  canonical (direction preserved), unions the descriptions, records a
  ``merged_from`` report, then deletes the duplicate nodes and their vector
  embeddings. ``dry_run`` computes and logs the plan without mutating anything.

The merge is backend-agnostic: it relies only on ``get_graph_data`` (directed
edges), ``add_edge`` (an UPSERT/MERGE on every adapter), ``add_nodes`` (an
upsert that updates existing node properties via ``ON MATCH SET``),
``delete_nodes`` (a detach-delete that cascades the duplicates' old edges), and
the vector engine's ``delete_data_points``. No per-edge delete primitive is
required or used.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models.Entity import Entity
from cognee.shared.logging_utils import get_logger

logger = get_logger("consolidate_entities")

# The vector collection that holds Entity name embeddings. Cognee names vector
# collections ``f"{type_name}_{index_field}"``; Entity declares
# ``index_fields=["name"]`` so its collection is ``Entity_name``.
ENTITY_VECTOR_COLLECTION = "Entity_name"

DEFAULT_CONFIG: Dict[str, Any] = {
    # Minimum cosine similarity between two entity-name embeddings to treat them
    # as the same entity.
    "similarity_threshold": 0.85,
    # EntityType names (or node types) that must never be merged.
    "protect_node_types": [],
    # When True, compute and log the merge plan but perform zero mutations.
    "dry_run": False,
    # Also treat entities whose normalized names are identical as duplicates
    # (catches e.g. "U.S.A." vs "USA", which hash to different node ids).
    "name_match": True,
    # Max neighbors per entity to consider during similarity clustering.
    "top_k": 10,
    # When False (default, conservative), only entities sharing the same
    # EntityType are merged together.
    "allow_cross_type": False,
}


def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the caller config over defaults and normalize derived fields."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update({key: value for key, value in config.items() if value is not None})
    cfg["protect_node_types"] = set(cfg.get("protect_node_types") or [])
    return cfg


def _normalize_name(name: Any) -> str:
    """Lower-case and strip every non-alphanumeric character from a name."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _cosine_similarity_matrix(vectors: List[List[float]]) -> np.ndarray:
    """Return the full pairwise cosine-similarity matrix for ``vectors``."""
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
    return normalized @ normalized.T


def _cluster_entities(
    members: List[Dict[str, Any]], vectors: List[List[float]], cfg: Dict[str, Any]
) -> List[List[Dict[str, Any]]]:
    """Group entities into duplicate clusters via union-find.

    Two entities are united when their name embeddings are at least
    ``similarity_threshold`` cosine-similar, or (when ``name_match`` is on) when
    their normalized names are identical. Unless ``allow_cross_type`` is set,
    entities are only united when they share the same EntityType.
    """
    count = len(members)
    if count < 2:
        return []

    threshold = cfg["similarity_threshold"]
    top_k = cfg["top_k"]
    name_match = cfg["name_match"]
    allow_cross_type = cfg["allow_cross_type"]

    similarity = _cosine_similarity_matrix(vectors)
    normalized_names = [_normalize_name(member["name"]) for member in members]

    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    def same_type(left: int, right: int) -> bool:
        return allow_cross_type or members[left]["type"] == members[right]["type"]

    # Pass 1 — cosine similarity, bounded to the top_k nearest neighbors per node.
    if similarity.size:
        for index in range(count):
            neighbors = sorted(
                (other for other in range(count) if other != index),
                key=lambda other: similarity[index][other],
                reverse=True,
            )
            if top_k:
                neighbors = neighbors[:top_k]
            for other in neighbors:
                if similarity[index][other] >= threshold and same_type(index, other):
                    union(index, other)

    # Pass 2 — exact normalized-name match (not bounded by top_k).
    if name_match:
        for index in range(count):
            if not normalized_names[index]:
                continue
            for other in range(index + 1, count):
                if normalized_names[index] == normalized_names[other] and same_type(index, other):
                    union(index, other)

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(members[index])

    return [cluster for cluster in clusters.values() if len(cluster) >= 2]


def _entity_type_map(
    entity_ids: set, edges: List[Tuple], nodes_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Optional[str]]:
    """Map each entity id to its EntityType name.

    Robust to the exact ``is_a`` relationship label: an entity's type is the
    name of any neighbor node whose ``type`` is ``"EntityType"``.
    """
    type_of: Dict[str, Optional[str]] = {}
    for edge in edges:
        source, target = str(edge[0]), str(edge[1])
        if source in entity_ids:
            target_props = nodes_by_id.get(target)
            if target_props and target_props.get("type") == "EntityType":
                type_of[source] = target_props.get("name")
    return type_of


async def detect_entity_duplicates(
    data: Any = None, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Find clusters of near-duplicate ``Entity`` nodes.

    Returns a payload ``{"clusters": [...], "edges": [...]}`` consumed by
    :func:`merge_entity_duplicates`. ``edges`` is the full directed edge list
    from the graph, forwarded so the merge step re-points edges without a
    second full scan. The ``data`` argument (the memify pipeline seed) is
    intentionally ignored — the entities are read straight from the graph.
    """
    cfg = _resolve_config(config)
    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    nodes, edges = await graph_engine.get_graph_data()
    nodes_by_id = {str(node_id): props for node_id, props in nodes}

    entity_ids = {
        str(node_id)
        for node_id, props in nodes
        if props.get("type") == "Entity" and props.get("name")
    }
    type_of = _entity_type_map(entity_ids, edges, nodes_by_id)

    members: List[Dict[str, Any]] = []
    for node_id, props in nodes:
        node_id = str(node_id)
        if node_id not in entity_ids:
            continue
        entity_type = type_of.get(node_id)
        if (
            entity_type in cfg["protect_node_types"]
            or props.get("type") in cfg["protect_node_types"]
        ):
            continue
        members.append(
            {
                "id": node_id,
                "name": props.get("name"),
                "type": entity_type,
                "created_at": props.get("created_at"),
                "description": props.get("description"),
                "belongs_to_set": props.get("belongs_to_set"),
                "props": props,
            }
        )

    if len(members) < 2:
        logger.info("consolidate_entities: fewer than 2 candidate entities; nothing to detect.")
        return {"clusters": [], "edges": edges}

    vectors = await vector_engine.embed_data([member["name"] for member in members])
    clusters = _cluster_entities(members, vectors, cfg)

    logger.info(
        "consolidate_entities: detected %d duplicate cluster(s) among %d entities.",
        len(clusters),
        len(members),
    )
    return {"clusters": clusters, "edges": edges}


def plan_edge_repointing(
    edges: List[Tuple], remap: Dict[str, str]
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Compute the edges to (re)create on canonical nodes.

    Given every directed graph edge and a ``{duplicate_id: canonical_id}`` map,
    both endpoints are remapped to their canonical. Edges untouched by the remap
    are skipped; edges that collapse into a self-loop on a canonical are dropped.
    The duplicates' original edges are removed later by the detach-delete
    cascade, so this function only needs to add the re-pointed copies. This is
    the backend-agnostic edge-move helper (it does not call any per-edge delete).
    """
    moved: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for edge in edges:
        source, target, relationship = str(edge[0]), str(edge[1]), edge[2]
        properties = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
        new_source = remap.get(source, source)
        new_target = remap.get(target, target)
        if new_source == source and new_target == target:
            continue  # neither endpoint is a duplicate
        if new_source == new_target:
            continue  # would become a self-loop on the canonical
        moved.append((new_source, new_target, relationship, properties))
    return moved


def _node_degrees(edges: List[Tuple]) -> Dict[str, int]:
    """Count incident edges (both directions) for every node id."""
    degree: Dict[str, int] = {}
    for edge in edges:
        for endpoint in (str(edge[0]), str(edge[1])):
            degree[endpoint] = degree.get(endpoint, 0) + 1
    return degree


def _pick_canonical(
    cluster: List[Dict[str, Any]], degree: Dict[str, int]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Pick the canonical member of a cluster.

    Deterministic rule: highest graph degree wins (it is the most connected,
    so re-pointing moves the fewest edges); ties break to the oldest
    ``created_at``; remaining ties break lexicographically by normalized name.
    """

    def sort_key(member: Dict[str, Any]):
        created_at = member.get("created_at")
        created_at = created_at if isinstance(created_at, (int, float)) else float("inf")
        return (-degree.get(member["id"], 0), created_at, _normalize_name(member["name"]))

    ordered = sorted(cluster, key=sort_key)
    return ordered[0], ordered[1:]


def _union_descriptions(members: List[Dict[str, Any]]) -> Optional[str]:
    """Concatenate distinct, non-empty descriptions in member order."""
    descriptions: List[str] = []
    for member in members:
        description = (member.get("description") or "").strip()
        if description and description not in descriptions:
            descriptions.append(description)
    return " ".join(descriptions) if descriptions else None


def _build_canonical_entity(
    canonical: Dict[str, Any], duplicates: List[Dict[str, Any]]
) -> Optional[Entity]:
    """Rebuild the canonical ``Entity`` with a unioned description.

    The node is reconstructed faithfully from its stored properties so that
    ``add_nodes`` (which overwrites the property blob via ``ON MATCH SET``)
    never drops existing data. The ``is_a`` relationship is preserved as a graph
    edge and is not part of the node blob, so it is omitted here. Returns
    ``None`` if reconstruction fails, in which case the caller leaves the
    canonical node untouched (no data loss).
    """
    props = dict(canonical.get("props") or {})
    props.pop("is_a", None)
    props["name"] = canonical["name"]

    unioned = _union_descriptions([canonical] + duplicates)
    if unioned is not None:
        props["description"] = unioned
    props.setdefault("description", "")

    try:
        return Entity.from_dict(props)
    except Exception as error:  # noqa: BLE001 - reconstruction is best-effort
        logger.warning(
            "consolidate_entities: could not rebuild canonical %s (%s); "
            "leaving its properties unchanged.",
            canonical.get("id"),
            error,
        )
        return None


async def merge_entity_duplicates(
    payload: Any, config: Optional[Dict[str, Any]] = None
) -> List[Entity]:
    """Collapse each detected cluster into a single canonical node.

    Steps per run (skipped entirely when ``dry_run`` is set):
      1. Re-point every duplicate's edges onto its canonical (direction kept).
      2. Union descriptions onto the canonical and persist via ``add_nodes``.
      3. Delete the duplicate nodes (cascading their old edges) and purge their
         name embeddings from the vector store.

    Returns the list of updated canonical entities (empty on ``dry_run`` or when
    nothing was merged). The ``merged_from`` provenance is logged as the merge
    report.
    """
    cfg = _resolve_config(config)
    dry_run = cfg["dry_run"]
    clusters, edges = _unwrap_payload(payload)

    if not clusters:
        logger.info("consolidate_entities: no duplicate clusters to merge.")
        return []

    degree = _node_degrees(edges)

    remap: Dict[str, str] = {}
    plans: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for cluster in clusters:
        canonical, duplicates = _pick_canonical(cluster, degree)
        for duplicate in duplicates:
            remap[duplicate["id"]] = canonical["id"]
        plans.append((canonical, duplicates))

    moved_edges = plan_edge_repointing(edges, remap)
    duplicate_ids = list(remap.keys())

    report = [
        {
            "canonical_id": canonical["id"],
            "canonical_name": canonical["name"],
            "merged_from": [
                {"id": duplicate["id"], "name": duplicate["name"]} for duplicate in duplicates
            ],
        }
        for canonical, duplicates in plans
    ]
    logger.info(
        "consolidate_entities plan: clusters=%d duplicates=%d edges_to_repoint=%d dry_run=%s",
        len(plans),
        len(duplicate_ids),
        len(moved_edges),
        dry_run,
    )
    logger.info("consolidate_entities merge report: %s", json.dumps(report, default=str))

    if dry_run:
        logger.info("consolidate_entities: dry_run=True — performing ZERO mutations.")
        return []

    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    # 1. Re-point edges onto the canonicals. add_edge is called positionally
    #    because adapters disagree on the 4th parameter's name
    #    (properties vs edge_properties); position is stable across all of them.
    for source, target, relationship, properties in moved_edges:
        await graph_engine.add_edge(source, target, relationship, properties or {})

    # 2. Persist unioned descriptions onto the canonicals (add_nodes upserts via
    #    ON MATCH SET). Membership in node sets is already unioned at the edge
    #    level by step 1, since belongs_to_set links are ordinary edges.
    updated_canonicals = [
        entity
        for canonical, duplicates in plans
        if duplicates
        for entity in [_build_canonical_entity(canonical, duplicates)]
        if entity is not None
    ]
    if updated_canonicals:
        await graph_engine.add_nodes(updated_canonicals)

    # 3. Delete the duplicates (detach-delete cascades their old edges) and
    #    purge their name embeddings, or stale vectors resurface in later runs.
    if duplicate_ids:
        await graph_engine.delete_nodes(duplicate_ids)
        await vector_engine.delete_data_points(
            ENTITY_VECTOR_COLLECTION, [UUID(duplicate_id) for duplicate_id in duplicate_ids]
        )

    logger.info(
        "consolidate_entities: merged %d duplicate(s) into %d canonical(s).",
        len(duplicate_ids),
        len(plans),
    )
    return updated_canonicals


def _unwrap_payload(payload: Any) -> Tuple[List[List[Dict[str, Any]]], List[Tuple]]:
    """Normalize the detect-task output into ``(clusters, edges)``.

    Tolerates the payload being wrapped in a single-element list by the
    pipeline runner.
    """
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return [], []
    return payload.get("clusters", []), payload.get("edges", [])
