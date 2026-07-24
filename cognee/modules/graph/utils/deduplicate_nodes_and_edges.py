from typing import Dict, List, Optional, Tuple

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.canonicalization import canonicalize_entity_name
from cognee.modules.graph.utils.merge_policy import MergePolicy
from cognee.modules.graph.models.MergeRecord import MergeRecord


def deduplicate_nodes_and_edges(
    nodes: List[DataPoint],
    edges: List[dict],
    alias_map: Optional[Dict[str, str]] = None,
    merge_policy: Optional[MergePolicy] = None,
) -> Tuple[List[DataPoint], List[dict], List[MergeRecord]]:
    """
    Deduplicate nodes and edges.
    
    Returns:
        final_nodes: Deduplicated nodes.
        final_edges: Deduplicated edges with normalized keys.
        merge_records: Audit trail of node merges.
    """
    if merge_policy is None:
        merge_policy = MergePolicy()

    # 1. Canonicalization Phase
    # We update the name attribute if canonicalization changes it.
    for node in nodes:
        if hasattr(node, "name"):
            new_name = canonicalize_entity_name(node.name, alias_map=alias_map)
            if new_name != node.name:
                object.__setattr__(node, "name", new_name)

    # 2. Node Deduplication
    seen_nodes: Dict[str, DataPoint] = {}
    final_nodes: List[DataPoint] = []
    merge_records: List[MergeRecord] = []

    for node in nodes:
        key = str(node.id)
        if key not in seen_nodes:
            seen_nodes[key] = node
            final_nodes.append(node)
        else:
            survivor = seen_nodes[key]
            # Since IDs embed types, a collision here means the types are identical.
            # We can directly merge.
            _, field_resolutions = merge_policy.merge_nodes(survivor, node)
            if field_resolutions:
                merge_records.append(MergeRecord(
                    survivor_id=survivor.id,
                    absorbed_id=node.id,
                    merge_reason="in_batch_dedup",
                    field_resolutions=field_resolutions,
                ))

    # 3. Edge Deduplication
    added_edge_keys = {}
    final_edges = []

    for edge in edges:
        # Edge format: (source_id, target_id, relationship_name, properties)
        # Use proper delimiter to avoid collision
        edge_key = f"{edge[0]}|{edge[2]}|{edge[1]}"
        if edge_key not in added_edge_keys:
            final_edges.append(edge)
            added_edge_keys[edge_key] = True

    return final_nodes, final_edges, merge_records
