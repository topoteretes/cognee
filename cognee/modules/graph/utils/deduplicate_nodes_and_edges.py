"""Deterministic, rule-based node/edge deduplication with merge (Approach A).

Previously this kept only the first node seen for a given id and dropped the
rest, so (a) duplicates that differ only by spelling survived as separate nodes
and (b) conflicting property values were discarded by insertion order.

This version blocks candidates by a versioned canonical key
(:func:`canonical_block_key`), merges each block into one surviving node using
explicit per-field policies applied in a deterministic order, rewrites edges
onto the surviving id, and records provenance (``merged_from`` + canonical key)
on the survivor. The survivor keeps its original UUID5 so existing graph
references stay valid.
"""

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.canonicalize import CANON_VERSION, canonical_block_key


def _merge_field(existing, incoming):
    """Merge one field value deterministically.

    - list/tuple/set -> ``union`` (order-preserving, no information lost)
    - scalars -> ``non_null_wins`` (keep survivor's value unless it is empty)
    """
    if isinstance(existing, (list, tuple, set)):
        merged = list(existing)
        seen = {repr(item) for item in merged}
        for item in incoming or []:
            key = repr(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged if isinstance(existing, list) else type(existing)(merged)

    # non_null_wins: survivor's non-empty value is authoritative.
    if existing in (None, "", [], {}):
        return incoming
    return existing


def _merge_group(members: list[DataPoint]) -> DataPoint:
    """Merge a block of duplicate nodes into one surviving node.

    Members are sorted by ``str(id)`` for reproducibility (never asynchronous
    arrival order); the first is the survivor. Remaining members are folded in
    field-by-field and recorded as provenance.
    """
    ordered = sorted(members, key=lambda n: str(n.id))
    survivor = ordered[0].model_copy(deep=True)

    if len(ordered) == 1:
        return survivor

    merged_from: list[str] = []
    for other in ordered[1:]:
        other_id = str(other.id)
        if other_id != str(survivor.id) and other_id not in merged_from:
            merged_from.append(other_id)
        for field_name in type(survivor).model_fields:
            if field_name in ("id", "metadata"):
                continue
            merged_value = _merge_field(
                getattr(survivor, field_name, None),
                getattr(other, field_name, None),
            )
            setattr(survivor, field_name, merged_value)

    # Provenance / auditability: what merged in, and under which rule version.
    if merged_from and isinstance(getattr(survivor, "metadata", None), dict):
        provenance = dict(survivor.metadata)
        provenance["merged_from"] = merged_from
        provenance["canonical_version"] = CANON_VERSION
        survivor.metadata = provenance

    return survivor


def deduplicate_nodes_and_edges(nodes: list[DataPoint], edges: list):
    # Block nodes by canonical key, preserving first-seen order for determinism.
    groups: dict[tuple, list[DataPoint]] = {}
    order: list[tuple] = []
    for node in nodes:
        key = canonical_block_key(node)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(node)

    final_nodes: list[DataPoint] = []
    id_remap: dict[str, object] = {}
    for key in order:
        survivor = _merge_group(groups[key])
        final_nodes.append(survivor)
        for member in groups[key]:
            if str(member.id) != str(survivor.id):
                id_remap[str(member.id)] = survivor.id

    # Rewrite edges onto surviving ids, then dedup by (source, rel, target).
    final_edges = []
    seen_edges = set()
    for edge in edges:
        source = id_remap.get(str(edge[0]), edge[0])
        target = id_remap.get(str(edge[1]), edge[1])
        remapped = (source, target) + tuple(edge[2:])
        edge_key = str(source) + str(edge[2]) + str(target)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            final_edges.append(remapped)

    return final_nodes, final_edges
