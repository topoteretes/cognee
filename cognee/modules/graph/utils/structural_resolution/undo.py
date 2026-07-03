"""Reversible merge undo for structural dedup (issue #3630, Approach D).

Merges applied by `apply_structural_merges` are non-destructive: the
absorbed node's data is preserved in the `MergeRecord` snapshot (and the
node itself stays in storage tagged `merged_into`). `undo_merge` restores
both the original node and its edges from that snapshot.
"""

from typing import Any, Dict, List, Optional, Tuple

from .merge_execution import MergeRecord


def undo_merge(
    merge_record: MergeRecord,
    nodes: List[Any],
    edges: List[Tuple[str, str, str, dict]],
    now_ms: int,
) -> Tuple[List[Any], List[Tuple[str, str, str, dict]], MergeRecord]:
    """Reverse a single merge, restoring the absorbed node and its edges.

    Parameters
    ----------
    merge_record : MergeRecord
        The record produced by `apply_structural_merges` for this merge.
    nodes : List[Any]
        Current node list (containing the canonical node, and — because
        merges are non-destructive — the absorbed node marked
        `merged_into`).
    edges : List[Tuple[str, str, str, dict]]
        Current edge list, with edges already repointed to the canonical id.
    now_ms : int
        Timestamp (epoch ms) to stamp on the record as `reversed_at`.

    Returns
    -------
    Tuple[List[Any], List[Tuple], MergeRecord]
        Updated nodes, updated edges, and the merge record marked reversed.

    Raises
    ------
    ValueError
        If the merge record has no snapshot to restore from, or the merge
        was already reversed.
    """
    if merge_record.reversed_at is not None:
        raise ValueError(
            f"Merge {merge_record.merged_id} -> {merge_record.canonical_id} "
            "was already reversed."
        )

    if merge_record.merged_node_snapshot is None:
        raise ValueError(
            "Cannot undo merge: no snapshot available for "
            f"{merge_record.merged_id}."
        )

    canonical_id = merge_record.canonical_id
    absorbed_id = merge_record.merged_id

    # Clear the merged_into marker on the absorbed node, and remove the
    # canonical node's reference to it in source_node_ids.
    nodes_by_id = {str(n.id): n for n in nodes}
    absorbed_node = nodes_by_id.get(absorbed_id)
    canonical_node = nodes_by_id.get(canonical_id)

    if absorbed_node is not None:
        try:
            absorbed_node.merged_into = None
        except (AttributeError, ValueError):
            object.__setattr__(absorbed_node, "merged_into", None)

    if canonical_node is not None:
        existing_sources = getattr(canonical_node, "source_node_ids", None) or []
        updated_sources = [s for s in existing_sources if s != absorbed_id]
        try:
            canonical_node.source_node_ids = updated_sources
        except (AttributeError, ValueError):
            object.__setattr__(canonical_node, "source_node_ids", updated_sources)

    # Restore edges that were repointed away from the absorbed node.
    restored_edges = list(edges)
    if merge_record.merged_node_edges_snapshot:
        # Remove any edge currently pointing at canonical_id that matches
        # the shape of a snapshot edge (i.e. was repointed from absorbed_id),
        # then re-add the original absorbed-id edges.
        snapshot_edges = merge_record.merged_node_edges_snapshot
        restored_edges = [
            e for e in restored_edges
            if not _is_repointed_version(e, snapshot_edges, canonical_id, absorbed_id)
        ]
        restored_edges.extend(snapshot_edges)

    merge_record.reversed_at = now_ms

    return list(nodes_by_id.values()), restored_edges, merge_record


def _is_repointed_version(
    edge: Tuple[str, str, str, dict],
    snapshot_edges: List[Tuple[str, str, str, dict]],
    canonical_id: str,
    absorbed_id: str,
) -> bool:
    """Return True if `edge` is the canonical-id-repointed version of one
    of the original absorbed-id edges in `snapshot_edges`."""
    source_id, target_id, relationship_name, _attrs = edge
    for snap_source, snap_target, snap_rel, _snap_attrs in snapshot_edges:
        if snap_rel != relationship_name:
            continue
        source_matches = (
            (source_id == canonical_id and snap_source == absorbed_id)
            or source_id == snap_source
        )
        target_matches = (
            (target_id == canonical_id and snap_target == absorbed_id)
            or target_id == snap_target
        )
        if source_matches and target_matches:
            return True
    return False