"""Group and apply distributed graph writes while preserving graph provenance.

Distributed graph writes arrive on ``add_nodes_and_edges_queue`` as
``(nodes, edges, source_ref_key, pipeline_run_id)`` items. The provenance stamp
(``source_ref_key`` + ``pipeline_run_id``) is per data item, so batches from
different items must not be merged into one write — a single fold key applied to
a merged batch would mis-attribute the other items' artifacts.

So the worker groups accumulated items by their provenance key and folds each
group independently, but still writes ALL nodes before ANY edge (across groups)
so a cross-item edge always finds its endpoint nodes already present. Items
without provenance (``source_ref_key is None`` — non-provenance backends or
non-attributed writes) group under ``(None, None)`` and are written unfolded,
exactly as before.

This module is import-light on purpose (no Modal) so the grouping/ordering logic
is unit-testable without the worker's Modal runtime.
"""

from typing import Awaitable, Callable, Dict, List, Optional, Tuple

# (source_ref_key, pipeline_run_id) -> (nodes, edges)
ProvenanceKey = Tuple[Optional[str], Optional[str]]
GraphWriteGroups = Dict[ProvenanceKey, Tuple[List, List]]


def group_graph_writes(items) -> GraphWriteGroups:
    """Group ``(nodes, edges, source_ref_key, pipeline_run_id)`` items by provenance key.

    Insertion order is preserved (dict keeps first-seen order) so writes stay
    deterministic across a batch.
    """
    groups: GraphWriteGroups = {}
    for nodes, edges, source_ref_key, pipeline_run_id in items:
        key = (source_ref_key, pipeline_run_id)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].extend(nodes)
        groups[key][1].extend(edges)
    return groups


async def apply_grouped_graph_writes(
    groups: GraphWriteGroups,
    add_nodes: Callable[[List, Optional[str], Optional[str]], Awaitable[None]],
    add_edges: Callable[[List, Optional[str], Optional[str]], Awaitable[None]],
) -> None:
    """Write every group's nodes, then every group's edges (nodes-before-edges).

    ``add_nodes`` / ``add_edges`` are async callables taking
    ``(batch, source_ref_key, pipeline_run_id)`` — the worker supplies its
    deadlock-retrying, engine-bound writers.
    """
    for (source_ref_key, pipeline_run_id), (nodes, _edges) in groups.items():
        if nodes:
            await add_nodes(nodes, source_ref_key, pipeline_run_id)
    for (source_ref_key, pipeline_run_id), (_nodes, edges) in groups.items():
        if edges:
            await add_edges(edges, source_ref_key, pipeline_run_id)
