"""decay_memory — a memify/improve task that makes memory *fade*.

Decays each node's `feedback_weight` over time (exponential half-life from `updated_at`/`created_at`),
then prunes nodes that fall below a staleness threshold, preferring orphans (degree-one / disconnected).
Today Cognee only has full `prune_*` wipes and targeted `forget()`; this adds gradual, time-based decay.

Design notes:
- Pure time-decay first: frequency/usage weights are non-functional today (see the RE-WEIGHT ticket),
  so this decays `feedback_weight` purely by age. Usage-based decay can layer on once RE-WEIGHT lands.
- `dry_run=True` is the default: it reports what *would* be decayed/pruned without mutating the graph,
  mirroring `cleanup_unused_data`'s dry-run safety.
- Composes existing primitives only: `get_graph_data`, `set_node_feedback_weights`, `delete_nodes`
  (so it sits on top of, and doesn't touch, the in-flight graph delete/rollback refactor).
"""

import time
from typing import Any, Optional, TypedDict

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger("decay_memory")

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_FEEDBACK_WEIGHT = 0.5  # matches DataPoint.feedback_weight default
_MS_PER_DAY = 24 * 60 * 60 * 1000


# --------------------------------------------------------------------------------------
# Pure core (no I/O — unit-tested in isolation; this is the deterministic-aging guarantee)
# --------------------------------------------------------------------------------------
def decay_weight(weight: float, age_ms: int, half_life_ms: int) -> float:
    """Exponential half-life decay: ``weight * 0.5 ** (age / half_life)``.

    Deterministic for a given (weight, age, half_life). No decay for age<=0 or half_life<=0.
    """
    if half_life_ms <= 0 or age_ms <= 0:
        return weight
    return weight * (0.5 ** (age_ms / half_life_ms))


def node_degrees(node_ids, edges) -> dict:
    """``edges`` are (source_id, target_id, ...) tuples. Returns {node_id: degree} for node_ids.

    Self-loops (source == target — Cognee writes a ``SELF`` edge per node) are ignored: they don't
    connect a node to anything else, so a node whose only edge is its self-loop counts as an orphan.
    """
    degrees = {nid: 0 for nid in node_ids}
    for edge in edges:
        if not edge:
            continue
        source = edge[0]
        target = edge[1] if len(edge) > 1 else None
        if source == target:
            continue
        if source in degrees:
            degrees[source] += 1
        if target in degrees:
            degrees[target] += 1
    return degrees


def select_prune(decayed_weights: dict, degrees: dict, min_weight: float) -> list:
    """Node ids to prune: below ``min_weight`` AND an orphan / degree-one node (degree <= 1).

    Restricting to leaves/orphans means a prune only ever removes a stale *leaf* — it can never delete
    a connector and orphan a live subgraph. Stale hubs are left in place (they keep decaying, and become
    prunable once their neighbours are gone). Ordered orphan-first (lowest degree, then weight).
    """
    candidates = [
        nid
        for nid, weight in decayed_weights.items()
        if weight < min_weight and degrees.get(nid, 0) <= 1
    ]
    candidates.sort(key=lambda nid: (degrees.get(nid, 0), decayed_weights[nid]))
    return candidates


def _node_age_ms(props: dict, now_ms: int) -> int:
    timestamp = props.get("updated_at") or props.get("created_at")
    if not isinstance(timestamp, (int, float)) or timestamp <= 0:
        return 0
    age = now_ms - int(timestamp)
    return age if age > 0 else 0


def _node_type(props: dict):
    metadata = props.get("metadata") if isinstance(props.get("metadata"), dict) else {}
    return props.get("type") or props.get("node_type") or metadata.get("type")


class DecayMemoryResult(TypedDict):
    scanned: int
    decayed: int
    pruned: int
    dry_run: bool
    pruned_ids: list


# --------------------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------------------
async def decay_memory(
    data: Any = None,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    protect_node_types: Optional[list] = None,
    dry_run: bool = True,
    now_ms: Optional[int] = None,
) -> DecayMemoryResult:
    """Age node ``feedback_weight`` by half-life, then prune nodes below ``min_weight`` (orphans first).

    Parameters
    ----------
    data:
        Accepted for memify-pipeline compatibility; ignored (this task walks the whole graph).
    half_life_days:
        Time for a node's weight to halve from disuse.
    min_weight:
        Nodes whose decayed weight falls below this are pruned.
    protect_node_types:
        Node types that are never decayed or pruned (e.g. structural / EntityType nodes).
    dry_run:
        When True (default), report what *would* change without mutating the graph.
    now_ms:
        Override the clock (epoch ms) — for deterministic tests.
    """
    protected = set(protect_node_types or [])
    half_life_ms = max(0, int(half_life_days * _MS_PER_DAY))
    now = now_ms if now_ms is not None else int(time.time() * 1000)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    # Collect scannable nodes (skip protected types); keep props for the age timestamp.
    scannable: dict = {}
    for node_id, props in nodes:
        props = props if isinstance(props, dict) else {}
        if _node_type(props) in protected:
            continue
        scannable[node_id] = props

    # Read AUTHORITATIVE current weights via the graph engine (same method apply_feedback_weights
    # uses) rather than trusting the serialized property blob — only-found ids return, 0.5 otherwise.
    stored_weights = await graph_engine.get_node_feedback_weights(list(scannable.keys()))

    decayed_weights: dict = {}
    updates: dict = {}
    for node_id, props in scannable.items():
        try:
            current = float(stored_weights.get(node_id, DEFAULT_FEEDBACK_WEIGHT))
        except (TypeError, ValueError):
            current = DEFAULT_FEEDBACK_WEIGHT
        new_weight = decay_weight(current, _node_age_ms(props, now), half_life_ms)
        decayed_weights[node_id] = new_weight
        if new_weight != current:
            updates[node_id] = new_weight

    scanned = len(scannable)
    degrees = node_degrees(list(decayed_weights.keys()), edges)
    prune_ids = select_prune(decayed_weights, degrees, min_weight)

    # don't bother writing a decayed weight for a node we're about to delete
    prune_set = set(prune_ids)
    weight_updates = {nid: w for nid, w in updates.items() if nid not in prune_set}

    if not dry_run:
        if weight_updates:
            await graph_engine.set_node_feedback_weights(weight_updates)
        if prune_ids:
            await graph_engine.delete_nodes(prune_ids)

    logger.info(
        "decay_memory: scanned=%d decayed=%d pruned=%d dry_run=%s",
        scanned,
        len(weight_updates),
        len(prune_ids),
        dry_run,
    )
    return {
        "scanned": scanned,
        "decayed": len(weight_updates),
        "pruned": len(prune_ids),
        "dry_run": dry_run,
        "pruned_ids": prune_ids,
    }
