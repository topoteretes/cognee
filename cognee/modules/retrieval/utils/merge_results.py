"""Combine two retrievals of the same result shape into one ranked list.

Used when a single turn retrieves twice — a session turn runs the raw question and a
conversational rewrite of it — and the two result sets have to become one before the
retriever formats its context. Retrievers own the merge for their own shape and call
these helpers; nothing here knows what a retriever's objects mean.
"""

from collections.abc import Callable, Hashable
from typing import Any, Optional

from cognee.modules.retrieval.hybrid.results import display_value, result_id


def conversational_reserve(limit: Optional[int]) -> int:
    """Slots held for results only the conversational lane found.

    Reserves roughly a third of the budget. Nothing is reserved below three slots: one
    reserved slot out of two would hand half the budget to the rewrite and leave a
    single hit for the question actually asked.
    """
    if not limit or limit <= 2:
        return 0
    return max(1, limit // 3)


def edge_identity(edge: Any) -> Hashable:
    """Identify a graph edge by its object id, else by the relationship it expresses."""
    if isinstance(edge, dict):
        attributes = edge.get("attributes") or edge.get("edge_attributes") or edge
        source = edge.get("source_id") or edge.get("source_node_id")
        target = edge.get("target_id") or edge.get("target_node_id")
        directed = edge.get("directed", True)
    else:
        attributes = getattr(edge, "attributes", {}) or {}
        source = getattr(getattr(edge, "node1", None), "id", None)
        target = getattr(getattr(edge, "node2", None), "id", None)
        directed = getattr(edge, "directed", True)

    if not isinstance(attributes, dict):
        attributes = {}

    edge_object_id = display_value(attributes.get("edge_object_id"))
    if edge_object_id:
        return ("edge", edge_object_id)

    relationship = display_value(
        attributes.get("relationship_name") or attributes.get("relationship")
    )
    return (
        "graph",
        display_value(source) or "",
        relationship or "",
        display_value(target) or "",
        bool(directed),
    )


def _rank_key(record: list) -> tuple[int, int]:
    primary_rank, secondary_rank, _item = record
    if primary_rank is not None and secondary_rank is not None:
        return (0, primary_rank)
    if primary_rank is not None:
        return (1, primary_rank)
    return (2, secondary_rank)


def merge_ranked(
    primary: Optional[list],
    secondary: Optional[list],
    *,
    identity: Callable[[Any], Optional[Hashable]] = result_id,
    limit: Optional[int] = None,
    secondary_reserve: int = 0,
) -> list:
    """Merge two ranked lists, strongest agreement first.

    Items returned by *both* lists come first — appearing in both is the strongest
    signal — ordered by their rank in ``primary``. Then the remaining ``primary`` items
    in their original order, then ``secondary`` items not already present. An item in
    both lists is represented by its ``primary`` object. Results without an identity are
    never merged away. A single list therefore comes back in its original order.

    When ``limit`` is set, up to ``secondary_reserve`` slots are held for items found
    only by the secondary lane. Unused reserve goes back to the primary side so
    single-lane and short-lane behaviour is unchanged. The total never exceeds ``limit``.
    """
    # identity -> [rank in primary, rank in secondary, representative object]
    records: dict[Hashable, list] = {}
    for lane, results in enumerate((primary or [], secondary or [])):
        for rank, item in enumerate(results):
            key = identity(item)
            if key is None:
                key = ("unidentified", lane, rank)
            record = records.setdefault(key, [None, None, item])
            if record[lane] is not None:
                continue
            record[lane] = rank
            if lane == 0:
                record[2] = item

    ranked = sorted(records.values(), key=_rank_key)
    if limit is None:
        return [record[2] for record in ranked]
    limit = max(0, limit)
    reserve = min(max(0, secondary_reserve), limit)

    # A record with no primary rank was found only by the secondary lane.
    secondary_indices = [i for i, record in enumerate(ranked) if record[0] is None]
    primary_indices = [i for i, record in enumerate(ranked) if record[0] is not None]

    take_secondary = secondary_indices[:reserve]
    take_primary = primary_indices[: limit - len(take_secondary)]
    # Unused reserve returns to the primary side above; unused primary slots here.
    shortfall = limit - len(take_primary) - len(take_secondary)
    if shortfall > 0:
        take_secondary += secondary_indices[len(take_secondary) : len(take_secondary) + shortfall]
    return [ranked[i][2] for i in sorted(set(take_primary + take_secondary))]
