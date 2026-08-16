"""Read/write helpers for the per-(user, dataset) preference subgraph.

Everything goes straight through the graph engine — never ``add_data_points``
or ``index_data_points`` — so preference data is never embedded and can never
come back as a search result. Only portable ``GraphDBInterface`` methods are
used (``get_neighborhood``, ``add_nodes``, ``update_node``, ``add_edges``,
``delete_edge_triples``), so the store works on any in-tree backend.

``load_preference_state`` returns the *stored* weights; applying decay is the
caller's job, so exactly one place knows the formula.
"""

from typing import Dict, Iterable, Mapping, Optional, Tuple
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import NodeSet
from cognee.shared.logging_utils import get_logger

from .constants import NEUTRAL_WEIGHT, PREFERENCE_NODE_SET, PREFERS_RELATIONSHIP
from .models import UserPreference

logger = get_logger("user_preferences.store")


def preference_node_id(user_id: str, dataset_id: str) -> UUID:
    """Deterministic id of the one preference node for (user, dataset).

    This is what enforces "at most one preference node per user per dataset":
    both the write side and the read side recompute the same id from the same
    two values, so exactly this helper builds it.
    """
    return generate_node_id(f"user_preference:{user_id}:{dataset_id}")


async def load_preference_state(
    user_id: str, dataset_id: str
) -> Tuple[Optional[dict], Dict[str, dict]]:
    """Load the preference node and its stored prefers weights in one round trip.

    Returns ``(node_properties, weights)`` where ``weights`` maps target node id
    to ``{"weight": float, "updated_at_turn": int}`` — the stored values, with
    no decay applied. Returns ``(None, {})`` when no preference node exists.
    """
    graph_engine = await get_graph_engine()

    pref_id = str(preference_node_id(user_id, dataset_id))
    nodes, edges = await graph_engine.get_neighborhood([pref_id], depth=1)

    node = next((props for nid, props in nodes if str(nid) == pref_id), None)
    if node is None:
        return None, {}

    stored = {
        str(target_id): {
            "weight": props.get("weight", NEUTRAL_WEIGHT),
            "updated_at_turn": props.get("updated_at_turn", 0),
        }
        # get_neighborhood returns every edge *between collected nodes*, not only
        # edges touching the seed — two preferred chunks linked to each other
        # land here too.
        for source_id, target_id, rel, props in edges or []
        if str(source_id) == pref_id and rel == PREFERS_RELATIONSHIP
    }
    return node, stored


async def upsert_preference_node(
    user_id: str,
    dataset_id: str,
    *,
    text: str = "",
    turn_counter: int = 0,
    text_watermark: str = "",
) -> str:
    """Create or update the preference node for (user, dataset); returns its id.

    Updates go through ``update_node`` — ``add_node`` is ``MERGE ... ON CREATE
    SET`` and would silently leave an existing row untouched. On create, the
    node is also merged into the ``user_preferences`` NodeSet so all preference
    nodes can be listed and deleted together.
    """
    graph_engine = await get_graph_engine()
    pref_id = str(preference_node_id(user_id, dataset_id))

    updated = await graph_engine.update_node(
        pref_id,
        {"text": text, "turn_counter": turn_counter, "text_watermark": text_watermark},
    )
    if updated:
        return pref_id

    preference = UserPreference(
        id=preference_node_id(user_id, dataset_id),
        user_id=str(user_id),
        dataset_id=str(dataset_id),
        text=text,
        turn_counter=turn_counter,
        text_watermark=text_watermark,
    )
    node_set = NodeSet(
        id=generate_node_id(f"NodeSet:{PREFERENCE_NODE_SET}"), name=PREFERENCE_NODE_SET
    )
    await graph_engine.add_nodes([preference, node_set])
    await graph_engine.add_edges(
        [
            (
                pref_id,
                str(node_set.id),
                "belongs_to_set",
                {
                    "relationship_name": "belongs_to_set",
                    "source_node_id": pref_id,
                    "target_node_id": str(node_set.id),
                },
            )
        ]
    )
    return pref_id


async def write_prefers_edges(
    user_id: str,
    dataset_id: str,
    weights_by_target: Mapping[str, float],
    updated_at_turn: int,
) -> None:
    """Upsert weighted ``prefers`` edges from the preference node to content nodes.

    ``add_edges`` is ``MERGE ... ON MATCH SET r.properties = edge.properties`` —
    it replaces the property blob rather than patching it, so ``weight`` and
    ``updated_at_turn`` are always written together. Writing only ``weight``
    would reset ``updated_at_turn`` to its default, which reads back as
    maximally idle and decays the edge to neutral on the next lookup.
    """
    if not weights_by_target:
        return

    graph_engine = await get_graph_engine()
    source_id = str(preference_node_id(user_id, dataset_id))

    edges = [
        (
            source_id,
            str(target_id),
            PREFERS_RELATIONSHIP,
            {
                "relationship_name": PREFERS_RELATIONSHIP,
                "source_node_id": source_id,
                "target_node_id": str(target_id),
                "weight": float(weight),
                "updated_at_turn": int(updated_at_turn),
            },
        )
        for target_id, weight in weights_by_target.items()
    ]
    await graph_engine.add_edges(edges)


async def delete_prefers_edges(user_id: str, dataset_id: str, target_ids: Iterable[str]) -> None:
    """Delete ``prefers`` edges to the given targets.

    ``delete_edge_triples`` issues ``DELETE r``, not ``DETACH DELETE``, so the
    preferred content nodes survive.
    """
    targets = [str(target_id) for target_id in target_ids]
    if not targets:
        return

    graph_engine = await get_graph_engine()
    source_id = str(preference_node_id(user_id, dataset_id))

    await graph_engine.delete_edge_triples(
        [
            EdgeIdentity(
                source_id=source_id,
                target_id=target_id,
                relationship_name=PREFERS_RELATIONSHIP,
            )
            for target_id in targets
        ]
    )
