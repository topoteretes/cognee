"""Pipeline helpers that survived the ingest consolidation.

Ingestion (parse + optional enrichment + materialize patterns) moved to
``cognee.modules.tools.ingest_skills.add_skills`` and is reachable as
``cognee.remember(path, enrich=...)``. This module now only holds
``remove_skill`` — the one graph-mutation primitive the skills client
still calls directly.
"""

from __future__ import annotations

import logging

from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models.node_set import NodeSet

from cognee.cognee_skills.utils import _make_change_event

logger = logging.getLogger(__name__)


async def remove_skill(skill_id: str) -> bool:
    """Remove a single skill by name from graph and vector stores.

    Also emits a SkillChangeEvent for temporal tracking.

    Returns True if the skill was found and deleted, False otherwise.
    """
    await setup()

    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=["skills"])

    skill_nid = None
    skill_props = None
    for nid, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("name") == skill_id:
            skill_nid = str(nid)
            skill_props = props
            break

    if skill_nid is None:
        logger.info("Skill '%s' not found in graph", skill_id)
        return False

    await engine.delete_nodes([skill_nid])

    vector_engine = get_vector_engine()
    for field in ["name", "instruction_summary", "description"]:
        collection = f"Skill_{field}"
        try:
            await vector_engine.delete_data_points(collection, [skill_nid])
        except Exception:
            pass

    event = _make_change_event(
        skill_id,
        skill_props.get("name", skill_id),
        "removed",
        old_hash=skill_props.get("content_hash", ""),
    )
    await add_data_points([event])

    logger.info("Removed skill '%s' (node %s)", skill_id, skill_nid)
    return True
