"""Memify task: find underperforming skills and improve them in one pass.

``improve_failing_skills`` is the one public function here. It's a
memify-compatible task: call it directly via ``cognee.memify`` or via
``cognee.remember(path, improve=True)``.

Flow per invocation:

  1. Query the graph for every Skill in the ``"skills"`` node_set.
  2. For each, call ``inspect_skill`` — skips Skills without enough
     low-scored ``SkillRun`` records (default: 3 runs below score 0.5).
  3. Inspections that return produce a fresh amendment proposal via the
     LLM.
  4. The amendment is applied in the graph immediately — no preview,
     no rollback. If you need those, call ``skill_preview_amendify``
     + ``skill_amendify`` directly (they still exist as lower-level
     helpers).
  5. A ``SkillChangeEvent`` is emitted for every applied amendment.

Returns the list of applied ``SkillAmendment`` nodes for inspection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet

from cognee.modules.memify.skill_inspect import inspect_skill
from cognee.modules.memify.skill_preview_amendify import preview_skill_amendify
from cognee.modules.memify.skill_amendify import amendify as _amendify_apply

logger = logging.getLogger(__name__)


async def _list_skill_names(node_set: str) -> List[str]:
    """Return the canonical ``name`` of every Skill in the given node_set."""
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    return [
        props["name"]
        for _, props in raw_nodes
        if props.get("type") == "Skill" and props.get("name")
    ]


async def _load_skill_dict(skill_name: str, node_set: str) -> Optional[Dict[str, Any]]:
    """Load a Skill node's full property dict, matching what preview_amendify expects."""
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    for _, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("name") == skill_name:
            return {
                "skill_id": props.get("name", ""),
                "name": props.get("name", ""),
                "instructions": props.get("procedure", ""),
                "instruction_summary": props.get("instruction_summary", ""),
                "description": props.get("description", ""),
                "source_path": props.get("source_path", ""),
            }
    return None


async def improve_failing_skills(
    data: Any = None,
    context: Optional[Dict[str, Any]] = None,
    *,
    node_set: str = "skills",
    min_runs: int = 3,
    score_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Find skills whose recent ``SkillRun`` records score poorly and fix them.

    Args:
        data: Ignored. Present so this function can be used as a memify
            task (memify tasks accept ``data`` as their first positional
            argument).
        context: Ignored. Same reason.
        node_set: Graph node_set to scope the skill listing.
        min_runs: Minimum number of sub-threshold ``SkillRun`` records
            required before a skill is considered for improvement.
        score_threshold: Runs with ``success_score`` below this count as
            failures.

    Returns:
        A list of dicts describing each applied amendment (empty list
        if no skills needed improvement). Each dict has the shape
        returned by :func:`cognee.modules.memify.skill_amendify.amendify`.
    """
    skill_names = await _list_skill_names(node_set)
    if not skill_names:
        logger.info("improve_failing_skills: no skills in node_set '%s'", node_set)
        return []

    applied: List[Dict[str, Any]] = []
    for skill_name in skill_names:
        inspection = await inspect_skill(
            skill_name,
            min_runs=min_runs,
            score_threshold=score_threshold,
            node_set=node_set,
        )
        if inspection is None:
            continue  # not enough failure signal, skip

        skill_dict = await _load_skill_dict(skill_name, node_set)
        if skill_dict is None:
            logger.warning("improve_failing_skills: skill '%s' disappeared mid-pass", skill_name)
            continue

        amendment = await preview_skill_amendify(inspection, skill_dict)
        if amendment is None:
            continue

        result = await _amendify_apply(amendment.amendment_id, node_set=node_set)
        if result.get("success"):
            applied.append(result)
            logger.info(
                "improve_failing_skills: amended '%s' (amendment_id=%s)",
                skill_name,
                amendment.amendment_id,
            )

    logger.info("improve_failing_skills: applied %d amendment(s)", len(applied))
    return applied
