"""Apply, rollback, and evaluate skill amendments."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.storage import add_data_points

from cognee.skills.utils import _make_change_event

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _replace_skill_md_body(file_content: str, new_body: str) -> str:
    """Replace the body of a SKILL.md file (below frontmatter) with new content."""
    pattern = re.compile(r"^(---\s*\n.*?\n---\s*\n)", re.DOTALL)
    match = pattern.match(file_content)
    if match:
        return match.group(1) + new_body
    return new_body


async def _load_amendment_from_graph(amendment_id: str, node_set: str) -> Optional[dict]:
    """Load a SkillAmendment node from graph by amendment_id."""
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    for nid, props in raw_nodes:
        if props.get("type") == "SkillAmendment" and props.get("amendment_id") == amendment_id:
            return {"nid": str(nid), **props}
    return None


async def _load_skill_from_graph(skill_id: str, node_set: str) -> Optional[tuple]:
    """Load a Skill node from graph, returning (nid, props)."""
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    for nid, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("skill_id") == skill_id:
            return (str(nid), props)
    return None


async def _update_skill_instructions(skill_nid: str, new_instructions: str, engine) -> str:
    """Update a skill node's instructions and content_hash in the graph."""
    new_hash = _content_hash(new_instructions)
    await engine.update_node(
        skill_nid,
        {"instructions": new_instructions, "content_hash": new_hash},
    )
    return new_hash


async def amendify(
    amendment_id: str,
    write_to_disk: bool = False,
    validate: bool = False,
    validation_task_text: str = "",
    node_set: str = "skills",
) -> dict:
    """Apply a proposed amendment to a skill.

    1. Load SkillAmendment from graph
    2. Load Skill from graph
    3. Optionally write amended instructions to SKILL.md on disk
    4. Update Skill node in graph with amended instructions
    5. Re-run enrichment pipeline on the updated skill
    6. Emit a SkillChangeEvent
    7. Update amendment status to "applied"
    8. Optionally validate by executing the skill

    Args:
        amendment_id: The amendment to apply.
        write_to_disk: If True, also update the SKILL.md file on disk.
        validate: If True, run execute_skill with validation_task_text after applying.
        validation_task_text: Task text for validation execution.
        node_set: Graph node set.

    Returns:
        Summary dict with applied amendment details and optional validation result.
    """
    amendment_node = await _load_amendment_from_graph(amendment_id, node_set)
    if amendment_node is None:
        return {"success": False, "error": f"Amendment '{amendment_id}' not found"}

    amendment_status = amendment_node.get("status", "")
    if amendment_status != "proposed":
        return {
            "success": False,
            "error": f"Amendment '{amendment_id}' has status '{amendment_status}', expected 'proposed'",
        }

    skill_id = amendment_node.get("skill_id", "")
    skill_result = await _load_skill_from_graph(skill_id, node_set)
    if skill_result is None:
        return {"success": False, "error": f"Skill '{skill_id}' not found"}

    skill_nid, skill_props = skill_result
    old_hash = skill_props.get("content_hash", "")
    amended_instructions = amendment_node.get("amended_instructions", "")
    skill_name = skill_props.get("name", skill_id)

    # Write to disk if requested
    if write_to_disk:
        source_path = skill_props.get("source_path", "")
        if source_path and Path(source_path).exists():
            file_content = Path(source_path).read_text(encoding="utf-8")
            new_content = _replace_skill_md_body(file_content, amended_instructions)
            Path(source_path).write_text(new_content, encoding="utf-8")
            logger.info("Wrote amended instructions to %s", source_path)

    # Update skill in graph
    engine = await get_graph_engine()
    new_hash = await _update_skill_instructions(skill_nid, amended_instructions, engine)

    # Re-enrich the updated skill
    try:
        from cognee.skills.tasks.enrich_skills import enrich_skills
        from cognee.skills.tasks.materialize_task_patterns import materialize_task_patterns
        from cognee.skills.models.skill import Skill

        # Build a minimal Skill object for enrichment
        skill_obj = Skill(
            id=skill_props.get("id", skill_nid),
            skill_id=skill_id,
            name=skill_name,
            description=skill_props.get("description", ""),
            instructions=amended_instructions,
            content_hash=new_hash,
        )

        enriched = await enrich_skills([skill_obj])
        if enriched:
            materialized = await materialize_task_patterns(enriched)
            await add_data_points(materialized)
            logger.info("Re-enriched skill '%s' after amendment", skill_name)
    except Exception as exc:
        logger.warning("Re-enrichment after amendment failed: %s", exc)

    # Emit change event
    event = _make_change_event(
        skill_id, skill_name, "amended", old_hash=old_hash, new_hash=new_hash
    )
    await add_data_points([event])

    # Update amendment status
    applied_at_ms = int(time.time() * 1000)
    amendment_nid = amendment_node.get("nid", "")
    if amendment_nid:
        await engine.update_node(
            amendment_nid, {"status": "applied", "applied_at_ms": applied_at_ms}
        )

    result = {
        "success": True,
        "amendment_id": amendment_id,
        "skill_id": skill_id,
        "skill_name": skill_name,
        "status": "applied",
        "old_hash": old_hash,
        "new_hash": new_hash,
    }

    # Optional validation
    if validate and validation_task_text:
        try:
            from cognee.skills.execute import execute_skill

            skill_dict = {
                "skill_id": skill_id,
                "name": skill_name,
                "instructions": amended_instructions,
                "instruction_summary": skill_props.get("instruction_summary", ""),
                "description": skill_props.get("description", ""),
                "tags": skill_props.get("tags", []),
                "complexity": skill_props.get("complexity", ""),
                "source_path": skill_props.get("source_path", ""),
                "task_patterns": [],
            }
            validation_result = await execute_skill(
                skill=skill_dict, task_text=validation_task_text
            )
            result["validation"] = validation_result
        except Exception as exc:
            result["validation"] = {"success": False, "error": str(exc)}

    return result


async def rollback_amendify(
    amendment_id: str,
    write_to_disk: bool = False,
    node_set: str = "skills",
) -> bool:
    """Rollback an applied amendment, restoring original instructions.

    Args:
        amendment_id: The amendment to rollback.
        write_to_disk: If True, also restore the original SKILL.md on disk.
        node_set: Graph node set.

    Returns:
        True if rollback succeeded, False otherwise.
    """
    amendment_node = await _load_amendment_from_graph(amendment_id, node_set)
    if amendment_node is None:
        logger.warning("Amendment '%s' not found", amendment_id)
        return False

    if amendment_node.get("status") != "applied":
        logger.warning(
            "Amendment '%s' status is '%s', not 'applied'",
            amendment_id,
            amendment_node.get("status"),
        )
        return False

    skill_id = amendment_node.get("skill_id", "")
    skill_result = await _load_skill_from_graph(skill_id, node_set)
    if skill_result is None:
        logger.warning("Skill '%s' not found for rollback", skill_id)
        return False

    skill_nid, skill_props = skill_result
    old_hash = skill_props.get("content_hash", "")
    original_instructions = amendment_node.get("original_instructions", "")
    skill_name = skill_props.get("name", skill_id)

    # Restore on disk if requested
    if write_to_disk:
        source_path = skill_props.get("source_path", "")
        if source_path and Path(source_path).exists():
            file_content = Path(source_path).read_text(encoding="utf-8")
            new_content = _replace_skill_md_body(file_content, original_instructions)
            Path(source_path).write_text(new_content, encoding="utf-8")
            logger.info("Restored original instructions to %s", source_path)

    # Restore original instructions
    engine = await get_graph_engine()
    new_hash = await _update_skill_instructions(skill_nid, original_instructions, engine)

    # Re-enrich
    try:
        from cognee.skills.tasks.enrich_skills import enrich_skills
        from cognee.skills.tasks.materialize_task_patterns import materialize_task_patterns
        from cognee.skills.models.skill import Skill

        skill_obj = Skill(
            id=skill_props.get("id", skill_nid),
            skill_id=skill_id,
            name=skill_name,
            description=skill_props.get("description", ""),
            instructions=original_instructions,
            content_hash=new_hash,
        )
        enriched = await enrich_skills([skill_obj])
        if enriched:
            materialized = await materialize_task_patterns(enriched)
            await add_data_points(materialized)
    except Exception as exc:
        logger.warning("Re-enrichment after rollback failed: %s", exc)

    # Emit change event
    event = _make_change_event(
        skill_id, skill_name, "rolled_back", old_hash=old_hash, new_hash=new_hash
    )
    await add_data_points([event])

    # Update amendment status
    amendment_nid = amendment_node.get("nid", "")
    if amendment_nid:
        await engine.update_node(amendment_nid, {"status": "rolled_back"})

    logger.info("Rolled back amendment '%s' for skill '%s'", amendment_id, skill_name)
    return True


async def evaluate_amendify(
    amendment_id: str,
    node_set: str = "skills",
) -> dict:
    """Evaluate an amendment by comparing pre- and post-amendment success scores.

    Args:
        amendment_id: The amendment to evaluate.
        node_set: Graph node set.

    Returns:
        Dict with pre_avg, post_avg, improvement, run_count, recommendation.
    """
    amendment_node = await _load_amendment_from_graph(amendment_id, node_set)
    if amendment_node is None:
        return {"error": f"Amendment '{amendment_id}' not found"}

    skill_id = amendment_node.get("skill_id", "")
    pre_avg = float(amendment_node.get("pre_amendment_avg_score", 0.0))
    applied_at_ms = int(amendment_node.get("applied_at_ms", 0))

    # Load all SkillRun nodes for this skill
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])

    # Find post-amendment runs (created after the amendment was applied)
    post_scores = []
    for _, props in raw_nodes:
        if props.get("type") == "SkillRun" and props.get("selected_skill_id") == skill_id:
            started = int(props.get("started_at_ms", 0))
            if applied_at_ms == 0 or started > applied_at_ms:
                score = float(props.get("success_score", 0.0))
                post_scores.append(score)

    post_avg = sum(post_scores) / len(post_scores) if post_scores else 0.0
    improvement = post_avg - pre_avg
    recommendation = "keep" if improvement >= 0 else "rollback"

    # Update the amendment node with post-amendment stats
    amendment_nid = amendment_node.get("nid", "")
    if amendment_nid:
        await engine.update_node(
            amendment_nid,
            {
                "post_amendment_avg_score": post_avg,
                "post_amendment_run_count": len(post_scores),
            },
        )

    return {
        "amendment_id": amendment_id,
        "skill_id": skill_id,
        "pre_avg": pre_avg,
        "post_avg": post_avg,
        "improvement": improvement,
        "run_count": len(post_scores),
        "recommendation": recommendation,
    }
