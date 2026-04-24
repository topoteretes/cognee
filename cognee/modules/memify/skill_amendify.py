"""Apply, rollback, and evaluate skill amendments."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.tasks.storage import add_data_points

from cognee.modules.tools.skill_change_events import _make_change_event

logger = logging.getLogger(__name__)


def _tag_with_nodeset(items, node_set: str = "skills"):
    """Tag DataPoints with belongs_to_set so they appear in nodeset subgraph queries."""
    ns = NodeSet(id=generate_node_id(f"NodeSet:{node_set}"), name=node_set)
    for item in items:
        if hasattr(item, "belongs_to_set"):
            item.belongs_to_set = [ns]
    return items


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
    """Load a Skill node from graph, returning (nid, props).

    The ``skill_id`` argument is a pointer to the Skill's canonical
    ``name`` property.
    """
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    for nid, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("name") == skill_id:
            return (str(nid), props)
    return None


def _reconstruct_amendment(node_dict: dict):
    """Reconstruct a SkillAmendment DataPoint from a raw graph node dict."""
    from cognee.modules.engine.models.SkillAmendment import SkillAmendment

    node_id = node_dict.get("id", node_dict.get("nid", ""))
    return SkillAmendment(
        id=uuid.UUID(str(node_id)) if node_id else uuid.uuid4(),
        amendment_id=node_dict.get("amendment_id", ""),
        skill_id=node_dict.get("skill_id", ""),
        skill_name=node_dict.get("skill_name", ""),
        inspection_id=node_dict.get("inspection_id", ""),
        original_instructions=node_dict.get("original_instructions", ""),
        amended_instructions=node_dict.get("amended_instructions", ""),
        change_explanation=node_dict.get("change_explanation", ""),
        expected_improvement=node_dict.get("expected_improvement", ""),
        status=node_dict.get("status", "proposed"),
        amendment_model=node_dict.get("amendment_model", ""),
        amendment_confidence=float(node_dict.get("amendment_confidence", 0.0)),
        pre_amendment_avg_score=float(node_dict.get("pre_amendment_avg_score", 0.0)),
        applied_at_ms=int(node_dict.get("applied_at_ms", 0)),
        post_amendment_avg_score=float(node_dict.get("post_amendment_avg_score", 0.0)),
        post_amendment_run_count=int(node_dict.get("post_amendment_run_count", 0)),
    )


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

    # Update skill in graph + re-enrich
    new_hash = _content_hash(amended_instructions)
    try:
        from cognee.modules.tools.skill_enrichment_tasks import enrich_skills
        from cognee.modules.tools.skill_pattern_tasks import materialize_task_patterns
        from cognee.modules.engine.models.Skill import Skill

        skill_obj = Skill(
            id=uuid.UUID(str(skill_props.get("id", skill_nid))),
            name=skill_name,
            description=skill_props.get("description", ""),
            procedure=amended_instructions,
            content_hash=new_hash,
        )

        enriched = await enrich_skills([skill_obj])
        if enriched:
            materialized = await materialize_task_patterns(enriched)
            _tag_with_nodeset(materialized, node_set)
            await add_data_points(materialized)
            logger.info("Re-enriched skill '%s' after amendment", skill_name)
        else:
            _tag_with_nodeset([skill_obj], node_set)
            await add_data_points([skill_obj])
    except Exception as exc:
        logger.warning("Re-enrichment after amendment failed, persisting basic update: %s", exc)
        from cognee.modules.engine.models.Skill import Skill

        skill_obj = Skill(
            id=uuid.UUID(str(skill_props.get("id", skill_nid))),
            name=skill_name,
            description=skill_props.get("description", ""),
            procedure=amended_instructions,
            content_hash=new_hash,
        )
        _tag_with_nodeset([skill_obj], node_set)
        await add_data_points([skill_obj])

    # Emit change event
    event = _make_change_event(
        skill_id, skill_name, "amended", old_hash=old_hash, new_hash=new_hash
    )
    _tag_with_nodeset([event], node_set)
    await add_data_points([event])

    # Update amendment status via DataPoint upsert
    applied_at_ms = int(time.time() * 1000)
    amendment_node["status"] = "applied"
    amendment_node["applied_at_ms"] = applied_at_ms
    amendment_dp = _reconstruct_amendment(amendment_node)
    _tag_with_nodeset([amendment_dp], node_set)
    await add_data_points([amendment_dp])

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
            from cognee.cognee_skills.execute import execute_skill

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

    # Restore original instructions + re-enrich
    new_hash = _content_hash(original_instructions)
    try:
        from cognee.modules.tools.skill_enrichment_tasks import enrich_skills
        from cognee.modules.tools.skill_pattern_tasks import materialize_task_patterns
        from cognee.modules.engine.models.Skill import Skill

        skill_obj = Skill(
            id=uuid.UUID(str(skill_props.get("id", skill_nid))),
            name=skill_name,
            description=skill_props.get("description", ""),
            procedure=original_instructions,
            content_hash=new_hash,
        )
        enriched = await enrich_skills([skill_obj])
        if enriched:
            materialized = await materialize_task_patterns(enriched)
            _tag_with_nodeset(materialized, node_set)
            await add_data_points(materialized)
        else:
            _tag_with_nodeset([skill_obj], node_set)
            await add_data_points([skill_obj])
    except Exception as exc:
        logger.warning("Re-enrichment after rollback failed, persisting basic update: %s", exc)
        from cognee.modules.engine.models.Skill import Skill

        skill_obj = Skill(
            id=uuid.UUID(str(skill_props.get("id", skill_nid))),
            name=skill_name,
            description=skill_props.get("description", ""),
            procedure=original_instructions,
            content_hash=new_hash,
        )
        _tag_with_nodeset([skill_obj], node_set)
        await add_data_points([skill_obj])

    # Emit change event
    event = _make_change_event(
        skill_id, skill_name, "rolled_back", old_hash=old_hash, new_hash=new_hash
    )
    _tag_with_nodeset([event], node_set)
    await add_data_points([event])

    # Update amendment status via DataPoint upsert
    amendment_node["status"] = "rolled_back"
    amendment_dp = _reconstruct_amendment(amendment_node)
    _tag_with_nodeset([amendment_dp], node_set)
    await add_data_points([amendment_dp])

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

    # Update the amendment node with post-amendment stats via DataPoint upsert
    amendment_node["post_amendment_avg_score"] = post_avg
    amendment_node["post_amendment_run_count"] = len(post_scores)
    amendment_dp = _reconstruct_amendment(amendment_node)
    _tag_with_nodeset([amendment_dp], node_set)
    await add_data_points([amendment_dp])

    return {
        "amendment_id": amendment_id,
        "skill_id": skill_id,
        "pre_avg": pre_avg,
        "post_avg": post_avg,
        "improvement": improvement,
        "run_count": len(post_scores),
        "recommendation": recommendation,
    }
