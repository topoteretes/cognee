"""Inspect why a skill fails by analyzing failed SkillRun records."""

from __future__ import annotations

import logging
import time
from typing import Optional
from uuid import uuid5, UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm import get_llm_config
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.storage import add_data_points

from cognee.skills.models.skill_inspection import InspectionResult, SkillInspection

logger = logging.getLogger(__name__)

INSPECTION_NAMESPACE = UUID("e5f6a7b8-c9d0-1234-ef01-23456789abcd")

SYSTEM_PROMPT = """\
You are an expert at diagnosing why agentic skills fail. Given a skill's instructions \
and one or more failed execution records, identify the root cause and propose a \
hypothesis for improvement. Be precise and actionable."""

USER_PROMPT_TEMPLATE = """\
Skill name: {skill_name}

Skill instructions:
---
{instructions}
---

Failed execution records ({run_count} runs, avg success score: {avg_score:.2f}):

{formatted_runs}

Analyze these failures and determine:
- failure_category: one of "instruction_gap", "ambiguity", "wrong_scope", \
"tooling", "context_missing", "other"
- root_cause: concise description of the root cause
- severity: "low", "medium", "high", or "critical"
- improvement_hypothesis: actionable hypothesis for improving the skill
- confidence: your confidence from 0.0 to 1.0"""


def _format_run(run_props: dict, index: int) -> str:
    """Format a single failed run for the LLM prompt."""
    parts = [f"Run {index + 1}:"]
    if run_props.get("task_text"):
        parts.append(f"  Task: {run_props['task_text'][:500]}")
    if run_props.get("error_type"):
        parts.append(f"  Error type: {run_props['error_type']}")
    if run_props.get("error_message"):
        parts.append(f"  Error message: {run_props['error_message'][:500]}")
    if run_props.get("result_summary"):
        parts.append(f"  Result summary: {run_props['result_summary'][:500]}")
    score = run_props.get("success_score", 0.0)
    parts.append(f"  Success score: {score}")
    if run_props.get("tool_trace"):
        trace = str(run_props["tool_trace"])[:1000]
        parts.append(f"  Tool trace (truncated): {trace}")
    return "\n".join(parts)


async def inspect_skill(
    skill_id: str,
    min_runs: int = 1,
    score_threshold: float = 0.5,
    node_set: str = "skills",
) -> Optional[SkillInspection]:
    """Analyze failed SkillRuns for a skill and produce an inspection.

    Args:
        skill_id: The skill to inspect.
        min_runs: Minimum number of failed runs required before inspecting.
        score_threshold: Runs with success_score below this are considered failures.
        node_set: Graph node set to search.

    Returns:
        A persisted SkillInspection DataPoint, or None if insufficient failures.
    """
    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])

    # Find the skill
    skill_node = None
    for _, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("skill_id") == skill_id:
            skill_node = props
            break

    if skill_node is None:
        logger.warning("Skill '%s' not found in graph", skill_id)
        return None

    # Find failed runs
    failed_runs = []
    for _, props in raw_nodes:
        if (
            props.get("type") == "SkillRun"
            and props.get("selected_skill_id") == skill_id
            and float(props.get("success_score", 1.0)) < score_threshold
        ):
            failed_runs.append(props)

    if len(failed_runs) < min_runs:
        logger.info(
            "Skill '%s' has %d failed runs (need %d), skipping inspection",
            skill_id,
            len(failed_runs),
            min_runs,
        )
        return None

    # Compute stats
    scores = [float(r.get("success_score", 0.0)) for r in failed_runs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    run_ids = [r.get("run_id", str(i)) for i, r in enumerate(failed_runs)]

    # Format runs for prompt (limit to 10 most recent)
    runs_to_show = failed_runs[:10]
    formatted_runs = "\n\n".join(_format_run(r, i) for i, r in enumerate(runs_to_show))

    instructions = skill_node.get("instructions", "")[:8000]
    skill_name = skill_node.get("name", skill_id)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        instructions=instructions,
        run_count=len(failed_runs),
        avg_score=avg_score,
        formatted_runs=formatted_runs,
    )

    try:
        result: InspectionResult = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=InspectionResult,
        )
    except Exception as exc:
        logger.warning("LLM inspection failed for skill '%s': %s", skill_id, exc)
        return None

    llm_config = get_llm_config()
    inspection_id = str(
        uuid5(INSPECTION_NAMESPACE, f"{skill_id}:{len(failed_runs)}:{avg_score}:{time.time()}")
    )

    inspection = SkillInspection(
        id=uuid5(INSPECTION_NAMESPACE, inspection_id),
        name=f"inspection: {skill_name}",
        description=f"Inspection for skill '{skill_name}': {result.root_cause[:200]}",
        inspection_id=inspection_id,
        skill_id=skill_id,
        skill_name=skill_name,
        failure_category=result.failure_category,
        root_cause=result.root_cause,
        severity=result.severity,
        improvement_hypothesis=result.improvement_hypothesis,
        analyzed_run_ids=run_ids,
        analyzed_run_count=len(failed_runs),
        avg_success_score=avg_score,
        inspection_model=llm_config.llm_model or "unknown",
        inspection_confidence=result.confidence,
    )

    await add_data_points([inspection])
    logger.info(
        "Inspected skill '%s': category=%s, severity=%s, confidence=%.2f",
        skill_name,
        result.failure_category,
        result.severity,
        result.confidence,
    )

    return inspection
