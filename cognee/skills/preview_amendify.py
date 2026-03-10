"""Propose an amendment to a skill's instructions based on an inspection."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid5, UUID

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm import get_llm_config
from cognee.tasks.storage import add_data_points

from cognee.skills.models.skill_inspection import SkillInspection
from cognee.skills.models.skill_amendment import AmendmentProposal, SkillAmendment

logger = logging.getLogger(__name__)

AMENDMENT_NAMESPACE = UUID("f6a7b8c9-d0e1-2345-f012-3456789abcde")

SYSTEM_PROMPT = """\
You are an expert at improving agentic skill instructions. Given an inspection of what's \
wrong and the current instructions, produce improved instructions that fix the issue. \
Preserve overall structure and style. Only change what needs to change. Output the \
COMPLETE amended instructions (not a diff)."""

USER_PROMPT_TEMPLATE = """\
Skill name: {skill_name}

Inspection:
- Failure category: {failure_category}
- Root cause: {root_cause}
- Severity: {severity}
- Improvement hypothesis: {improvement_hypothesis}
- Analyzed {run_count} failed runs (avg success score: {avg_score:.2f})

Current instructions:
---
{instructions}
---

Produce:
- amended_instructions: The COMPLETE improved instructions
- change_explanation: What was changed and why
- expected_improvement: What improvement is expected
- confidence: Your confidence from 0.0 to 1.0"""


async def preview_skill_amendify(
    inspection: SkillInspection,
    skill: dict,
) -> Optional[SkillAmendment]:
    """Generate a proposed amendment for a skill based on its inspection.

    Args:
        inspection: The SkillInspection describing what's wrong.
        skill: The skill dict (from client.load()) with current instructions.

    Returns:
        A persisted SkillAmendment DataPoint with status="proposed".
    """
    instructions = skill.get("instructions", "")
    skill_name = skill.get("name", inspection.skill_name)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        failure_category=inspection.failure_category,
        root_cause=inspection.root_cause,
        severity=inspection.severity,
        improvement_hypothesis=inspection.improvement_hypothesis,
        run_count=inspection.analyzed_run_count,
        avg_score=inspection.avg_success_score,
        instructions=instructions[:8000],
    )

    try:
        result: AmendmentProposal = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=AmendmentProposal,
        )
    except Exception as exc:
        logger.warning("LLM amendment generation failed for skill '%s': %s", skill_name, exc)
        return None

    llm_config = get_llm_config()
    amendment_id = str(
        uuid5(AMENDMENT_NAMESPACE, f"{inspection.inspection_id}:{inspection.skill_id}")
    )

    amendment = SkillAmendment(
        id=uuid5(AMENDMENT_NAMESPACE, amendment_id),
        name=f"amendment: {skill_name}",
        description=f"Proposed amendment for skill '{skill_name}': {result.change_explanation[:200]}",
        amendment_id=amendment_id,
        skill_id=inspection.skill_id,
        skill_name=skill_name,
        inspection_id=inspection.inspection_id,
        original_instructions=instructions,
        amended_instructions=result.amended_instructions,
        change_explanation=result.change_explanation,
        expected_improvement=result.expected_improvement,
        status="proposed",
        amendment_model=llm_config.llm_model or "unknown",
        amendment_confidence=result.confidence,
        pre_amendment_avg_score=inspection.avg_success_score,
    )

    await add_data_points([amendment])
    logger.info(
        "Proposed amendment for skill '%s' (confidence=%.2f): %s",
        skill_name,
        result.confidence,
        result.change_explanation[:100],
    )

    return amendment
