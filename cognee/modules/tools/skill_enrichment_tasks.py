"""LLM enrichment task: fills derived Skill fields from raw content."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm import get_llm_config

from cognee.modules.engine.models.Skill import Skill

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert at analyzing agentic skill documentation.
Given the raw content of a skill document, extract structured metadata.
Be concise and precise. Only include information that is clearly supported by the text.
If a field cannot be determined, use a sensible default rather than guessing."""

USER_PROMPT_TEMPLATE = """\
Skill name: {name}
Skill description (may be empty): {description}

Full skill document:
---
{instructions}
---

Analyze this skill and fill in the following fields:
- description: A concise, normalized 1-2 sentence description of what this skill is. \
Improve on the raw description if present, or generate one if empty.
- instruction_summary: A concise 2-3 sentence summary of what this skill does and when to use it.
- triggers: A list of short phrases describing when this skill should be activated \
(e.g. "user asks to summarize a document", "code review requested").
- tags: A list of category tags from this vocabulary: \
context-management, evaluation, code, memory, multi-agent, tool-design, data-processing, \
web, filesystem, planning, debugging, testing, documentation, security, performance, other. \
Pick 1-5 that apply.
- complexity: Classify as "simple" (single-step, no orchestration), \
"workflow" (multi-step with defined sequence), or "agent" (requires autonomous decision-making).
- task_pattern_candidates: 3-8 short slug-style patterns this skill can solve \
(e.g. "llm_evaluation", "context_compression", "tool_api_design"). Use snake_case.
- confidence: Your confidence in the extraction from 0.0 to 1.0."""


class SkillEnrichment(BaseModel):
    """LLM response model for skill enrichment — only the derived fields."""

    description: str = Field(
        description="Concise, normalized 1-2 sentence description of the skill"
    )
    instruction_summary: str = Field(description="2-3 sentence summary of the skill")
    triggers: List[str] = Field(default_factory=list, description="When to activate this skill")
    tags: List[str] = Field(default_factory=list, description="Category tags")
    complexity: Literal["simple", "workflow", "agent"] = Field(default="simple")
    task_pattern_candidates: List[str] = Field(
        default_factory=list, description="Canonical task patterns this skill solves"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


async def _enrich_one(skill: Skill, llm_model_name: str) -> Skill:
    """Call the LLM to enrich a single Skill's derived fields."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=skill.name,
        description=skill.description,
        instructions=skill.procedure[:8000],
    )

    try:
        result: SkillEnrichment = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=SkillEnrichment,
        )

        skill.description = result.description
        skill.instruction_summary = result.instruction_summary
        skill.triggers = result.triggers
        skill.tags = result.tags
        skill.complexity = result.complexity
        skill.task_pattern_candidates = result.task_pattern_candidates
        skill.enrichment_model = llm_model_name
        skill.enrichment_confidence = result.confidence

        logger.info("Enriched skill '%s' (confidence=%.2f)", skill.name, result.confidence)

    except Exception as exc:
        logger.warning("Failed to enrich skill '%s': %s", skill.name, exc)

    return skill


async def enrich_skills(
    skills: List[Skill],
    context: Optional[Dict[str, Any]] = None,
) -> List[Skill]:
    """Enrich a batch of Skills using LLM structured extraction.

    Fills: description, instruction_summary, triggers, tags, complexity,
           task_pattern_candidates, enrichment_model, enrichment_confidence.
    Raw parser values preserved in: description_raw, triggers_raw, tags_raw.
    Never modifies: skill_id, instructions, source_path, source_repo,
                    content_hash, resources, tools.
    """
    llm_config = get_llm_config()
    llm_model_name = llm_config.llm_model or "unknown"

    enriched = await asyncio.gather(*[_enrich_one(s, llm_model_name) for s in skills])

    logger.info(
        "Enriched %d/%d skills", len([s for s in enriched if s.instruction_summary]), len(skills)
    )
    return list(enriched)
