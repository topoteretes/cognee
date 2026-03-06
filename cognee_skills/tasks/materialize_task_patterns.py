"""Materialize TaskPattern nodes from Skill.task_pattern_candidates + LLM enrich."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid5

from pydantic import BaseModel, Field

from cognee.infrastructure.engine import Edge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm import get_llm_config

from cognee_skills.models.skill import Skill
from cognee_skills.models.task_pattern import TaskPattern

logger = logging.getLogger(__name__)

NAMESPACE = UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")

CATEGORIES = [
    "context-management",
    "evaluation",
    "code",
    "memory",
    "multi-agent",
    "tool-design",
    "data-processing",
    "web",
    "filesystem",
    "planning",
    "debugging",
    "testing",
    "documentation",
    "security",
    "performance",
    "other",
]

SYSTEM_PROMPT = """\
You are an expert at categorizing agentic task patterns.
Given a task pattern slug and the names of skills that solve it,
produce a short human-readable description and a category."""

USER_PROMPT_TEMPLATE = """\
Task pattern slug: {slug}
Skills that solve this pattern: {skill_names}

Fill in:
- text: A concise 1-2 sentence description of what this task pattern represents.
- category: Pick exactly one from: {categories}"""


class TaskPatternEnrichment(BaseModel):
    """LLM response model for task pattern enrichment."""

    text: str = Field(description="1-2 sentence description of the task pattern")
    category: str = Field(default="other", description="Category from controlled vocabulary")


def _make_pattern_id(slug: str) -> UUID:
    return uuid5(NAMESPACE, slug)


async def _enrich_one_pattern(
    slug: str,
    skill_names: List[str],
    llm_model: str,
) -> TaskPatternEnrichment:
    """LLM call to produce text + category for one TaskPattern."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        slug=slug,
        skill_names=", ".join(skill_names),
        categories=", ".join(CATEGORIES),
    )
    try:
        return await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=TaskPatternEnrichment,
        )
    except Exception as exc:
        logger.warning("Failed to enrich pattern '%s': %s", slug, exc)
        return TaskPatternEnrichment(
            text=slug.replace("_", " "),
            category="other",
        )


async def materialize_task_patterns(
    skills: List[Skill],
    context: Optional[Dict[str, Any]] = None,
) -> List[Skill]:
    """Create TaskPattern nodes from candidates and wire solves edges.

    1. Collect all task_pattern_candidates across skills, deduplicate.
    2. LLM-enrich each unique pattern (text + category).
    3. Create TaskPattern DataPoints with deterministic IDs.
    4. Populate Skill.solves with (Edge, TaskPattern) tuples.
    """
    slug_to_skill_ids: Dict[str, List[str]] = {}
    slug_to_skill_names: Dict[str, List[str]] = {}
    slug_to_examples: Dict[str, List[str]] = {}

    for skill in skills:
        for slug in skill.task_pattern_candidates:
            slug_to_skill_ids.setdefault(slug, []).append(skill.skill_id)
            slug_to_skill_names.setdefault(slug, []).append(skill.name)
            for trigger in skill.triggers[:3]:
                slug_to_examples.setdefault(slug, []).append(trigger)

    if not slug_to_skill_ids:
        logger.info("No task_pattern_candidates found; skipping materialization.")
        return skills

    llm_config = get_llm_config()
    llm_model = llm_config.llm_model or "unknown"

    enrichments = await asyncio.gather(
        *[
            _enrich_one_pattern(slug, names, llm_model)
            for slug, names in slug_to_skill_names.items()
        ]
    )

    patterns: Dict[str, TaskPattern] = {}
    for (slug, _), enrichment in zip(slug_to_skill_ids.items(), enrichments):
        examples = list(dict.fromkeys(slug_to_examples.get(slug, [])))[:10]
        patterns[slug] = TaskPattern(
            id=_make_pattern_id(slug),
            pattern_id=slug,
            name=slug,
            pattern_key=slug,
            text=enrichment.text,
            category=enrichment.category,
            source_skill_ids=slug_to_skill_ids[slug],
            examples=examples,
            enrichment_model=llm_model,
            enrichment_confidence=0.9,
        )

    for skill in skills:
        skill.solves = [
            (Edge(relationship_type="solves"), patterns[slug])
            for slug in skill.task_pattern_candidates
            if slug in patterns
        ]

    logger.info(
        "Materialized %d TaskPatterns from %d skills",
        len(patterns),
        len(skills),
    )
    return skills
