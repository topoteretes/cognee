"""Skill generation and update logic."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from pydantic import BaseModel, Field

from examples.demos.job_finding_agent.agent.agent_models import SkillProfileOutput


def build_skill_markdown(profile: SkillProfileOutput) -> str:
    """Build a Claude-style operational skill document."""
    strengths = "\n".join(f"- {item}" for item in profile.core_strengths) or "- (none)"
    heuristics = "\n".join(f"- {item}" for item in profile.heuristics) or "- (none)"
    return (
        "---\n"
        "name: Job Evaluation Skill\n"
        "description: Evaluate job descriptions against the applicant profile, return APPLY or DONT_APPLY, and refine heuristics from feedback in the job-finding demo workflow.\n"
        "---\n\n"
        "# Job Evaluation Skill\n\n"
        "## Purpose\n"
        "Evaluate incoming jobs against the candidate profile and return APPLY/DONT_APPLY.\n\n"
        "## Profile Summary\n"
        f"{profile.profile_summary}\n\n"
        "## Core Strengths\n"
        f"{strengths}\n\n"
        "## Decision Heuristics\n"
        f"{heuristics}\n\n"
        "## Decision Output Contract\n"
        "- Decision: APPLY or DONT_APPLY\n"
        "- Rationale: concise and evidence-based\n"
    )


async def generate_skill_from_cv(cv_text: str) -> str:
    """Create initial skill markdown from CV text."""
    profile = await LLMGateway.acreate_structured_output(
        cv_text,
        (
            "Extract a concise candidate profile from this CV. "
            "Return practical strengths and hiring heuristics for choosing jobs."
        ),
        SkillProfileOutput,
    )
    return build_skill_markdown(profile)


async def reset_skill_file_from_cv(cv_text: str, skill_path: Path) -> str:
    """Reset skill file to the initial non-updated content derived from CV."""
    initial_skill = await generate_skill_from_cv(cv_text)
    skill_path.write_text(initial_skill, encoding="utf-8")
    return initial_skill


def _strip_feedback_prefix(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    if lower.startswith("feedback learned:"):
        return cleaned.split(":", 1)[1].strip()
    return cleaned


class SkillPreferenceUpdates(BaseModel):
    """Structured LLM output for user-specific preference updates."""

    preferences: list[str] = Field(default_factory=list)


def _split_feedback_section(skill_text: str) -> tuple[str, list[str]]:
    heading = "## Feedback Updates"
    if heading not in skill_text:
        return skill_text.rstrip(), []

    before, after = skill_text.split(heading, 1)
    lines = [line.strip() for line in after.strip().splitlines() if line.strip()]
    existing = [line[2:].strip() for line in lines if line.startswith("- ")]
    return before.rstrip(), existing


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


async def update_skill_with_feedback(current_skill_text: str, feedbacks: list[str]) -> str:
    """Learn user-specific preferences from feedback via LLM and keep one section."""
    base_text, existing_items = _split_feedback_section(current_skill_text)
    cleaned_feedbacks = [_strip_feedback_prefix(item) for item in feedbacks if item.strip()]
    if not cleaned_feedbacks:
        return base_text + "\n"

    feedback_block = "\n".join(f"- {item}" for item in cleaned_feedbacks)
    existing_block = "\n".join(f"- {item}" for item in existing_items) or "- (none)"
    update_output = await LLMGateway.acreate_structured_output(
        (
            f"Current skill:\n{base_text}\n\n"
            f"Existing preference updates:\n{existing_block}\n\n"
            f"New feedback items:\n{feedback_block}"
        ),
        (
            "Infer user-specific job preference rules from feedback and return concise, reusable bullet points. "
            "Focus on stable preferences (work style, constraints, role scope, domain fit, seniority fit, "
            "technical stack tolerance). Avoid job-specific details. Keep each preference actionable and generic."
        ),
        SkillPreferenceUpdates,
    )
    merged_items = _dedupe_keep_order([*existing_items, *update_output.preferences])[:16]

    if not merged_items:
        return base_text + "\n"

    bullets = "\n".join(f"- {item}" for item in merged_items)
    return f"{base_text}\n\n## Feedback Updates\n{bullets}\n"
