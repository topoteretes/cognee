"""Skill generation and update logic."""

from __future__ import annotations

from pathlib import Path

from cognee.infrastructure.llm.LLMGateway import LLMGateway

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
        "- Confidence: float in [0,1]\n"
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


def update_skill_with_feedback(current_skill_text: str, latest_feedback: str) -> str:
    """Append one deterministic feedback-derived rule block."""
    appended_rule = f"- Feedback learned: {latest_feedback.strip()}"
    return current_skill_text.rstrip() + "\n\n## Feedback Updates\n" + appended_rule + "\n"
