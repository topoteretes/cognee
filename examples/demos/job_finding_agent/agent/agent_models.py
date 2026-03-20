"""Typed models for the modular job-finding agent loop."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    """Tools available to the agent."""

    PROCESS_JOB_AGENT = "ProcessJobAgent"
    UPDATE_PROCESS_JOB_AGENT_SKILL = "UpdateProcessJobAgentSkill"
    REQUEST_FEEDBACK = "RequestFeedback"


class RecommendationDecision(str, Enum):
    """Recommendation output enum."""

    APPLY = "APPLY"
    DONT_APPLY = "DONT_APPLY"


class JobFeedbackTriplet(BaseModel):
    """One mocked input job and its feedback branches."""

    job_id: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    feedback_if_recommended: str = Field(min_length=1)
    feedback_if_not_recommended: str = Field(min_length=1)


class FormattedJobOutput(BaseModel):
    """Structured extraction for job text."""

    role_title: str = ""
    seniority: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    location_or_remote: str = ""


class RecommendationOutput(BaseModel):
    """Structured recommendation produced by the job processor."""

    decision: RecommendationDecision
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class NextToolDecision(BaseModel):
    """Decision object for one loop step."""

    thought: str = Field(min_length=1)
    tool_name: ToolName
    continue_loop: bool = True
    stop_reason: Optional[str] = None


class SkillProfileOutput(BaseModel):
    """Initial CV-to-skill extraction."""

    profile_summary: str
    core_strengths: list[str] = []
    heuristics: list[str] = []

