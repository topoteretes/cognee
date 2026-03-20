"""Typed models for the modular job-finding agent loop."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from cognee.low_level import DataPoint


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


class FormattedJobOutput(DataPoint):
    """Structured extraction for job text."""

    role_title: str = Field(min_length=1)
    job_sequence: Optional[DataPoint] = None
    recommendation: Optional["RecommendationOutput"] = None
    text: str = ""
    metadata: dict = {"index_fields": ["text"]}


class RecommendationOutput(DataPoint):
    """Structured recommendation produced by the job processor."""

    decision: RecommendationDecision
    rationale: str = Field(min_length=1)
    text: str = ""
    metadata: dict = {"index_fields": ["text"]}


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
