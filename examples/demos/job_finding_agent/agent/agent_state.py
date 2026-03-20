"""Loop state models for the modular job-finding agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .agent_models import JobFeedbackTriplet, RecommendationOutput, ToolName


@dataclass
class AgentActionRecord:
    """One Think/Act/Observe trace step."""

    iteration: int
    thought: str
    tool_name: ToolName
    observation: str
    continue_loop: bool
    stop_reason: Optional[str] = None


@dataclass
class JobAgentState:
    """Mutable state for a single job run."""

    job: JobFeedbackTriplet
    skill_text: str
    active: bool = True
    iteration: int = 0
    formatted_job: Optional[dict[str, Any]] = None
    recommendation: Optional[RecommendationOutput] = None
    feedback_text: Optional[str] = None
    pending_feedbacks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobAgentLoopResult:
    """Result object returned by the loop engine."""

    final_state: JobAgentState
    action_trace: list[AgentActionRecord]
    termination_reason: str

