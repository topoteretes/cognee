"""Tool contracts for the job-finding agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from cognee.modules.users.models import User

from examples.demos.job_finding_agent.agent.agent_models import ToolName
from examples.demos.job_finding_agent.agent.agent_state import JobAgentState


@dataclass
class ToolExecutionResult:
    """Typed tool response consumed by the loop engine."""

    observation: str
    should_end_process: bool = False
    continue_loop: bool = True
    stop_reason: str | None = None
    cognee_usage: list[str] = field(default_factory=list)


@dataclass
class RunnerContext:
    """Shared context across jobs and tools."""

    dataset_name: str
    session_id: str
    skill_md_path: Path
    user: User | None = None
    skill_text: str = ""
    runtime_data: dict = field(default_factory=dict)


class AgentTool(Protocol):
    """Protocol for tool implementations."""

    async def __call__(
        self,
        state: JobAgentState,
        context: RunnerContext,
    ) -> ToolExecutionResult: ...


DecisionFunction = Callable[[JobAgentState, list[ToolName], RunnerContext], Awaitable]
ToolRegistry = dict[ToolName, AgentTool]
