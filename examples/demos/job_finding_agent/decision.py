"""Structured next-tool decision function."""

from __future__ import annotations

from cognee.infrastructure.llm.LLMGateway import LLMGateway

from .agent.agent_models import NextToolDecision, ToolName
from .agent.agent_state import JobAgentState
from .agent.tool_contracts import RunnerContext


async def structured_decision_fn(
    state: JobAgentState,
    available_tools: list[ToolName],
    context: RunnerContext,
) -> NextToolDecision:
    """Let the LLM choose one of the three allowed tools."""
    tool_list = ", ".join(tool.value for tool in available_tools)
    recommendation_status = "present" if state.recommendation is not None else "missing"
    pending_feedback = bool(context.runtime_data.get("pending_feedbacks"))

    prompt = (
        f"Job id: {state.job.job_id}\n"
        f"Current iteration: {state.iteration}\n"
        f"Recommendation status: {recommendation_status}\n"
        f"Pending feedback from previous jobs: {pending_feedback}\n"
        f"Allowed tools: {tool_list}\n"
        "Choose the next best tool. "
        "If recommendation is missing, choose ProcessJobAgent. "
        "If recommendation exists for this job, choose RequestFeedback to end this job.\n"
    )

    return await LLMGateway.acreate_structured_output(
        prompt,
        (
            "You are a job-finding agent controller. "
            "Select exactly one allowed tool and set continue_loop accordingly."
        ),
        NextToolDecision,
    )

