"""Structured next-tool decision function."""

from __future__ import annotations

from cognee.infrastructure.llm.LLMGateway import LLMGateway

from examples.demos.job_finding_agent.agent.agent_models import NextToolDecision, ToolName
from examples.demos.job_finding_agent.agent.agent_state import JobAgentState
from examples.demos.job_finding_agent.agent.tool_contracts import RunnerContext


async def structured_decision_fn(
    state: JobAgentState,
    available_tools: list[ToolName],
    context: RunnerContext,
) -> NextToolDecision:
    """Let the LLM choose one of the three allowed tools."""
    tool_list = ", ".join(tool.value for tool in available_tools)
    recommendation_status = "present" if state.recommendation is not None else "missing"
    max_iterations = context.runtime_data.get("max_iterations", 10)
    has_feedback_for_job = bool(state.feedback_text)
    current_stage = (
        "PROCESS_JOB"
        if state.recommendation is None
        else ("REQUEST_FEEDBACK" if not has_feedback_for_job else "DONE")
    )

    prompt = (
        "You are the controller for one job-level agent loop.\n\n"
        "What we are doing:\n"
        "1) Process this job and produce APPLY/DONT_APPLY recommendation.\n"
        "2) Request feedback for that recommendation.\n"
        "3) Update the job evaluation skill from latest feedback.\n"
        "4) End the current job process.\n\n"
        "Tool meanings:\n"
        "- ProcessJobAgent: structure job + create recommendation.\n"
        "- RequestFeedback: capture feedback for current recommendation.\n"
        "- UpdateProcessJobAgentSkill: refresh skill from accumulated feedback memory and finish this job.\n\n"
        "Decision rules:\n"
        "- If recommendation is missing: choose ProcessJobAgent.\n"
        "- If recommendation exists and feedback for this job is missing: choose RequestFeedback.\n"
        "- If recommendation and feedback both exist: choose UpdateProcessJobAgentSkill.\n"
        "- If this job is already complete: set continue_loop=false with stop_reason.\n\n"
        "Context:\n"
        f"Job id: {state.job.job_id}\n"
        f"Current stage: {current_stage}\n"
        f"Current iteration: {state.iteration}\n"
        f"Max iterations: {max_iterations}\n"
        f"Recommendation status: {recommendation_status}\n"
        f"Feedback status: {'present' if has_feedback_for_job else 'missing'}\n"
        f"Allowed tools: {tool_list}\n"
    )

    result = await LLMGateway.acreate_structured_output(
        prompt,
        (
            "You are a job-finding agent controller. "
            "Select exactly one allowed tool and set continue_loop accordingly."
        ),
        NextToolDecision,
    )

    return result
