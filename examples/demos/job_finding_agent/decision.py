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
    required_tool_by_stage = {
        "PROCESS_JOB": ToolName.PROCESS_JOB_AGENT.value,
        "REQUEST_FEEDBACK": ToolName.REQUEST_FEEDBACK.value,
        "DONE": ToolName.UPDATE_PROCESS_JOB_AGENT_SKILL.value,
    }
    required_tool = required_tool_by_stage[current_stage]

    prompt = (
        "You are the controller for ONE job-level agent loop.\n\n"
        "STRICT TRANSITION POLICY (no exceptions):\n"
        "Stage PROCESS_JOB -> ONLY ProcessJobAgent is valid.\n"
        "Stage REQUEST_FEEDBACK -> ONLY RequestFeedback is valid.\n"
        "Stage DONE -> ONLY UpdateProcessJobAgentSkill is valid.\n\n"
        "Forbidden behavior:\n"
        "- Never call ProcessJobAgent after a recommendation already exists.\n"
        "- Never call RequestFeedback before a recommendation exists.\n"
        "- Never skip UpdateProcessJobAgentSkill once feedback exists.\n"
        "- Never repeat a finished stage.\n\n"
        "Execution goal for this step:\n"
        f"- Current stage requires EXACT tool: {required_tool}\n"
        "- Return that tool in tool_name.\n"
        "- Keep continue_loop=true unless the loop should immediately stop.\n\n"
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
            "Select exactly one tool. Follow strict stage policy. "
            "If your selected tool differs from the required stage tool, it is incorrect."
        ),
        NextToolDecision,
    )

    return result
