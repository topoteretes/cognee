"""Reusable Think/Act/Observe/Repeat loop engine."""

from __future__ import annotations

from cognee.shared.logging_utils import get_logger

from examples.demos.job_finding_agent.agent.agent_models import ToolName
from examples.demos.job_finding_agent.agent.agent_state import (
    AgentActionRecord,
    JobAgentLoopResult,
    JobAgentState,
)
from examples.demos.job_finding_agent.agent.tool_contracts import (
    DecisionFunction,
    RunnerContext,
    ToolRegistry,
)

logger = get_logger("job_finding_agent_loop")


async def run_job_agent_loop(
    initial_state: JobAgentState,
    tool_registry: ToolRegistry,
    decision_fn: DecisionFunction,
    context: RunnerContext,
    max_iterations: int = 10,
) -> JobAgentLoopResult:
    """
    Run one modular job-agent loop.

    This function is intentionally generic so it can be reused as a building block in
    larger agent orchestration systems.
    """
    state = initial_state
    trace: list[AgentActionRecord] = []
    allowed_tools = list(tool_registry.keys())
    termination_reason = "UNKNOWN"

    while state.active and state.iteration < max_iterations:
        state.iteration += 1
        try:
            next_step = await decision_fn(state, allowed_tools, context)
            if next_step.tool_name not in allowed_tools:
                raise ValueError(f"Unsupported tool selected: {next_step.tool_name}")

            tool = tool_registry[next_step.tool_name]
            result = await tool(state, context)

            continue_loop = bool(next_step.continue_loop and result.continue_loop)
            stop_reason = result.stop_reason or next_step.stop_reason
            trace.append(
                AgentActionRecord(
                    iteration=state.iteration,
                    thought=next_step.thought,
                    tool_name=next_step.tool_name,
                    observation=result.observation,
                    continue_loop=continue_loop,
                    stop_reason=stop_reason,
                )
            )

            if result.should_end_process:
                state.active = False
                termination_reason = result.stop_reason or "TOOL_REQUESTED_TERMINATION"
            elif not continue_loop:
                state.active = False
                termination_reason = stop_reason or "DECISION_STOPPED_LOOP"

        except Exception as error:
            state.active = False
            termination_reason = f"ERROR: {error}"
            logger.error("Agent loop failed on iteration %s: %s", state.iteration, error)
            trace.append(
                AgentActionRecord(
                    iteration=state.iteration,
                    thought="Tool execution failed",
                    tool_name=ToolName.PROCESS_JOB_AGENT,
                    observation=str(error),
                    continue_loop=False,
                    stop_reason="ERROR",
                )
            )

    if state.active and state.iteration >= max_iterations:
        state.active = False
        termination_reason = "MAX_ITERATIONS_REACHED"

    if termination_reason == "UNKNOWN":
        termination_reason = "LOOP_EXITED"

    return JobAgentLoopResult(
        final_state=state,
        action_trace=trace,
        termination_reason=termination_reason,
    )
