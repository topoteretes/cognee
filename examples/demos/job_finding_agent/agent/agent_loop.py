"""Reusable Think/Act/Observe/Repeat loop engine."""

from __future__ import annotations

import textwrap

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


def _normalize_text(value: str | None) -> str:
    """Normalize whitespace for readable terminal progress logs without truncation."""
    if not value:
        return ""
    return " ".join(value.split())


def _format_block_field(
    label: str,
    value: str,
    content_width: int,
) -> list[str]:
    """Render one field as wrapped hashtag-prefixed lines."""
    normalized = _normalize_text(value)
    prefix = f"# {label}: "
    continuation_prefix = "# " + (" " * (len(label) + 2))
    line_width = max(20, content_width - len(prefix))
    wrapped = textwrap.wrap(normalized, width=line_width) or [""]
    lines = [f"{prefix}{wrapped[0]}"]
    lines.extend(f"{continuation_prefix}{line}" for line in wrapped[1:])
    return lines


def _format_usage_lines(values: list[str], content_width: int) -> list[str]:
    """Render usage lines as wrapped hashtag bullet lines."""
    if not values:
        values = ["No Cognee calls recorded by this tool."]
    lines: list[str] = []
    bullet_prefix = "# - "
    continuation_prefix = "#   "
    line_width = max(20, content_width - len(bullet_prefix))
    for value in values:
        wrapped = textwrap.wrap(_normalize_text(value), width=line_width) or [""]
        lines.append(f"{bullet_prefix}{wrapped[0]}")
        lines.extend(f"{continuation_prefix}{part}" for part in wrapped[1:])
    return lines


def _build_iteration_block(
    *,
    job_id: str,
    iteration: int,
    think: str,
    recommendation_status: str,
    feedback_status: str,
    tool_name: str,
    observation: str,
    tool_usage: list[str],
    should_stop: bool,
) -> str:
    """Build one large hashtag-framed block for a single iteration."""
    border_width = 120
    border = "#" * border_width
    content_width = border_width - 2

    lines: list[str] = [
        border,
        f"# ITERATION BLOCK | Job={job_id} | Iteration={iteration}",
        border,
    ]
    lines.extend(_format_block_field("Think", think, content_width))
    lines.extend(
        _format_block_field(
            "State",
            f"recommendation={recommendation_status}, feedback={feedback_status}",
            content_width,
        )
    )
    lines.extend(_format_block_field("Act", f'Use the "{tool_name}" TOOL', content_width))
    lines.append("#")
    lines.append("# COGNEE USAGE")
    lines.extend(_format_usage_lines(tool_usage, content_width))
    lines.append("#")
    lines.extend(_format_block_field("Observe", observation, content_width))
    lines.extend(_format_block_field("Stop/Repeat", "stop" if should_stop else "repeat", content_width))
    lines.append(border)
    return "\n" + "\n".join(lines) + "\n"


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
            recommendation_status_before = "yes" if state.recommendation is not None else "no"
            feedback_status_before = "yes" if bool(state.feedback_text) else "no"

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
                    cognee_usage=list(result.cognee_usage),
                    stop_reason=stop_reason,
                )
            )

            if context.runtime_data.get("show_progress", True):
                iteration_block = _build_iteration_block(
                    job_id=state.job.job_id,
                    iteration=state.iteration,
                    think=next_step.thought,
                    recommendation_status=recommendation_status_before,
                    feedback_status=feedback_status_before,
                    tool_name=next_step.tool_name.value,
                    observation=result.observation,
                    tool_usage=list(result.cognee_usage),
                    should_stop=bool(result.should_end_process or not continue_loop),
                )
                print(iteration_block)

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
            if context.runtime_data.get("show_progress", True):
                error_block = _build_iteration_block(
                    job_id=state.job.job_id,
                    iteration=state.iteration,
                    think="Tool execution failed while running the selected action.",
                    recommendation_status="yes" if state.recommendation is not None else "no",
                    feedback_status="yes" if bool(state.feedback_text) else "no",
                    tool_name=ToolName.PROCESS_JOB_AGENT.value,
                    observation=f"ERROR: {error}",
                    tool_usage=[],
                    should_stop=True,
                )
                print(error_block)
            trace.append(
                AgentActionRecord(
                    iteration=state.iteration,
                    thought="Tool execution failed",
                    tool_name=ToolName.PROCESS_JOB_AGENT,
                    observation=str(error),
                    continue_loop=False,
                    cognee_usage=[],
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
