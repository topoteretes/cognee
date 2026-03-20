"""Unit tests for the reusable job agent loop."""

from pathlib import Path

import pytest

from examples.demos.job_finding_agent.agent.agent_loop import run_job_agent_loop
from examples.demos.job_finding_agent.agent.agent_models import (
    JobFeedbackTriplet,
    NextToolDecision,
    ToolName,
)
from examples.demos.job_finding_agent.agent.agent_state import JobAgentState
from examples.demos.job_finding_agent.agent.tool_contracts import (
    RunnerContext,
    ToolExecutionResult,
)


@pytest.mark.asyncio
async def test_loop_stops_at_max_iterations():
    async def endless_decision(state, _tools, _context):
        return NextToolDecision(
            thought="Keep processing",
            tool_name=ToolName.PROCESS_JOB_AGENT,
            continue_loop=True,
        )

    async def process_tool(_state, _context):
        return ToolExecutionResult(observation="processed", continue_loop=True)

    state = JobAgentState(
        job=JobFeedbackTriplet(
            job_id="j1",
            job_description="test job",
            feedback_if_recommended="good",
            feedback_if_not_recommended="bad",
        ),
        skill_text="skill",
    )
    context = RunnerContext(
        dataset_name="d",
        session_id="s",
        skill_md_path=Path("/tmp/skill.md"),
    )
    result = await run_job_agent_loop(
        initial_state=state,
        tool_registry={ToolName.PROCESS_JOB_AGENT: process_tool},
        decision_fn=endless_decision,
        context=context,
        max_iterations=10,
    )

    assert result.termination_reason == "MAX_ITERATIONS_REACHED"
    assert result.final_state.iteration == 10
    assert len(result.action_trace) == 10


@pytest.mark.asyncio
async def test_tool_requested_termination_stops_process():
    async def choose_feedback(_state, _tools, _context):
        return NextToolDecision(
            thought="Tool asks for termination.",
            tool_name=ToolName.REQUEST_FEEDBACK,
            continue_loop=False,
            stop_reason="end current job",
        )

    async def feedback_tool(_state, _context):
        return ToolExecutionResult(
            observation="feedback saved",
            should_end_process=True,
            continue_loop=False,
            stop_reason="TOOL_DONE",
        )

    state = JobAgentState(
        job=JobFeedbackTriplet(
            job_id="j2",
            job_description="test job",
            feedback_if_recommended="good",
            feedback_if_not_recommended="bad",
        ),
        skill_text="skill",
    )
    context = RunnerContext(
        dataset_name="d",
        session_id="s",
        skill_md_path=Path("/tmp/skill.md"),
    )
    result = await run_job_agent_loop(
        initial_state=state,
        tool_registry={ToolName.REQUEST_FEEDBACK: feedback_tool},
        decision_fn=choose_feedback,
        context=context,
        max_iterations=10,
    )

    assert result.termination_reason == "TOOL_DONE"
    assert result.final_state.iteration == 1
    assert len(result.action_trace) == 1
    assert result.action_trace[0].tool_name == ToolName.REQUEST_FEEDBACK
