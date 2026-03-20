"""High-level orchestration for CV init + per-job loop execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cognee
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user

from examples.demos.job_finding_agent.agent.agent_loop import run_job_agent_loop
from examples.demos.job_finding_agent.agent.agent_models import ToolName
from examples.demos.job_finding_agent.agent.agent_state import JobAgentLoopResult, JobAgentState
from examples.demos.job_finding_agent.agent.tool_contracts import RunnerContext
from examples.demos.job_finding_agent.config import (
    ACTION_DATASET_NAME,
    APPLICANT_DATASET_NAME,
    CV_FILE,
    DATA_FILE,
    MAX_ITERATIONS,
    SESSION_ID,
    SKILL_FILE,
)
from examples.demos.job_finding_agent.decision import structured_decision_fn
from examples.demos.job_finding_agent.io_utils import read_mock_jobs
from examples.demos.job_finding_agent.skill_logic import reset_skill_file_from_cv
from examples.demos.job_finding_agent.tools import (
    process_job_agent_tool,
    request_feedback_tool,
    store_agent_action,
    update_process_job_agent_skill_tool,
)


def _format_action_trace_for_session(loop_result: JobAgentLoopResult) -> str:
    """Render full per-job action trace to answer text for session storage."""
    if not loop_result.action_trace:
        return "No actions recorded."

    lines: list[str] = []
    for step in loop_result.action_trace:
        lines.append(
            (
                f"Iteration {step.iteration} | Tool={step.tool_name.value} | "
                f"Thought={step.thought} | Observation={step.observation} | "
                f"Continue={step.continue_loop} | Stop={step.stop_reason or ''}"
            )
        )
    return "\n".join(lines)


async def run_jobs_from_json(
    cv_text: str | None = None,
    cv_path: Path = CV_FILE,
    jobs_path: Path = DATA_FILE,
    skill_path: Path = SKILL_FILE,
    jobs_dataset_name: str = APPLICANT_DATASET_NAME,
    action_dataset_name: str = ACTION_DATASET_NAME,
    session_id: str = SESSION_ID,
) -> dict[str, Any]:
    """Entry point for the full flow: CV init + per-job modular loop."""
    jobs = read_mock_jobs(jobs_path)

    # Ensure a deterministic clean run for the demo lifecycle.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    if cv_text is None:
        cv_text = cv_path.read_text(encoding="utf-8")

    # Always reset to the initial non-updated skill at the start of a full run.
    skill_text = await reset_skill_file_from_cv(cv_text, skill_path)

    # Initial ingestion is CV-only in applicant_data dataset.
    await cognee.add(
        cv_text,
        dataset_name=APPLICANT_DATASET_NAME,
        user=user,
    )
    await cognee.cognify(datasets=[APPLICANT_DATASET_NAME], user=user)

    context = RunnerContext(
        dataset_name=jobs_dataset_name,
        session_id=session_id,
        skill_md_path=skill_path,
        user=user,
        skill_text=skill_text,
        runtime_data={
            "pending_feedbacks": [],
            "max_iterations": MAX_ITERATIONS,
            "action_dataset_name": action_dataset_name,
            "last_job_sequence_node": None,
            "last_action_task_job_node": None,
            "last_skill_snapshot": None,
            "skill_snapshot_version": 0,
        },
    )

    tool_registry = {
        ToolName.PROCESS_JOB_AGENT: process_job_agent_tool,
        ToolName.UPDATE_PROCESS_JOB_AGENT_SKILL: update_process_job_agent_skill_tool,
        ToolName.REQUEST_FEEDBACK: request_feedback_tool,
    }

    results: list[dict[str, Any]] = []
    session_manager = get_session_manager()
    for job_index, job in enumerate(jobs, start=1):
        state = JobAgentState(
            job=job,
            skill_text=context.skill_text,
            pending_feedbacks=list(context.runtime_data.get("pending_feedbacks", [])),
            metadata={"job_sequence_index": job_index},
        )
        loop_result = await run_job_agent_loop(
            initial_state=state,
            tool_registry=tool_registry,
            decision_fn=structured_decision_fn,
            context=context,
            max_iterations=MAX_ITERATIONS,
        )

        previous_action_dp = None
        for action in loop_result.action_trace:
            previous_action_dp = await store_agent_action(
                state=loop_result.final_state,
                context=context,
                iteration=action.iteration,
                thought=action.thought,
                tool_name=action.tool_name,
                observation=action.observation,
                stop_reason=action.stop_reason,
                prev_action=previous_action_dp,
            )

        feedback_score = None
        if loop_result.final_state.recommendation is not None:
            feedback_score = (
                5 if loop_result.final_state.recommendation.decision.value == "APPLY" else 2
            )

        recommendation_value = (
            loop_result.final_state.recommendation.decision.value
            if loop_result.final_state.recommendation
            else "NONE"
        )
        question_text = (
            f"Job {job.job_id} | Decision {recommendation_value} | "
            f"Termination {loop_result.termination_reason} | Actions"
        )
        answer_text = (
            f"Job Description:\n{job.job_description}\n\n"
            f"Agent Actions:\n{_format_action_trace_for_session(loop_result)}\n\n"
            f"Feedback:\n{loop_result.final_state.feedback_text or 'No feedback recorded.'}"
        )

        await session_manager.add_qa(
            user_id=str(user.id),
            session_id=session_id,
            question=question_text,
            context="",
            answer=answer_text,
            feedback_text=loop_result.final_state.feedback_text,
            feedback_score=feedback_score,
        )

        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=jobs_dataset_name,
            run_in_background=False,
        )
        await session_manager.delete_session(
            user_id=str(user.id),
            session_id=session_id,
        )

        results.append(
            {
                "job_id": job.job_id,
                "decision": (
                    loop_result.final_state.recommendation.decision.value
                    if loop_result.final_state.recommendation
                    else None
                ),
                "termination_reason": loop_result.termination_reason,
                "iterations": loop_result.final_state.iteration,
                "feedback_text": loop_result.final_state.feedback_text,
            }
        )

    return {
        "jobs_dataset_name": jobs_dataset_name,
        "action_dataset_name": action_dataset_name,
        "session_id": session_id,
        "max_iterations": MAX_ITERATIONS,
        "jobs_processed": len(results),
        "results": results,
        "skill_md_path": str(skill_path),
    }
