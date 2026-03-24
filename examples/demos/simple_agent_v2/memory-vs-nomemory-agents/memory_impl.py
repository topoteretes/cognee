"""Memory-enabled streaming email offer demo."""

from __future__ import annotations

from examples.demos.simple_agent_v2.agentic_context_trace import agentic_trace_root

from common import check_eligibility, propose_offer, run_stream_impl, setup_runtime


@agentic_trace_root(
    with_memory=True,
    save_traces=False,
    task_query="List ALL method return values where from the context, Focus on the feedbacks only.",
)
async def _subagent_propose_offer(payload: dict) -> dict:
    return await propose_offer(payload)


@agentic_trace_root(with_memory=False, save_traces=True)
async def _subagent_check_eligibility(payload: dict) -> dict:
    return await check_eligibility(payload)


async def setup_memory() -> None:
    await setup_runtime()


async def run_stream() -> None:
    await run_stream_impl(
        subagent_propose_offer=_subagent_propose_offer,
        subagent_check_eligibility=_subagent_check_eligibility,
    )
