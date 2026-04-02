"""Memory-enabled implementation of the simple agent demo."""

from __future__ import annotations

import cognee

from common import (
    AGENTIC_TRACES_DATASET,
    check_eligibility,
    propose_offer,
    run_stream_impl,
    setup_runtime,
)


@cognee.agent_memory(
    with_memory=True,
    save_traces=False,
    dataset_name=AGENTIC_TRACES_DATASET,
    memory_query_from_method="email",
)
async def _subagent_propose_offer(payload: dict, email: str) -> dict:
    return await propose_offer(payload, email)


@cognee.agent_memory(
    with_memory=False,
    save_traces=True,
    dataset_name=AGENTIC_TRACES_DATASET,
)
async def _subagent_check_eligibility(payload: dict) -> dict:
    return await check_eligibility(payload)


async def setup_memory() -> None:
    await setup_runtime()


async def run_stream() -> None:
    await run_stream_impl(
        subagent_propose_offer=_subagent_propose_offer,
        subagent_check_eligibility=_subagent_check_eligibility,
    )
