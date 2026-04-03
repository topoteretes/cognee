"""No-memory implementation of the simple agent demo."""

from __future__ import annotations

from common import check_eligibility, propose_offer, run_stream_impl, setup_runtime


async def _subagent_propose_offer(payload: dict, email: dict) -> dict:
    return await propose_offer(payload, email)


async def _subagent_check_eligibility(payload: dict) -> dict:
    return await check_eligibility(payload)


async def setup_nomemory() -> None:
    await setup_runtime()


async def run_stream() -> None:
    await run_stream_impl(
        subagent_propose_offer=_subagent_propose_offer,
        subagent_check_eligibility=_subagent_check_eligibility,
    )
