"""No-memory sales benchmark implementation with parallel execution."""

from __future__ import annotations

import asyncio
import time

from .agents import run_conversation, sales_agent_turn, setup_runtime
from .leads import LEADS, BuyingProfile
from .metrics import LeadMetrics, MetricsCollector
from .models import ConversationResult, SalesResponse

CONCURRENCY = 10


async def sales_turn_no_memory(
    conversation_history: list, lead_intro: str, round_num: int
) -> SalesResponse:
    return await sales_agent_turn(conversation_history, lead_intro, round_num)


async def setup_nomemory() -> None:
    await setup_runtime()


async def _run_one_lead(lead: BuyingProfile) -> tuple[BuyingProfile, ConversationResult, float]:
    """Run a single lead conversation, return (lead, result, wall_time)."""
    t0 = time.monotonic()
    result = await run_conversation(sales_turn_no_memory, lead)
    wall = round(time.monotonic() - t0, 2)
    print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")
    return lead, result, wall


async def run_all_leads(collector: MetricsCollector) -> list:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def bounded(lead):
        async with semaphore:
            return await _run_one_lead(lead)

    print(f"  [nomemory] Running {len(LEADS)} leads with concurrency={CONCURRENCY}")
    lead_results = await asyncio.gather(*[bounded(lead) for lead in LEADS])

    # Build per-lead metrics and distribute bulk token counts proportionally by rounds
    total_rounds = sum(r.rounds for _, r, _ in lead_results)
    results = []
    for lead, result, wall in lead_results:
        # Distribute bulk tokens proportionally by number of rounds (proxy for LLM calls)
        share = result.rounds / total_rounds if total_rounds > 0 else 0
        m = LeadMetrics(
            lead_id=lead.lead_id,
            persona_tag=lead.persona_tag,
            mode="nomemory",
            outcome=result.outcome,
            rounds=result.rounds,
            wall_time_s=wall,
            llm_calls=result.rounds * 2,  # 1 sales + 1 customer per round
            prompt_tokens=round(collector._bulk_prompt_tokens * share),
            completion_tokens=round(collector._bulk_completion_tokens * share),
            r1_feature_correct=(
                result.features_pitched[0] == lead.must_have_feature
                if result.features_pitched else False
            ),
        )
        collector.results.append(m)
        results.append(result)
    return results
