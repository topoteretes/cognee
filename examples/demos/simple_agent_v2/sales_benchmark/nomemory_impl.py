"""No-memory sales benchmark implementation."""

from __future__ import annotations

from .agents import run_conversation, sales_agent_turn, setup_runtime
from .leads import LEADS
from .metrics import MetricsCollector
from .models import SalesResponse


async def sales_turn_no_memory(
    conversation_history: list, lead_intro: str, round_num: int
) -> SalesResponse:
    return await sales_agent_turn(conversation_history, lead_intro, round_num)


async def setup_nomemory() -> None:
    await setup_runtime()


async def run_all_leads(collector: MetricsCollector) -> list:
    results = []
    for lead in LEADS:
        print(f"\n--- Lead {lead.lead_id}: {lead.persona_tag} ---")
        collector.start_lead(lead.lead_id, lead.persona_tag, "nomemory")
        result = await run_conversation(
            sales_turn_no_memory,
            lead,
        )
        collector.end_lead(result)
        results.append(result)
        print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")
    return results
