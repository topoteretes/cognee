"""Context-stuffing benchmark implementation.

Appends ALL past full conversation transcripts (across all personas) as raw text
into the sales agent's prompt. Single shared history, not per-persona.
This deliberately bloats the prompt to show context stuffing degrading at scale.
"""

from __future__ import annotations

import json

from cognee.infrastructure.llm.LLMGateway import LLMGateway

from .agents import (
    run_conversation,
    sales_agent_turn,
    setup_runtime,
)
from .leads import LEADS, BuyingProfile
from .metrics import MetricsCollector
from .models import ConversationResult, ContextSummary, SalesResponse

SUMMARIZE_AFTER = 10

_all_transcripts: list[str] = []
_summary: str = ""


def _format_full_transcript(lead: BuyingProfile, result: ConversationResult) -> str:
    """Format the full conversation transcript for context stuffing."""
    lines = [
        f"=== Lead {lead.lead_id} ===",
        f"Customer opening: {lead.initial_message}",
    ]
    for msg in result.conversation_history:
        role = msg["role"].upper()
        lines.append(f"  [{role}]: {msg['message']}")
    lines.append(f"OUTCOME: {result.outcome} after {result.rounds} rounds")
    if result.winning_feature:
        lines.append(f"WINNING: feature={result.winning_feature}, angle={result.winning_angle}")
    return "\n".join(lines)


async def sales_turn_context(
    conversation_history: list, lead_intro: str, round_num: int, memory_context: str
) -> SalesResponse:
    return await sales_agent_turn(conversation_history, lead_intro, round_num, memory_context)


async def _summarize_transcripts() -> str:
    """Use LLM to summarize all accumulated transcripts."""
    all_text = "\n---\n".join(_all_transcripts)
    result = await LLMGateway.acreate_structured_output(
        text_input=all_text,
        system_prompt=(
            "Summarize these sales conversation logs. For each customer persona type, "
            "state which Cognee features and pitch angles led to CLOSED_WON vs CLOSED_LOST. "
            "Be concise but preserve all actionable patterns."
        ),
        response_model=ContextSummary,
    )
    return result.summary


async def setup_context() -> None:
    global _all_transcripts, _summary
    _all_transcripts = []
    _summary = ""
    await setup_runtime()


async def run_all_leads(collector: MetricsCollector) -> list:
    global _all_transcripts, _summary
    results = []
    for i, lead in enumerate(LEADS):
        print(f"\n--- Lead {lead.lead_id}: {lead.persona_tag} ---")

        # Build context: summary (if exists) + recent transcripts
        parts = []
        if _summary:
            parts.append(f"SUMMARY OF EARLIER CONVERSATIONS:\n{_summary}")
        if _all_transcripts:
            parts.append("RECENT CONVERSATION LOGS:\n" + "\n---\n".join(_all_transcripts))

        if parts:
            memory_context = "\n\n".join(parts)
            print(f"  [context] {len(memory_context)} chars (summary={'yes' if _summary else 'no'}, transcripts={len(_all_transcripts)})")
        else:
            memory_context = ""

        collector.start_lead(lead.lead_id, lead.persona_tag, "context")

        result = await run_conversation(
            sales_turn_context,
            lead,
            memory_context=memory_context,
        )

        collector.end_lead(result)
        results.append(result)
        print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")

        transcript = _format_full_transcript(lead, result)
        _all_transcripts.append(transcript)

        # After SUMMARIZE_AFTER transcripts, compress into a summary
        if len(_all_transcripts) >= SUMMARIZE_AFTER:
            print(f"  [context] Summarizing {len(_all_transcripts)} transcripts...")
            _summary = await _summarize_transcripts()
            _all_transcripts = []

    return results
