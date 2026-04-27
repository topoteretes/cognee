"""Three @cognee.agent_memory-wrapped tools for Staff Backend interview scheduling.

Tools are plain domain functions. They return pure domain objects. They do
NOT know they're being observed — no proposal fields, no compliance_notes,
no applied_rule_ids. The decorator transparently:
  - retrieves relevant rules from `human_memory` (when with_memory=True)
    and prepends them to the LLM system prompt,
  - saves a structured trace entry per call via SessionManager,
  - asks the LLM for a one-sentence `session_feedback` summary of the return
    value (session_trace_summary=True, default).

Learning extraction happens after the agentic loop finishes, via the
memify pipeline `persist_agent_trace_feedbacks_in_knowledge_graph` that
reads those stored session_feedback summaries.

Env vars read at import time:
  RECRUITING_WITH_MEMORY   "true" (default) / "false" — flips naive vs grounded
  RECRUITING_SESSION_ID    session id for trace persistence
  RECRUITING_MEMORY_TOP_K  retrieval breadth (default: "6")
"""

import os

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway

from examples.demos.recruiting_distill_memory.rule_models import (
    InterviewFormatProposal,
    PanelSchedule,
    ScreenInvite,
)

DATASET = "human_memory"
WITH_MEMORY = os.environ.get("RECRUITING_WITH_MEMORY", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
CANDIDATE = os.environ.get("RECRUITING_CANDIDATE", "dev_rao").strip()
_MODE_TAG = "grounded" if WITH_MEMORY else "naive"
SESSION_ID = os.environ.get(
    "RECRUITING_SESSION_ID",
    f"recruiting-demo-{_MODE_TAG}-{CANDIDATE}",
).strip()
MEMORY_TOP_K = int(os.environ.get("RECRUITING_MEMORY_TOP_K", "6"))


_ALEX_PERSONA = (
    "You are a senior recruiter at Ledgerline (a 40-person payments fintech "
    "based in Berlin) drafting decisions for a Staff Backend Engineer hire. "
    "If an 'Additional Memory Context' block is present above the user "
    "input, it contains Ledgerline-specific policies — FOLLOW THEM "
    "LITERALLY, even when they look narrower than industry best practice. "
    "If a rule specifies an exact number, an exact set of names, an exact "
    "phrase, or an exact position in a list, reproduce it verbatim — do not "
    "substitute a generic value or round to a common default. If no memory "
    "context is present, fall back to general recruiting best practice. "
    "Respond only with the structured fields requested."
)


_FORMAT_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: choose the interview format. Fields: "
    "`format` ∈ {live_coding, take_home, system_design, behavioral_only}, "
    "`duration_minutes` (int — use the exact value the memory context "
    "prescribes; do not round to 60 or 90), "
    "`medium` ∈ {video, onsite, phone}."
)

_PANEL_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: propose the interview panel. Fields: "
    "`panelists` (list of role/name strings, e.g. 'Sam — CTO'). If the "
    "memory context names specific panelists, use exactly those names and "
    "no others — do not add a fifth, do not drop one. "
    "`total_hours` (float — use the exact value the memory context "
    "prescribes), "
    "`cto_included` (bool — true iff the CTO is on your panelists list)."
)

_SCREEN_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: draft a screening-call invite email. Fields: "
    "`subject` (email subject line), "
    "`body` (2–4 sentences — if the memory context says to mention a "
    "specific product or brand name, include it verbatim), "
    "`disclosure_questions` (ordered list of questions). If a memory rule "
    "says a particular question must come FIRST, place it at index 0. If a "
    "rule says the question must contain a specific phrase (e.g. "
    "'non-compete'), reproduce that phrase literally in the question text."
)


_MEMORY_SYSTEM_PROMPT = (
    "Return every Ledgerline rule whose trigger matches the candidate described in "
    "the query. Include the full rule_id, the trigger, and the action verbatim. "
    "Skip rules whose trigger clearly does not apply."
)


@cognee.agent_memory(
    with_memory=WITH_MEMORY,
    dataset_name=DATASET,
    memory_query_from_method="candidate_summary",
    memory_top_k=MEMORY_TOP_K,
    memory_system_prompt=_MEMORY_SYSTEM_PROMPT,
    save_session_traces=True,
    session_id=SESSION_ID,
)
async def propose_interview_format(candidate_summary: str) -> InterviewFormatProposal:
    return await LLMGateway.acreate_structured_output(
        text_input=candidate_summary,
        system_prompt=_FORMAT_PROMPT,
        response_model=InterviewFormatProposal,
    )


@cognee.agent_memory(
    with_memory=WITH_MEMORY,
    dataset_name=DATASET,
    memory_query_from_method="candidate_summary",
    memory_top_k=MEMORY_TOP_K,
    memory_system_prompt=_MEMORY_SYSTEM_PROMPT,
    save_session_traces=True,
    session_id=SESSION_ID,
)
async def schedule_panel(candidate_summary: str) -> PanelSchedule:
    return await LLMGateway.acreate_structured_output(
        text_input=candidate_summary,
        system_prompt=_PANEL_PROMPT,
        response_model=PanelSchedule,
    )


@cognee.agent_memory(
    with_memory=WITH_MEMORY,
    dataset_name=DATASET,
    memory_query_from_method="candidate_summary",
    memory_top_k=MEMORY_TOP_K,
    memory_system_prompt=_MEMORY_SYSTEM_PROMPT,
    save_session_traces=True,
    session_id=SESSION_ID,
)
async def compose_screen_invite(candidate_summary: str) -> ScreenInvite:
    return await LLMGateway.acreate_structured_output(
        text_input=candidate_summary,
        system_prompt=_SCREEN_PROMPT,
        response_model=ScreenInvite,
    )


def format_candidate(candidate: dict) -> str:
    """Render a candidate record as the retrieval query + LLM input text."""
    lines = [f"{k}: {v}" for k, v in candidate.items()]
    return "Candidate profile:\n" + "\n".join(lines)
