"""Three @cognee.agent_memory-wrapped tools for Staff Backend interview scheduling.

Each tool takes a plain-text `candidate_summary` so the decorator can use it
verbatim as the retrieval query (`memory_query_from_method="candidate_summary"`),
reads relevant rules from `human_memory` when `with_memory=True`, asks the LLM
for a pydantic-structured plan, and returns `applied_rule_ids` +
`proposed_new_rules` on the output so the trace → rule linker doesn't need
any LLM judging.

Env vars read at import time:
  RECRUITING_WITH_MEMORY   "true" (default) / "false" — flips naive vs grounded
  RECRUITING_SESSION_ID    session id for trace persistence (default: "recruiting-demo")
  RECRUITING_MEMORY_TOP_K  retrieval breadth (default: "6")

run_naive.py / run_grounded.py set these before importing this module.
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
SESSION_ID = os.environ.get("RECRUITING_SESSION_ID", "recruiting-demo").strip()
MEMORY_TOP_K = int(os.environ.get("RECRUITING_MEMORY_TOP_K", "6"))


_ALEX_PERSONA = (
    "You are Alex Chen, a senior recruiter at Ledgerline (a 40-person payments "
    "fintech). You are drafting decisions for a Staff Backend Engineer hire. "
    "If an 'Additional Memory Context' block is present above the user input, "
    "it contains Ledgerline-specific rules. Follow every rule whose trigger "
    "matches this candidate; list those rule IDs in `applied_rule_ids` "
    "(format: 'R1_live_coding'). If the candidate exposes a gap no listed rule "
    "covers, suggest a new rule via `proposed_new_rules` (coin an id prefixed "
    "with 'proposed_'). If no memory context is available, decide from general "
    "best practice and leave `applied_rule_ids` empty — never invent rule IDs."
)


_FORMAT_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: choose the interview format. Fields: "
    "`format` ∈ {live_coding, take_home, system_design, behavioral_only}, "
    "`duration_minutes` (int), `medium` ∈ {video, onsite, phone}."
)

_PANEL_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: propose the interview panel. Fields: "
    "`panelists` (list of role/name strings, e.g. 'Sam — CTO'), "
    "`total_hours` (float), `cto_included` (bool — must match whether Sam/CTO is on the panel)."
)

_SCREEN_PROMPT = (
    _ALEX_PERSONA
    + "\n\nTask: draft a screening-call invite email. Fields: "
    "`subject` (email subject line), `body` (2–4 sentences), "
    "`disclosure_questions` (list of questions the recruiter should ask on the screen)."
)


_MEMORY_SYSTEM_PROMPT = (
    "Return every Ledgerline rule whose trigger matches the candidate described in "
    "the query. CRITICAL: every rule has a rule_id string of the form "
    "'R<number>_<slug>' (e.g. R1_live_coding, R2_panel_footprint, R4_noncompete_screen). "
    "Always quote the rule_id EXACTLY as stored — never shorten to just the number. "
    "For each match, include the full rule_id, the trigger, and the action verbatim. "
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
