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
    "(format: 'R1_live_coding'). "
    "\n\n"
    "IMPORTANT — proposing new rules. You are actively responsible for "
    "spotting gaps in the rulebook. Populate `proposed_new_rules` whenever "
    "ANY of these hold:\n"
    "  (a) A retrieved rule's trigger lists specific items (e.g. companies) "
    "      and the candidate has a similar-but-not-listed item — propose "
    "      extending the trigger to include it.\n"
    "  (b) The candidate has a recruiting-relevant attribute (location/visa, "
    "      seniority, counter-offer posture, tech stack, relocation, clearance) "
    "      that no retrieved rule addresses and that a future candidate will "
    "      plausibly share.\n"
    "  (c) You made a judgment call that a future recruiter should repeat — "
    "      bottle it as a rule.\n"
    "Each proposal needs rule_id='proposed_R<N>_<slug>' (e.g. "
    "'proposed_R7_fintech_noncompete'), a domain, a concrete trigger, a "
    "concrete action, and a rationale. A human reviews every proposal before "
    "it joins the rulebook — so err on the side of proposing; overproposing "
    "is cheap, under-proposing is expensive.\n\n"
    "Set `compliance_notes` to one of: 'followed-as-is' (pure rule application), "
    "'extended' (applied a rule but also proposed an extension), 'novel' (no "
    "rule applied, proposed one or decided unaided), 'overrode' (had to deviate "
    "from a rule — explain why).\n\n"
    "If no memory context is available, decide from general best practice, "
    "leave `applied_rule_ids` empty, and set compliance_notes='novel'. "
    "Never invent rule IDs for `applied_rule_ids` — only cite IDs that appear "
    "in the memory context."
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
