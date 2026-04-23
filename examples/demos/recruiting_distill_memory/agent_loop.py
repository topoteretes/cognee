"""Minimal planner-driven agentic loop.

A planner LLM sees the candidate + the tools already called + their return
values, and picks the next tool or says 'done'. We execute the chosen tool
(each tool is wrapped with @cognee.agent_memory, so the decorator handles
memory retrieval + trace persistence transparently) and feed the result
back into the next planner turn.

This keeps the loop genuinely agent-driven — the LLM chooses order and
termination — while staying small enough to read in one sitting. Three
tools plus a 'done' option means the worst case is four planner calls.
"""

from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway

from examples.demos.recruiting_distill_memory.agent_tools import (
    compose_screen_invite,
    format_candidate,
    propose_interview_format,
    schedule_panel,
)


ToolName = Literal[
    "propose_interview_format",
    "schedule_panel",
    "compose_screen_invite",
    "done",
]


_TOOLS: dict[str, Callable[..., Awaitable[Any]]] = {
    "propose_interview_format": propose_interview_format,
    "schedule_panel": schedule_panel,
    "compose_screen_invite": compose_screen_invite,
}

_TOOL_DESCRIPTIONS = {
    "propose_interview_format": "Decide the interview format, duration, and medium.",
    "schedule_panel": "Propose the interview panel (panelists, total hours, CTO inclusion).",
    "compose_screen_invite": "Draft the screening-call invite email and disclosure questions.",
    "done": "Return this when every part of the interview plan has been drafted.",
}

MAX_STEPS = 8


class PlannerDecision(BaseModel):
    """One planner turn: which tool to call next (or stop)."""

    reasoning: str = Field(
        description="One sentence: why this tool next given what has been done.",
    )
    next_tool: ToolName


_PLANNER_SYSTEM_PROMPT = (
    "You are the planner for a recruiter agent drafting an interview plan for a "
    "Staff Backend Engineer. You coordinate three tools: "
    + "; ".join(f"`{name}` — {desc}" for name, desc in _TOOL_DESCRIPTIONS.items())
    + ". Call each relevant tool exactly once, in the order you judge best. "
    "Return `next_tool='done'` only when the plan is complete. Never pick a tool "
    "that has already been called."
)


def _render_history(results: dict[str, Any]) -> str:
    if not results:
        return "No tools have been called yet."
    lines = ["Tools already called:"]
    for name, result in results.items():
        payload = result.model_dump() if hasattr(result, "model_dump") else repr(result)
        lines.append(f"- {name} → {payload}")
    return "\n".join(lines)


async def _pick_next_tool(summary: str, results: dict[str, Any]) -> PlannerDecision:
    user_input = f"Candidate:\n{summary}\n\n{_render_history(results)}"
    return await LLMGateway.acreate_structured_output(
        text_input=user_input,
        system_prompt=_PLANNER_SYSTEM_PROMPT,
        response_model=PlannerDecision,
    )


async def run_agentic_plan(candidate: dict) -> tuple[dict[str, Any], list[PlannerDecision]]:
    """Run the planner loop until it says 'done' or we hit MAX_STEPS.

    Returns the collected tool results keyed by tool name, plus the planner's
    decision trail so the runner can log it.
    """
    summary = format_candidate(candidate)
    results: dict[str, Any] = {}
    decisions: list[PlannerDecision] = []

    for step in range(MAX_STEPS):
        decision = await _pick_next_tool(summary, results)
        decisions.append(decision)
        print(f"  planner[{step}] → {decision.next_tool}  ({decision.reasoning})")

        if decision.next_tool == "done":
            break
        if decision.next_tool in results:
            # Planner tried to re-call a tool — skip, nudge toward 'done' next turn.
            continue

        tool = _TOOLS[decision.next_tool]
        results[decision.next_tool] = await tool(candidate_summary=summary)

    return results, decisions
