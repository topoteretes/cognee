"""Shared logic for the memory-vs-no-memory agent demo."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Literal, Optional

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.engine.operations.setup import setup
from cognee.modules.pipelines.layers.resolve_authorized_user_dataset import (
    resolve_authorized_user_dataset,
)
from pydantic import BaseModel, Field


DATA_DIR = Path(__file__).resolve().parent / "data"
EMAILS_FILE = DATA_DIR / "emails_stream.jsonl"
MAX_ROUNDS = 8
MAX_LOOP_ITERATIONS = 32


class ProposalOutput(BaseModel):
    location: str = Field(min_length=1)
    user_category: str = Field(min_length=1)
    requested_service_tier: str = Field(min_length=1)
    proposed_action: str = Field(pattern=r"^OFFER_(FREE|STARTER|PLUS|PRO|TEAM|ENTERPRISE)$")
    rationale: str = Field(min_length=1, max_length=120)


class EligibilityOutput(BaseModel):
    decision: Literal["YES", "NO"]
    feedback: str


RootFn = Callable[[dict], Awaitable[dict]]


class ToolName(str, Enum):
    PROPOSE_OFFER = "ProposeOffer"
    CHECK_ELIGIBILITY = "CheckEligibility"
    RETRY_OR_FINISH = "RetryOrFinish"


class NextToolDecision(BaseModel):
    """Controller output for the next tool in the loop."""

    thought: str = Field(min_length=1)
    tool_name: ToolName
    continue_loop: bool = True
    stop_reason: Optional[str] = None


async def propose_offer(payload: dict) -> dict:
    result = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Email id: {payload['email_id']}\n"
            f"Email text:\n{payload['email_text']}\n\n"
            f"Feedback history: {payload['feedback_history']}\n"
            f"Proposal history: {payload['proposal_history']}\n"
            f"Rejected packages: {payload['rejected_offers']}\n"
            "Return one normalized offer proposal."
        ),
        system_prompt=(
            "You are Agent A (offer proposer).\n"
            "Propose exactly one package for the user: OFFER_FREE, OFFER_STARTER, "
            "OFFER_PLUS, OFFER_PRO, OFFER_TEAM, or OFFER_ENTERPRISE.\n"
            "If you have access to memory related information use it to make a decision "
            "which package to offer.\n"
            "Never propose a package that already appears in Rejected packages.\n"
            "Rationale must be short (one sentence)."
        ),
        response_model=ProposalOutput,
    )
    return result.model_dump()


async def check_eligibility(payload: dict) -> dict:
    proposal = ProposalOutput.model_validate(payload["proposal"])
    result = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Proposed action: {proposal.proposed_action}\n"
            f"User category: {proposal.user_category}\n"
            f"Location: {proposal.location}\n"
        ),
        system_prompt=(
            "You are Agent B (eligibility checker).\n"
            "Policy for this demo:\n"
            "- YES only if proposed action is OFFER_FREE.\n"
            "- NO for every other proposed action.\n"
            "Return structured output only.\n"
            "Feedback must be an original one-sentence explanation for this specific "
            "proposal. Do not mention what is available if it is not proposed.\n"
            "Do not use fixed canned phrases."
        ),
        response_model=EligibilityOutput,
    )
    return result.model_dump()


async def controller_decide_tool(
    *,
    email_id: str,
    loop_iteration: int,
    max_loop_iterations: int,
    has_proposal: bool,
    has_check: bool,
    eligibility_decision: Literal["YES", "NO"] | None,
    retry_cycle: int,
    max_retry_cycles: int,
) -> NextToolDecision:
    available_tools = list(ToolName)
    tool_list = ", ".join(tool.value for tool in available_tools)
    current_stage = "PROPOSE" if not has_proposal else ("CHECK" if not has_check else "RETRY_OR_FINISH")
    required_tool = {
        "PROPOSE": ToolName.PROPOSE_OFFER.value,
        "CHECK": ToolName.CHECK_ELIGIBILITY.value,
        "RETRY_OR_FINISH": ToolName.RETRY_OR_FINISH.value,
    }[current_stage]
    prompt = (
        "You are the controller for ONE email offer workflow.\n\n"
        "STRICT TRANSITION POLICY (no exceptions):\n"
        "Stage PROPOSE -> ONLY ProposeOffer is valid.\n"
        "Stage CHECK -> ONLY CheckEligibility is valid.\n"
        "Stage RETRY_OR_FINISH -> ONLY RetryOrFinish is valid.\n\n"
        "Forbidden behavior:\n"
        "- Never call CheckEligibility before a proposal exists.\n"
        "- Never call ProposeOffer while a proposal exists and eligibility has not run.\n"
        "- Never call ProposeOffer or CheckEligibility after eligibility is present; use RetryOrFinish.\n\n"
        "continue_loop (only you may end the workflow):\n"
        "- Stages PROPOSE and CHECK: always continue_loop=true.\n"
        "- Stage RETRY_OR_FINISH: if Eligibility decision is YES, set continue_loop=false "
        "(stop_reason e.g. ACCEPTED). If NO, continue_loop=true.\n\n"
        "Execution goal for this step:\n"
        f"- Current stage requires EXACT tool: {required_tool}\n"
        "- Return that tool in tool_name.\n\n"
        "THOUGHT field quality requirements:\n"
        "- Write 1-2 short sentences.\n"
        "- Explain why this tool is correct now using proposal and eligibility status.\n"
        "- Mention what this action should produce next.\n\n"
        "Context:\n"
        f"Email id: {email_id}\n"
        f"Current stage: {current_stage}\n"
        f"Controller step: {loop_iteration}/{max_loop_iterations}\n"
        f"Retry cycle (NO outcomes so far): {retry_cycle}/{max_retry_cycles}\n"
        f"Proposal status: {'present' if has_proposal else 'missing'}\n"
        f"Eligibility status: {'present' if has_check else 'missing'}\n"
        f"Eligibility decision: {eligibility_decision or 'n/a'}\n"
        f"Allowed tools: {tool_list}\n"
    )
    return await LLMGateway.acreate_structured_output(
        prompt,
        (
            "Select exactly one tool. Follow strict stage policy and continue_loop rules. "
            "If your selected tool differs from the required stage tool, it is incorrect. "
            "In thought, explain why this tool is the right next action now and what output "
            "is expected."
        ),
        NextToolDecision,
    )


def load_emails() -> list[dict]:
    rows: list[dict] = []
    for line in EMAILS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


async def setup_runtime() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    await resolve_authorized_user_dataset('main_dataset')


async def run_stream_impl(
    *,
    subagent_propose_offer: RootFn,
    subagent_check_eligibility: RootFn,
) -> None:
    for email in load_emails():
        feedback_history: list[str] = []
        proposal_history: list[str] = []
        rejected_offers: set[str] = set()
        current_proposal: ProposalOutput | None = None
        current_check: EligibilityOutput | None = None

        loop_iteration = 0
        retry_cycle = 0
        while loop_iteration < MAX_LOOP_ITERATIONS:
            loop_iteration += 1
            next_step = await controller_decide_tool(
                email_id=email["email_id"],
                loop_iteration=loop_iteration,
                max_loop_iterations=MAX_LOOP_ITERATIONS,
                has_proposal=current_proposal is not None,
                has_check=current_check is not None,
                eligibility_decision=current_check.decision if current_check else None,
                retry_cycle=retry_cycle,
                max_retry_cycles=MAX_ROUNDS,
            )

            if next_step.tool_name == ToolName.PROPOSE_OFFER:
                proposal_payload = await subagent_propose_offer(
                    {
                        "email_id": email["email_id"],
                        "email_text": email["email_text"],
                        "feedback_history": feedback_history,
                        "proposal_history": proposal_history,
                        "rejected_offers": sorted(rejected_offers),
                    }
                )
                current_proposal = ProposalOutput.model_validate(proposal_payload)
                current_check = None

            elif next_step.tool_name == ToolName.CHECK_ELIGIBILITY:
                check_payload = await subagent_check_eligibility(
                    {"proposal": current_proposal.model_dump()}
                )
                current_check = EligibilityOutput.model_validate(check_payload)
                feedback_history.append(current_check.feedback)
                proposal = current_proposal
                offer = proposal.proposed_action if proposal else "none"
                user = proposal.user_category if proposal else "none"
                location = proposal.location if proposal else "none"
                email_id = email["email_id"]
                print(
                    f"[{email_id}] STATE user={user} location={location} "
                    f"offer={offer} decision={current_check.decision} "
                    f"feedback={current_check.feedback}"
                )

            elif next_step.tool_name == ToolName.RETRY_OR_FINISH:
                if current_check is not None and current_check.decision == "NO":
                    if current_proposal is not None:
                        proposal_history.append(current_proposal.proposed_action)
                        rejected_offers.add(current_proposal.proposed_action)
                    current_proposal = None
                    current_check = None
                    retry_cycle += 1

            if not next_step.continue_loop:
                break

        proposal = current_proposal
        final_offer = proposal.proposed_action if proposal else "none"
        final_decision = current_check.decision if current_check else "none"
        user = proposal.user_category if proposal else "none"
        location = proposal.location if proposal else "none"
        email_id = email["email_id"]
        final_feedback = current_check.feedback if current_check else "none"
        print(
            f"[{email_id}] FINAL user={user} location={location} {final_offer} - "
            f"{final_decision} feedback={final_feedback}"
        )
