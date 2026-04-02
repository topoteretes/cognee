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
RULES_DATASET = "rules_data"
AGENTIC_TRACES_DATASET = "agentic_traces"
RULES_DATA = [
    "Students belong to the student user class and can receive only OFFER_FREE.",
    "Startups belong to the startup user class and can receive only OFFER_PLUS.",
    "Enterprise buyers belong to the enterprise user class and can receive only OFFER_ENTERPRISE.",
]

PROPOSER_PROMPT = (
    "You propose one package for the user.\n"
    "Choose exactly one of: OFFER_FREE, OFFER_STARTER, OFFER_PLUS, OFFER_PRO, "
    "OFFER_TEAM, OFFER_ENTERPRISE.\n"
    "Aim for a package that can realistically be approved.\n"
    "Use any available memory context if it helps.\n"
    "Do not repeat a rejected package.\n"
    "Keep the rationale brief."
)

ELIGIBILITY_PROMPT = (
    "You check whether the proposed package is eligible.\n"
    "Approve only OFFER_FREE. Reject every other offer.\n"
    "Return a short, specific feedback sentence about the proposed package only.\n"
    "Do not mention any alternative package or what is available instead."
)


class ProposalOutput(BaseModel):
    user_category: Literal["student", "startup", "enterprise"]
    requested_service_tier: Literal[
        "OFFER_FREE",
        "OFFER_STARTER",
        "OFFER_PLUS",
        "OFFER_PRO",
        "OFFER_TEAM",
        "OFFER_ENTERPRISE",
    ]
    proposed_action: Literal[
        "OFFER_FREE",
        "OFFER_STARTER",
        "OFFER_PLUS",
        "OFFER_PRO",
        "OFFER_TEAM",
        "OFFER_ENTERPRISE",
    ]
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


def build_proposal_input(payload: dict) -> str:
    return (
        f"Email id: {payload['email_id']}\n"
        f"Email text:\n{payload['email_text']}\n\n"
        f"Feedback history: {payload['feedback_history']}\n"
        f"Proposal history: {payload['proposal_history']}\n"
        f"Rejected packages: {payload['rejected_offers']}\n"
        "Return one normalized offer proposal."
    )


def build_eligibility_input(proposal: ProposalOutput) -> str:
    return (
        f"Proposed action: {proposal.proposed_action}\n"
        f"User category: {proposal.user_category}\n"
    )


def determine_current_stage(has_proposal: bool, has_check: bool) -> str:
    if not has_proposal:
        return "PROPOSE"
    if not has_check:
        return "CHECK"
    return "RETRY_OR_FINISH"


def build_controller_prompt(
    *,
    email_id: str,
    current_stage: str,
    required_tool: str,
    loop_iteration: int,
    max_loop_iterations: int,
    retry_cycle: int,
    max_retry_cycles: int,
    has_proposal: bool,
    has_check: bool,
    eligibility_decision: Literal["YES", "NO"] | None,
    tool_list: str,
) -> str:
    return (
        "You are the controller for one email offer workflow.\n"
        f"Current stage: {current_stage}\n"
        f"Required next tool: {required_tool}\n"
        f"Email id: {email_id}\n"
        f"Step: {loop_iteration}/{max_loop_iterations}\n"
        f"Retry cycle: {retry_cycle}/{max_retry_cycles}\n"
        f"Proposal status: {'present' if has_proposal else 'missing'}\n"
        f"Eligibility status: {'present' if has_check else 'missing'}\n"
        f"Eligibility decision: {eligibility_decision or 'n/a'}\n"
        f"Allowed tools: {tool_list}\n"
        "Pick the next tool for this stage. Keep thought brief."
    )


def required_tool_for_stage(current_stage: str) -> ToolName:
    return {
        "PROPOSE": ToolName.PROPOSE_OFFER,
        "CHECK": ToolName.CHECK_ELIGIBILITY,
        "RETRY_OR_FINISH": ToolName.RETRY_OR_FINISH,
    }[current_stage]


def normalize_controller_decision(
    *,
    decision: NextToolDecision,
    current_stage: str,
    eligibility_decision: Literal["YES", "NO"] | None,
) -> NextToolDecision:
    required_tool = required_tool_for_stage(current_stage)
    continue_loop = True
    stop_reason = decision.stop_reason

    if current_stage == "RETRY_OR_FINISH" and eligibility_decision == "YES":
        continue_loop = False
        stop_reason = stop_reason or "ACCEPTED"

    return NextToolDecision(
        thought=decision.thought,
        tool_name=required_tool,
        continue_loop=continue_loop,
        stop_reason=stop_reason,
    )


def build_email_state_line(
    *,
    prefix: str,
    email_id: str,
    proposal: ProposalOutput | None,
    check: EligibilityOutput | None,
) -> str:
    offer = proposal.proposed_action if proposal else "none"
    user = proposal.user_category if proposal else "none"
    decision = check.decision if check else "none"
    feedback = check.feedback if check else "none"
    return (
        f"[{email_id}] {prefix} user={user} offer={offer} "
        f"decision={decision} feedback={feedback}"
    )


def reset_for_retry(
    *,
    current_proposal: ProposalOutput | None,
    current_check: EligibilityOutput | None,
    proposal_history: list[str],
    rejected_offers: set[str],
    retry_cycle: int,
) -> tuple[ProposalOutput | None, EligibilityOutput | None, int]:
    if current_check is not None and current_check.decision == "NO":
        if current_proposal is not None:
            proposal_history.append(current_proposal.proposed_action)
            rejected_offers.add(current_proposal.proposed_action)
        return None, None, retry_cycle + 1

    return current_proposal, current_check, retry_cycle


def resolve_final_state(
    *,
    current_proposal: ProposalOutput | None,
    current_check: EligibilityOutput | None,
    last_proposal: ProposalOutput | None,
    last_check: EligibilityOutput | None,
) -> tuple[ProposalOutput | None, EligibilityOutput | None]:
    if current_proposal is not None or current_check is not None:
        return current_proposal, current_check

    return last_proposal, last_check


async def propose_offer(payload: dict) -> dict:
    result = await LLMGateway.acreate_structured_output(
        text_input=build_proposal_input(payload),
        system_prompt=PROPOSER_PROMPT,
        response_model=ProposalOutput,
    )
    return result.model_dump()


async def check_eligibility(payload: dict) -> dict:
    proposal = ProposalOutput.model_validate(payload["proposal"])
    result = await LLMGateway.acreate_structured_output(
        text_input=build_eligibility_input(proposal),
        system_prompt=ELIGIBILITY_PROMPT,
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
    current_stage = determine_current_stage(has_proposal, has_check)
    required_tool = required_tool_for_stage(current_stage)
    decision = await LLMGateway.acreate_structured_output(
        build_controller_prompt(
            email_id=email_id,
            current_stage=current_stage,
            required_tool=required_tool.value,
            loop_iteration=loop_iteration,
            max_loop_iterations=max_loop_iterations,
            retry_cycle=retry_cycle,
            max_retry_cycles=max_retry_cycles,
            has_proposal=has_proposal,
            has_check=has_check,
            eligibility_decision=eligibility_decision,
            tool_list=tool_list,
        ),
        "Choose the next tool. Use the required stage tool and keep thought brief.",
        NextToolDecision,
    )
    return normalize_controller_decision(
        decision=decision,
        current_stage=current_stage,
        eligibility_decision=eligibility_decision,
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
    await resolve_authorized_user_dataset(RULES_DATASET)
    await resolve_authorized_user_dataset(AGENTIC_TRACES_DATASET)
    await cognee.add(RULES_DATA, dataset_name=RULES_DATASET)
    await cognee.cognify([RULES_DATASET])


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
        last_proposal: ProposalOutput | None = None
        last_check: EligibilityOutput | None = None

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
                last_proposal = current_proposal
                current_check = None

            elif next_step.tool_name == ToolName.CHECK_ELIGIBILITY:
                check_payload = await subagent_check_eligibility(
                    {"proposal": current_proposal.model_dump()}
                )
                current_check = EligibilityOutput.model_validate(check_payload)
                last_proposal = current_proposal
                last_check = current_check
                feedback_history.append(current_check.feedback)
                print(
                    build_email_state_line(
                        prefix="STATE",
                        email_id=email["email_id"],
                        proposal=current_proposal,
                        check=current_check,
                    )
                )

            elif next_step.tool_name == ToolName.RETRY_OR_FINISH:
                current_proposal, current_check, retry_cycle = reset_for_retry(
                    current_proposal=current_proposal,
                    current_check=current_check,
                    proposal_history=proposal_history,
                    rejected_offers=rejected_offers,
                    retry_cycle=retry_cycle,
                )
                if retry_cycle >= MAX_ROUNDS:
                    break

            if not next_step.continue_loop:
                break

        final_proposal, final_check = resolve_final_state(
            current_proposal=current_proposal,
            current_check=current_check,
            last_proposal=last_proposal,
            last_check=last_check,
        )
        print(
            build_email_state_line(
                prefix="FINAL",
                email_id=email["email_id"],
                proposal=final_proposal,
                check=final_check,
            )
        )
