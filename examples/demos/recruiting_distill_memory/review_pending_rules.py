"""Human review CLI for agent-proposed rules.

Reads `output/grounded_plan.json` (the JSON is cleaner than parsing the
truncated repr-string traces), surfaces each `proposed_new_rules` entry
with its origin_function context, and prompts
  [a]pprove / [r]eject / [s]kip
per proposal. Approved proposals become `Rule` DataPoints with
`status='approved'`, `source='agent_proposal:<origin>'` and are ingested
into the `human_memory` dataset — merging cleanly into the rulebook so
subsequent grounded runs can retrieve them.

Rejected proposals are logged; skipped ones leave nothing behind.

If the plan has no proposals, the CLI prints a message and exits — the
clean "Dev Rao" demo case has zero proposals, which is itself a useful
signal (grounded retrieval was sufficient, nothing novel to codify).

Usage:
    python -m examples.demos.recruiting_distill_memory.review_pending_rules
    python -m examples.demos.recruiting_distill_memory.review_pending_rules --auto-approve
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import cognee
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage import add_data_points

from examples.demos.recruiting_distill_memory.rule_models import ProposedRule, Rule


HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
HUMAN_MEMORY_DATASET = "human_memory"


def _collect_proposals(plan_path: Path) -> list[tuple[str, ProposedRule]]:
    """Return (origin_function, proposal) pairs for every proposal in the plan."""
    plan = json.loads(plan_path.read_text())
    found: list[tuple[str, ProposedRule]] = []
    for section in ("interview_format", "panel", "screen_invite"):
        section_data = plan.get(section) or {}
        for raw in section_data.get("proposed_new_rules") or []:
            found.append((section, ProposedRule.model_validate(raw)))
    return found


def _prompt_decision(origin: str, proposal: ProposedRule) -> str:
    print(f"\n--- Proposal from {origin} ---")
    print(f"  rule_id:    {proposal.rule_id}")
    print(f"  domain:     {proposal.domain}")
    print(f"  trigger:    {proposal.trigger}")
    print(f"  action:     {proposal.action}")
    print(f"  rationale:  {proposal.rationale}")
    while True:
        choice = input("[a]pprove / [r]eject / [s]kip? ").strip().lower()
        if choice in {"a", "r", "s"}:
            return choice
        print("  (please answer 'a', 'r', or 's')")


def _promote(origin: str, proposal: ProposedRule) -> Rule:
    rule = Rule(
        rule_id=proposal.rule_id,
        domain=proposal.domain,
        status="approved",
        source=f"agent_proposal:{origin}",
        trigger=proposal.trigger,
        action=proposal.action,
        rationale=proposal.rationale,
    )
    rule.belongs_to_set = ["rule", rule.status]
    return rule


def _make_ingest_task(approved: list[Rule]):
    """Close over the approved rules so the pipeline task just persists them."""

    async def _ingest(_data: Any, ctx: PipelineContext = None) -> list[Rule]:
        await add_data_points(approved, ctx=ctx)
        return approved

    return _ingest


async def _ingest_approved(approved: list[Rule]) -> None:
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_make_ingest_task(approved))],
        data=[None],
        dataset=HUMAN_MEMORY_DATASET,
        user=user,
        pipeline_name="review_pending_rules",
    )


async def main(plan_name: str, auto_approve: bool) -> None:
    plan_path = OUTPUT_DIR / plan_name
    if not plan_path.exists():
        raise SystemExit(
            f"Plan file not found: {plan_path}. Run run_grounded.py first."
        )

    proposals = _collect_proposals(plan_path)
    if not proposals:
        print(
            f"No proposed_new_rules in {plan_name}. "
            "Grounded retrieval covered every decision — nothing to review."
        )
        return

    print(f"Found {len(proposals)} proposal(s) in {plan_name}\n")

    approved: list[Rule] = []
    rejected: list[tuple[str, ProposedRule]] = []
    skipped: list[tuple[str, ProposedRule]] = []

    for origin, proposal in proposals:
        decision = "a" if auto_approve else _prompt_decision(origin, proposal)
        if decision == "a":
            approved.append(_promote(origin, proposal))
            print(f"  → approved {proposal.rule_id}")
        elif decision == "r":
            rejected.append((origin, proposal))
            print(f"  → rejected {proposal.rule_id}")
        else:
            skipped.append((origin, proposal))
            print(f"  → skipped {proposal.rule_id}")

    print()
    print(f"Summary: {len(approved)} approved, {len(rejected)} rejected, {len(skipped)} skipped")

    if not approved:
        print("No rules approved. human_memory unchanged.")
        return

    print(
        f"\nIngesting {len(approved)} approved rule(s) "
        f"into dataset {HUMAN_MEMORY_DATASET!r} ..."
    )
    await _ingest_approved(approved)
    print("Done. The rulebook now includes the newly approved rules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="grounded_plan.json",
        help="Plan JSON filename in ./output/ (default: grounded_plan.json)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve every proposal without prompting (useful for CI/demos)",
    )
    args = parser.parse_args()
    asyncio.run(main(plan_name=args.plan, auto_approve=args.auto_approve))
