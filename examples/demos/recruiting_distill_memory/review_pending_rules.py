"""Human review CLI for agent-proposed learnings.

After a grounded run, the memify pipeline
`persist_agent_trace_feedbacks_in_knowledge_graph` has cognified the
agent's session_feedback summaries into the `human_memory` graph under
node_set='agent_proposed_rule'. Those new nodes are entities /
summaries / chunks extracted by the standard cognify LLM pass.

This CLI:
  1. Finds every node tagged 'agent_proposed_rule' in the graph.
  2. Shows each one (type, name, description) to the human.
  3. On [a]pprove: asks an LLM to convert the node's content into a
     structured Rule (rule_id, domain, trigger, action, rationale)
     and ingests it into human_memory with
     belongs_to_set=['rule','approved','agent_authored'].
  4. On [r]eject / [s]kip: leaves the graph untouched.

The proposed-rule nodes are left in place regardless of the decision —
they're part of the agent_memory trail.

Usage:
    python -m examples.demos.recruiting_distill_memory.review_pending_rules
    python -m examples.demos.recruiting_distill_memory.review_pending_rules --auto-approve
"""

import argparse
import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage import add_data_points

from examples.demos.recruiting_distill_memory.rule_models import Domain, Rule


HUMAN_MEMORY_DATASET = "human_memory"
PROPOSED_NODE_SET = "agent_proposed_rule"


class _PromotedRule(BaseModel):
    """Structured Rule extracted from a graph node by the LLM."""

    rule_id: str = Field(description="Short id, e.g. 'R7_fintech_noncompete' — lowercase snake")
    domain: Domain
    trigger: str = Field(description="Concrete condition under which the rule fires")
    action: str = Field(description="What the recruiter should do when it fires")
    rationale: str = Field(description="One-sentence why")


_PROMOTION_PROMPT_TEMPLATE = (
    "You are distilling an AI agent's captured observation into a reusable "
    "recruiting rule for Ledgerline. Given the node description/text below, "
    "emit a structured rule a future recruiter can apply mechanically. Keep "
    "the trigger specific enough to be actionable.\n\n"
    "CRITICAL — rule_id uniqueness. The following rule_ids already exist and "
    "MUST NOT be reused: {existing_ids}. Pick a fresh id of the form "
    "`R<N>_<slug>` where N is strictly greater than the largest existing R<N>."
)

MIN_NODE_TEXT_CHARS = 30


def _contains_node_set(value: Any, target: str) -> bool:
    """`belongs_to_set` is sometimes list[str], sometimes list[dict], sometimes a str."""
    if not value:
        return False
    if isinstance(value, str):
        return target in value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item == target:
                return True
            if isinstance(item, dict) and item.get("name") == target:
                return True
    return False


def _node_text(props: dict) -> str:
    parts = []
    for key in ("name", "description", "text", "content"):
        val = props.get(key)
        if val and isinstance(val, str):
            parts.append(f"{key}: {val.strip()}")
    return "\n".join(parts) or "(no text fields)"


def _substantive_length(props: dict) -> int:
    """Characters across description/text/content — excludes the usually-short `name`."""
    total = 0
    for key in ("description", "text", "content"):
        val = props.get(key)
        if val and isinstance(val, str):
            total += len(val.strip())
    return total


def _prompt_decision(node_id: str, props: dict) -> str:
    print(f"\n--- Proposed node {node_id} ---")
    print(f"  type: {props.get('type', '?')}")
    print(_node_text(props))
    while True:
        choice = input("[a]pprove / [r]eject / [s]kip? ").strip().lower()
        if choice in {"a", "r", "s"}:
            return choice
        print("  (please answer 'a', 'r', or 's')")


async def _promote_via_llm(
    node_id: str, props: dict, existing_ids: set[str]
) -> Optional[Rule]:
    text = _node_text(props)
    prompt = _PROMOTION_PROMPT_TEMPLATE.format(existing_ids=sorted(existing_ids))
    try:
        promoted = await LLMGateway.acreate_structured_output(
            text_input=f"Node id: {node_id}\n{text}",
            system_prompt=prompt,
            response_model=_PromotedRule,
        )
    except Exception as error:
        print(f"  (LLM promotion failed: {error}) — skipping")
        return None

    # The LLM occasionally collides anyway — append a numeric suffix until unique.
    rule_id = promoted.rule_id
    suffix = 2
    while rule_id in existing_ids:
        rule_id = f"{promoted.rule_id}_{suffix}"
        suffix += 1

    rule = Rule(
        rule_id=rule_id,
        domain=promoted.domain,
        status="approved",
        source=f"agent_proposal:{node_id}",
        trigger=promoted.trigger,
        action=promoted.action,
        rationale=promoted.rationale,
    )
    rule.belongs_to_set = ["rule", "approved", "agent_authored"]
    return rule


def _make_review_task(auto_approve: bool):
    """Return a pipeline task closed over the auto_approve flag."""

    async def _review(_data: Any, ctx: PipelineContext = None) -> list[Rule]:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph = await get_graph_engine()
        nodes, _edges = await graph.get_graph_data()

        # Collect existing rule_ids so _promote_via_llm can warn the LLM
        # and the collision guard can rename any duplicates.
        existing_rule_ids: set[str] = set()
        for node in nodes:
            if not isinstance(node, tuple):
                continue
            _nid, props = node
            if props and props.get("rule_id"):
                existing_rule_ids.add(str(props["rule_id"]))

        proposed: list[tuple[str, dict]] = []
        skipped_for_noise = 0
        for node in nodes:
            if not isinstance(node, tuple):
                continue
            node_id, props = node
            if not props:
                continue
            if not _contains_node_set(props.get("belongs_to_set"), PROPOSED_NODE_SET):
                continue
            if _substantive_length(props) < MIN_NODE_TEXT_CHARS:
                skipped_for_noise += 1
                continue
            proposed.append((str(node_id), props))

        if skipped_for_noise:
            print(
                f"Filtered {skipped_for_noise} node(s) with too little text "
                f"(< {MIN_NODE_TEXT_CHARS} chars) from review."
            )

        if not proposed:
            print(f"No reviewable nodes tagged '{PROPOSED_NODE_SET}'. Run run_grounded.py first.")
            return []

        print(f"Found {len(proposed)} proposed node(s) tagged '{PROPOSED_NODE_SET}'.\n")

        approved_rules: list[Rule] = []
        for node_id, props in proposed:
            decision = "a" if auto_approve else _prompt_decision(node_id, props)
            if decision == "a":
                rule = await _promote_via_llm(node_id, props, existing_rule_ids)
                if rule:
                    existing_rule_ids.add(rule.rule_id)
                    approved_rules.append(rule)
                    print(f"  → approved as {rule.rule_id} (domain={rule.domain})")
            elif decision == "r":
                print(f"  → rejected {node_id}")
            else:
                print(f"  → skipped {node_id}")

        if approved_rules:
            print(
                f"\nIngesting {len(approved_rules)} approved rule(s) "
                f"into '{HUMAN_MEMORY_DATASET}' ..."
            )
            await add_data_points(approved_rules, ctx=ctx)
            print("Done. The rulebook now includes the newly approved rules.")
        else:
            print("\nNo rules approved. human_memory unchanged.")

        return approved_rules

    return _review


async def main(auto_approve: bool) -> None:
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_make_review_task(auto_approve))],
        data=[None],
        dataset=HUMAN_MEMORY_DATASET,
        user=user,
        pipeline_name="review_pending_rules",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve every proposal without prompting (useful for CI/demos)",
    )
    args = parser.parse_args()
    asyncio.run(main(auto_approve=args.auto_approve))
