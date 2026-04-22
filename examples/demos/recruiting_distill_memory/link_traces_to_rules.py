"""Deterministic trace → rule linker.

Reads session traces produced by run_grounded.py (via SessionManager), turns
each step into a typed AgentTraceStep graph node in the `agent_memory`
dataset, and attaches each step to the Rule nodes it cited in
`applied_rule_ids` — so the agent_memory graph has explicit cross-edges to
the human_memory rulebook and the path from action → rule is inspectable.

No LLM judging. Hallucinated rule IDs (not in seed_rules.yaml) are logged
and skipped; real citations get real edges.

Bucket tagging on each trace step:
  grounded_in_rule — at least one valid rule cited
  overrode_rule    — compliance_notes mentions "overrode" / "overrule"
  novel            — no rule cited (either proposed a new one or decided unaided)

Usage:
    python -m examples.demos.recruiting_distill_memory.link_traces_to_rules \
        --session-id recruiting-demo-grounded
"""

import argparse
import ast
import asyncio
import re
from pathlib import Path
from typing import Annotated, Any

import yaml

import cognee
from cognee.infrastructure.engine import DataPoint, Dedup, Embeddable
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage import add_data_points

from examples.demos.recruiting_distill_memory.rule_models import Rule


HERE = Path(__file__).parent
SEED_RULES = HERE / "data" / "seed_rules.yaml"
TRACE_DATASET = "agent_memory"
DEFAULT_SESSION = "recruiting-demo-grounded"


class AgentTraceStep(DataPoint):
    """One agent tool invocation, linked to the rules it cited."""

    trace_id: Annotated[str, Dedup("session_id + trace_id from SessionManager")]
    origin_function: str
    bucket: str
    compliance_notes: Annotated[str, Embeddable("one-sentence rationale for the action")]
    applied_rules: list[Rule] = []


def load_canonical_rules() -> dict[str, Rule]:
    """Load seed rules and index by rule_id so we can resolve deterministic UUIDs."""
    raw = yaml.safe_load(SEED_RULES.read_text())
    out: dict[str, Rule] = {}
    for record in raw["rules"]:
        rule = Rule.model_validate(record)
        # Must match ingest_human_memory.py's tagging — same Rule, same
        # Dedup key, so rebuilding this instance for edge targets yields
        # the identical UUID and merges onto the persisted node.
        rule.belongs_to_set = ["rule", rule.status, "human_authored"]
        out[rule.rule_id] = rule
    return out


def _bucket_for(applied_rule_ids: list[str], compliance_notes: str) -> str:
    notes = (compliance_notes or "").lower()
    if "overrode" in notes or "override" in notes:
        return "overrode_rule"
    if applied_rule_ids:
        return "grounded_in_rule"
    return "novel"


def _extract_list_literal(body: str, start: int) -> list[str]:
    """Find the `[...]` at `body[start:]` and ast.literal_eval it.

    The decorator serializes pydantic returns via str(model), which produces
    `field=value field=value ...`. For list[str] fields we can walk the
    brackets and literal_eval the slice.
    """
    if start >= len(body) or body[start] != "[":
        return []
    depth = 0
    for i in range(start, len(body)):
        ch = body[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return list(ast.literal_eval(body[start : i + 1]))
                except (ValueError, SyntaxError):
                    return []
    return []


def _parse_return_value(rv: Any) -> tuple[list[str], str]:
    """Return (applied_rule_ids, compliance_notes) from either a dict or a repr string.

    sanitize_value currently falls through to str(model) for pydantic outputs
    — we handle both shapes so this keeps working if that changes upstream.
    """
    if isinstance(rv, dict):
        return list(rv.get("applied_rule_ids") or []), rv.get("compliance_notes") or ""

    if not isinstance(rv, str):
        return [], ""

    match = re.search(r"applied_rule_ids=", rv)
    applied: list[str] = []
    if match:
        applied = _extract_list_literal(rv, match.end())

    notes_match = re.search(r"compliance_notes='((?:[^'\\]|\\.)*)'", rv)
    notes = notes_match.group(1) if notes_match else ""
    return applied, notes


def _make_link_task(session_id: str):
    """Return a pipeline task closed over `session_id`.

    We pass session_id via closure rather than through the pipeline data batch —
    run_custom_pipeline batches `data` into a list, and there's no extras kwarg
    for per-call scalars.
    """

    async def _link_task(
        _data: Any,
        ctx: PipelineContext = None,
    ) -> list[AgentTraceStep]:
        """Read traces for `session_id`, build AgentTraceStep nodes, persist them.

        Runs inside a custom pipeline so the per-user ACL storage scope is active
        — the SessionManager reads from the same cache dir run_grounded.py wrote to.
        """
        canonical = load_canonical_rules()
        print(f"Loaded {len(canonical)} canonical rules from seed_rules.yaml")

        user = ctx.user
        session_manager = get_session_manager()
        entries = await session_manager.get_agent_trace_session(
            user_id=str(user.id),
            session_id=session_id,
        )
        if not entries:
            raise SystemExit(
                f"No traces found for session {session_id!r}. Run run_grounded.py first."
            )
        print(f"Loaded {len(entries)} trace entries for session {session_id!r}\n")

        steps: list[AgentTraceStep] = []
        hallucinated: list[tuple[str, str]] = []  # (origin_function, bad_id)

        for entry in entries:
            origin = entry.get("origin_function", "unknown")
            raw_ids, notes = _parse_return_value(entry.get("method_return_value"))

            valid_ids: list[str] = []
            for rid in raw_ids:
                if rid in canonical:
                    valid_ids.append(rid)
                else:
                    hallucinated.append((origin, rid))

            bucket = _bucket_for(valid_ids, notes)
            step = AgentTraceStep(
                trace_id=f"{session_id}:{entry.get('trace_id', origin)}",
                origin_function=origin,
                bucket=bucket,
                compliance_notes=notes or "(no notes)",
                applied_rules=[canonical[rid] for rid in valid_ids],
            )
            step.belongs_to_set = ["trace_step", bucket]
            steps.append(step)

            print(
                f"  {origin:<30} bucket={bucket:<18} "
                f"applied={valid_ids or '[]'}  (raw: {raw_ids or '[]'})"
            )

        if hallucinated:
            print("\nSkipped hallucinated rule IDs (not in rulebook):")
            for origin, bad in hallucinated:
                print(f"  - {origin}: {bad!r}")

        print(f"\nPersisting {len(steps)} trace steps → dataset {TRACE_DATASET!r} ...")
        await add_data_points(steps, ctx=ctx)
        return steps

    return _link_task


async def main(session_id: str) -> None:
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_make_link_task(session_id))],
        data=[None],
        dataset=TRACE_DATASET,
        user=user,
        pipeline_name="link_traces_to_rules",
    )
    print("Done. AgentTraceStep nodes + edges to Rule nodes are now in the graph.")
    print(
        "Open the Cognee UI (cognee-cli -ui) to see the 'agent_memory' and "
        "'human_memory' graphs with cross-edges."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-id",
        default=DEFAULT_SESSION,
        help=f"Session id to link (default: {DEFAULT_SESSION!r})",
    )
    args = parser.parse_args()
    asyncio.run(main(session_id=args.session_id))
