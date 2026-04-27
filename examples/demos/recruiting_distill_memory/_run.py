"""Shared runner: drive the agentic loop, then memify the traces.

Imported by run_naive.py / run_grounded.py *after* they set the env flags
(RECRUITING_WITH_MEMORY, RECRUITING_SESSION_ID) — the agent_tools module
reads those flags at its own import time.

Flow:
  1. Load candidate JSON.
  2. Run the planner-driven loop (agent_loop.run_agentic_plan) — the LLM
     chooses which tool to call next until it says 'done'. Each tool call
     goes through @cognee.agent_memory; the decorator retrieves rules from
     human_memory and persists a SessionManager trace entry per call with
     an LLM-generated one-sentence feedback summary.
  3. In grounded mode, invoke cognee's memify pipeline
     `persist_agent_trace_feedbacks_in_knowledge_graph` — this reads the
     stored session_feedback summaries from SessionManager, cognifies them,
     and adds the extracted entities/relationships to the graph on
     node_set=['agent_proposed_rule']. Those nodes are what the human
     reviews in review_pending_rules.py.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
    persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
)
from cognee.modules.users.methods import get_default_user

from examples.demos.recruiting_distill_memory.agent_loop import run_agentic_plan
from examples.demos.recruiting_distill_memory.agent_tools import (
    CANDIDATE,
    DATASET,
    SESSION_ID,
    WITH_MEMORY,
)


HERE = Path(__file__).parent
CANDIDATES_DIR = HERE / "data" / "candidates"
OUTPUT_DIR = HERE / "output"
PROPOSED_NODE_SET = "agent_proposed_rule"


async def run_plan() -> dict[str, Any]:
    candidate_path = CANDIDATES_DIR / f"{CANDIDATE}.json"
    candidate = json.loads(candidate_path.read_text())
    mode = "grounded" if WITH_MEMORY else "naive"
    output_filename = f"{mode}_plan_{CANDIDATE}.json"

    print(f"=== Run mode: {mode}  (session_id={SESSION_ID}) ===")
    print(f"Candidate: {candidate['name']} ({candidate['prior_company']} → {candidate['role']})\n")
    print("Running agentic loop ...")

    results, decisions = await run_agentic_plan(candidate)

    plan = {
        "mode": mode,
        "session_id": SESSION_ID,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate": candidate,
        "planner_decisions": [d.model_dump() for d in decisions],
        "interview_format": (
            results["propose_interview_format"].model_dump()
            if "propose_interview_format" in results else None
        ),
        "panel": (
            results["schedule_panel"].model_dump()
            if "schedule_panel" in results else None
        ),
        "screen_invite": (
            results["compose_screen_invite"].model_dump()
            if "compose_screen_invite" in results else None
        ),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / output_filename
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"\nWrote {out_path.relative_to(HERE.parent.parent.parent)}")

    if WITH_MEMORY:
        print(
            f"\nMemifying session traces → dataset '{DATASET}' "
            f"(node_set=['{PROPOSED_NODE_SET}']) ..."
        )
        user = await get_default_user()
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[SESSION_ID],
            dataset=DATASET,
            node_set_name=PROPOSED_NODE_SET,
            raw_trace_content=False,
        )
        print(
            "Done. Review newly-proposed nodes via "
            "`python -m examples.demos.recruiting_distill_memory.review_pending_rules`."
        )

    return plan
