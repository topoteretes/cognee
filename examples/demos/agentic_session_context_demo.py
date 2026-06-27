"""Agentic session-context demo.

The agentic counterpart to ``live_session_context_feedback_demo.py``. Where the QA demo learns
lessons from conversation feedback, this one learns ``agent``-profile lessons from an agent's
tool/action traces — and, like that demo, the real lesson extraction is LLM-backed.

Run (full, requires a configured LLM provider):

    uv run python examples/demos/agentic_session_context_demo.py

Run a reduced, no-LLM version (deterministic failure capture + recall only):

    uv run python examples/demos/agentic_session_context_demo.py --offline

What it shows:

  1. Processing agentic traces
     Store five traces one by one. For demo readability only, the periodic extractor uses
     interval=2 and overlap=1, so automatic LLM extraction triggers after traces 2 and 4.
     The demo prints session memory after every trace.
  2. Distillation
     Run final batch context extraction for the pending tail, then distill_session() rewrites
     accepted guidance into markdown documents and add+cognifies those documents into the graph.
  3. Recall
     recall(scope=["session_context"], context_profile="agent") returns the active guidance
     read-only; context_profile="qa" returns nothing because this demo wrote no QA-profile
     entries; scope=["trace"] still returns the raw evidence.

Mental model:

  raw traces -> active agent guidance -> recall prompt context
                     |
                     +-> distillation -> session_learnings graph documents

Exact lesson wording varies by model.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

os.environ["CACHING"] = "true"
os.environ["CACHE_BACKEND"] = "fs"
os.environ["AUTO_FEEDBACK"] = "true"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee
from cognee.memory import TraceEntry
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.methods import get_default_user

DATASET_NAME = "agentic_session_context_demo"
SESSION_ID = "agentic_demo_session"
OUTPUT_PATH = "agentic_session_context_demo_output.json"
DEMO_TRACE_EXTRACTION_INTERVAL = 2
DEMO_TRACE_EXTRACTION_OVERLAP = 1

# One agent working a task. The full demo uses a tiny interval so five traces show:
# live capture, automatic periodic LLM extraction, a pending tail, and final flush.
TRACES = [
    {
        "origin_function": "run_tests",
        "status": "error",
        "method_params": {"command": "pytest -q"},
        "error_message": "ModuleNotFoundError: No module named 'dotenv'",
    },
    {
        "origin_function": "run_command",
        "status": "success",
        "method_params": {"command": "uv sync"},
        "method_return_value": "Installed 42 packages in 1.2s",
    },
    {
        "origin_function": "run_tests",
        "status": "success",
        "method_params": {"command": "uv run pytest -q"},
        "method_return_value": "188 passed in 4.1s",
    },
    {
        "origin_function": "read_file",
        "status": "success",
        "method_params": {"path": "pyproject.toml"},
        "method_return_value": "[project] name = 'demo'  # managed with uv",
    },
    {
        "origin_function": "run_lint",
        "status": "error",
        "method_params": {"command": "uv run ruff check ."},
        "error_message": "F401 'os' imported but unused",
    },
]


def progress(message: str):
    print(f"[agentic-demo] {message}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip LLM extraction/distillation; show deterministic trace capture and recall.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    await setup_demo_data()
    user = await get_default_user()
    await reset_demo_session(user)

    output = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "mode": "offline" if args.offline else "full",
    }
    if not args.offline:
        output["demo_extraction_window"] = {
            "min_new_traces": DEMO_TRACE_EXTRACTION_INTERVAL,
            "overlap": DEMO_TRACE_EXTRACTION_OVERLAP,
        }

    if args.offline:
        processing_explanation = (
            "Store five traces one by one. Offline mode skips LLM extraction, so only "
            "deterministic live failure context entries appear."
        )
    else:
        processing_explanation = (
            "Store five traces one by one. Session memory is printed after every trace; "
            "trace 2 and trace 4 trigger a periodic extraction pass."
        )
    print_act_header("1. Processing agentic traces", processing_explanation)
    output["act1_processing_agentic_traces"] = await process_agentic_traces(
        user, drive_periodic_extraction=not args.offline
    )

    if not args.offline:
        print_act_header(
            "2. Distillation",
            (
                "Run final batch context extraction for pending trace 5, then distill the "
                "resulting session-memory entries into graph documents."
            ),
        )
        output["act2_distillation"] = await run_distillation_act(user)
        print_distillation(output["act2_distillation"])

    print_act_header(
        "3. Recall",
        (
            "Recall renders agent-profile session memory read-only. QA-profile context stays "
            "empty because this demo only wrote agent context."
        ),
    )
    output["act3_recall"] = await run_recall_act(user)
    print_recall(output["act3_recall"])

    await save_demo_output(output)


async def run_distillation_act(user) -> dict:
    """Flush the pending tail, then run distillation."""
    from cognee.infrastructure.session.agent_context_extraction import extract_pending_agent_context
    from cognee.modules.session_distillation import distill_session

    try:
        before_flush = await snapshot(user)
        touched_ids = await extract_pending_agent_context(
            session_manager=get_session_manager(),
            user_id=str(user.id),
            session_id=SESSION_ID,
            min_new_traces=1,
        )
        distillation_input = await snapshot(user)
        result = await distill_session(SESSION_ID, dataset=DATASET_NAME, user=user)
        return {
            "flush_touched_entry_ids": touched_ids,
            "before_flush": before_flush,
            "distillation_input": distillation_input,
            "distillation_status": result.status,
            "cognified_documents": result.documents,
        }
    except Exception as exc:
        progress(f"Act 2 failed: {exc}")
        return {
            "flush_touched_entry_ids": [],
            "before_flush": await snapshot(user),
            "distillation_input": await snapshot(user),
            "distillation_status": "failed",
            "cognified_documents": [],
            "error": str(exc),
        }


async def save_demo_output(output: dict):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(output, indent=2))
    print("", file=sys.stderr)
    print("--- Saved structured output ---", file=sys.stderr)
    print(OUTPUT_PATH, file=sys.stderr)


async def process_agentic_traces(user, *, drive_periodic_extraction: bool) -> dict:
    """Store the demo traces one by one and print session memory after each trace.

    When ``drive_periodic_extraction`` is set, call the periodic batch pass with a tiny interval
    after each trace. The trace-write path (``SessionManager.add_agent_trace_step``) already runs
    this for you every ``TRACE_EXTRACTION_INTERVAL`` traces; we drive it directly with a smaller
    interval so a five-trace demo shows the pass firing (after traces 2 and 4) instead of only at
    the final flush.
    """
    from cognee.infrastructure.session.agent_context_extraction import (
        extract_pending_agent_context,
    )

    trace_states = []
    for index, trace in enumerate(TRACES, start=1):
        before = await snapshot(user)
        await cognee.remember(
            TraceEntry(**trace),
            dataset_name=DATASET_NAME,
            session_id=SESSION_ID,
            self_improvement=False,
            user=user,
        )
        if drive_periodic_extraction:
            await extract_pending_agent_context(
                session_manager=get_session_manager(),
                user_id=str(user.id),
                session_id=SESSION_ID,
                min_new_traces=DEMO_TRACE_EXTRACTION_INTERVAL,
                overlap=DEMO_TRACE_EXTRACTION_OVERLAP,
            )
        after = await snapshot(user)
        trace_state = {
            "trace_number": index,
            "origin_function": trace["origin_function"],
            "status": trace["status"],
            "trace": trace,
            "watermark_before": before["processed_trace_count"],
            "watermark_after": after["processed_trace_count"],
            "automatic_batch_triggered": (
                after["processed_trace_count"] > before["processed_trace_count"]
            ),
            "snapshot": after,
        }
        trace_states.append(trace_state)
        print_trace_memory(trace_state)

    return {"trace_states": trace_states}


async def run_recall_act(user) -> dict:
    served_before = await served_state(user)

    agent_ctx = await cognee.recall(
        "what should I know before running tests in this repo?",
        scope=["session_context"],
        context_profile="agent",
        session_id=SESSION_ID,
        only_context=True,
        user=user,
    )
    qa_ctx = await cognee.recall(
        "what should I know before running tests in this repo?",
        scope=["session_context"],
        context_profile="qa",
        session_id=SESSION_ID,
        only_context=True,
        user=user,
    )
    raw_traces = await cognee.recall(
        "dotenv",
        scope=["trace"],
        session_id=SESSION_ID,
        only_context=True,
        user=user,
    )
    served_after = await served_state(user)

    result = {
        "agent_profile_block": block_text(agent_ctx),
        "qa_profile_result_count": len(qa_ctx),
        "trace_evidence_count": len(raw_traces),
        "recall_made_no_writes": served_before == served_after,
    }
    return result


# --------------------------------------------------------------------------- evidence


async def snapshot(user) -> dict:
    traces = await get_session_manager().get_agent_trace_session(
        user_id=str(user.id), session_id=SESSION_ID
    )
    processed_trace_count = await processed_trace_count_for(user)
    return {
        "trace_count": len(traces),
        "processed_trace_count": processed_trace_count,
        "pending_trace_count": max(0, len(traces) - processed_trace_count),
        "agent_lessons": await agent_lessons_for(user),
    }


async def processed_trace_count_for(user) -> int:
    from cognee.infrastructure.session.agent_context_extraction import (
        TRACE_EXTRACTION_STATE_ID,
        TRACE_EXTRACTION_STATE_KIND,
    )

    rows = await get_session_manager().get_session_context_entries(
        user_id=str(user.id), session_id=SESSION_ID
    )
    for row in rows:
        if (
            row.get("id") != TRACE_EXTRACTION_STATE_ID
            and row.get("kind") != TRACE_EXTRACTION_STATE_KIND
        ):
            continue
        try:
            return max(0, int(row.get("processed_trace_count") or 0))
        except (TypeError, ValueError):
            return 0
    return 0


async def agent_lessons_for(user) -> list[dict]:
    rows = await get_session_manager().get_session_context_entries(
        user_id=str(user.id), session_id=SESSION_ID
    )
    lessons = []
    for row in rows:
        if row.get("kind", "context") != "context":
            continue
        if row.get("context_profile", "qa") != "agent":
            continue
        lessons.append(
            {
                "section": row.get("section"),
                "content": row.get("content"),
                "confidence": row.get("confidence"),
                "source_trace_ids": row.get("source_trace_ids", []),
                "id": row.get("id"),
                "harmful_count": row.get("harmful_count", 0),
            }
        )
    return lessons


async def served_state(user) -> dict:
    """Map of agent-lesson id -> last_served_at, to prove recall does not stamp anything."""
    rows = await get_session_manager().get_session_context_entries(
        user_id=str(user.id), session_id=SESSION_ID
    )
    return {
        row.get("id"): row.get("last_served_at")
        for row in rows
        if row.get("context_profile", "qa") == "agent"
    }


def block_text(recall_result: Any) -> str:
    for item in recall_result or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if content:
            return content
    return ""


# --------------------------------------------------------------------------- printing


def print_act_header(title: str, explanation: str):
    print("", file=sys.stderr)
    print("===", file=sys.stderr)
    print(title, file=sys.stderr)
    print(explanation, file=sys.stderr)
    print("===", file=sys.stderr)
    print("", file=sys.stderr)
    sys.stderr.flush()


def print_trace_memory(trace_state: dict):
    label = f"Session memory content after trace {trace_state['trace_number']}"
    if trace_state["automatic_batch_triggered"]:
        label += f" ({_distillation_label(trace_state['trace_number'])})"

    print("---", file=sys.stderr)
    print(label, file=sys.stderr)
    print("---", file=sys.stderr)
    print_trace_details(trace_state["trace"])
    print("", file=sys.stderr)
    print_session_memory(trace_state["snapshot"])
    print("", file=sys.stderr)


def _distillation_label(trace_number: int) -> str:
    labels = {
        2: "first batch context extraction",
        4: "second batch context extraction",
    }
    return labels.get(trace_number, "batch context extraction")


def print_distillation(result: dict):
    before = result["before_flush"]
    distillation_input = result["distillation_input"]
    print(
        f"final batch context extraction: pending "
        f"{before['pending_trace_count']} -> {distillation_input['pending_trace_count']}",
        file=sys.stderr,
    )
    print(f"context entries touched: {len(result['flush_touched_entry_ids'])}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Distillation input: session memory", file=sys.stderr)
    print_snapshot(distillation_input)
    print("", file=sys.stderr)
    print("Graph documents written", file=sys.stderr)
    print(f"status: {result['distillation_status']}", file=sys.stderr)
    documents = result["cognified_documents"]
    if not documents:
        print("documents: none", file=sys.stderr)
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
        return
    for index, document in enumerate(documents, start=1):
        print(f"document {index}:", file=sys.stderr)
        print(indent_block(document.strip(), prefix="  "), file=sys.stderr)


def print_agent_lessons(agent_lessons: list[dict]):
    if not agent_lessons:
        print("  (none)", file=sys.stderr)
        return
    for lesson in agent_lessons:
        provenance = "live trace" if lesson["source_trace_ids"] else "batch LLM"
        print(
            f"    - [{lesson['section']}] {lesson['content']} "
            f"(confidence={lesson['confidence']}, source={provenance})",
            file=sys.stderr,
        )


def print_snapshot(snap: dict):
    print_session_memory(snap)


def print_trace_details(trace: dict):
    print("Trace", file=sys.stderr)
    print(f"  tool: {trace['origin_function']}", file=sys.stderr)
    print(f"  status: {trace['status']}", file=sys.stderr)
    params = trace.get("method_params") or {}
    if params:
        print("  input:", file=sys.stderr)
        for key, value in params.items():
            print(f"    {key}: {value}", file=sys.stderr)
    if trace.get("method_return_value") is not None:
        print("  output:", file=sys.stderr)
        print(indent_block(str(trace["method_return_value"]), prefix="    "), file=sys.stderr)
    if trace.get("error_message"):
        print("  error:", file=sys.stderr)
        print(indent_block(str(trace["error_message"]), prefix="    "), file=sys.stderr)


def print_session_memory(snap: dict):
    print("Session memory", file=sys.stderr)
    if not snap["agent_lessons"]:
        print("  entries: none", file=sys.stderr)
        return
    print(f"  entries ({len(snap['agent_lessons'])}):", file=sys.stderr)
    print_agent_lessons(snap["agent_lessons"])


def print_recall(result: dict):
    print("Agent-profile session memory recall", file=sys.stderr)
    print((result["agent_profile_block"] or "  (empty)"), file=sys.stderr)
    print("", file=sys.stderr)
    print(
        f"QA-profile session memory results: {result['qa_profile_result_count']}", file=sys.stderr
    )
    print(f"raw trace recall results: {result['trace_evidence_count']}", file=sys.stderr)
    print(f"writes during recall: {not result['recall_made_no_writes']}", file=sys.stderr)


def indent_block(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


async def setup_demo_data():
    progress("Clearing previous demo state.")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    progress("Creating the demo dataset.")
    await cognee.add(["Agentic session-context demo dataset."], dataset_name=DATASET_NAME)


async def reset_demo_session(user):
    deleted = await get_session_manager().delete_session(
        user_id=str(user.id), session_id=SESSION_ID
    )
    progress("Old demo session deleted." if deleted else "No previous demo session found.")


if __name__ == "__main__":
    asyncio.run(main())
