"""Agentic session-context demo.

The agentic counterpart to ``live_session_context_feedback_demo.py``. Where the QA demo learns
lessons from conversation feedback, this one learns ``agent``-profile lessons from an agent's
tool/action traces — and, like that demo, the real lesson extraction is LLM-backed.

Run (full, requires a configured LLM provider):

    uv run python examples/demos/agentic_session_context_demo.py

Run a reduced, no-LLM version (only the immediate deterministic failure capture + recall):

    uv run python examples/demos/agentic_session_context_demo.py --offline

What it shows:

  Act 1  remember(TraceEntry ...) stores raw tool traces in the session cache. The errored trace
         also creates one immediate, deterministic failure lesson. The demo prints every active
         context entry extracted so far.
  Act 2  The batch extractor reads all traces and adds richer active guidance. Then
         distill_session() rewrites accepted guidance into markdown documents and add+cognifies
         those documents into the graph. The demo prints the exact cognified documents.
  Act 3  recall(scope=["session_context"], context_profile="agent") returns the active guidance
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
import shutil
import sys
from pathlib import Path
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
DEMO_ROOT = Path(__file__).resolve().parents[2] / "temp" / "agentic_session_context_demo"

# One agent working a task: a few tool steps, including a failure, that together imply reusable
# lessons (e.g. "this repo needs `uv sync` before tests", "tests run with uv run pytest").
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
]


def progress(message: str):
    print(f"[agentic-demo] {message}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the LLM extraction acts; show only the deterministic capture and recall.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    await configure_demo_storage()
    user = await get_default_user()
    await reset_demo_session(user)

    output = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "mode": "offline" if args.offline else "full",
    }

    # Act 1: store agent traces. The errored step yields an immediate, no-LLM failure capture.
    progress("Act 1: storing agent traces via remember(TraceEntry).")
    print_phase_overview()
    for trace in TRACES:
        await cognee.remember(
            TraceEntry(**trace),
            dataset_name=DATASET_NAME,
            session_id=SESSION_ID,
            self_improvement=False,
            user=user,
        )
    output["act1_traces_and_live_capture"] = await snapshot(user)
    print_extracted_entries(
        "Act 1 — extracted context entries after raw trace storage",
        output["act1_traces_and_live_capture"],
        note=(
            "Expected: exactly one live failure lesson, because only errored traces with "
            "an error_message are extracted without an LLM."
        ),
    )

    if args.offline:
        progress("Offline mode: skipping batch extraction and distillation.")
        output["act3_recall"] = await run_recall_act(user)
        print(json.dumps(output, indent=2))
        return

    # Act 2: run the extraction/distillation steps explicitly so the demo can print documents.
    progress("Act 2: batch extraction, then distill_session() add+cognify.")
    output["act2_distillation"] = await run_distillation_act(user)
    print_distillation(output["act2_distillation"])

    # Act 3: read-only recall — profile rendering, isolation, evidence preservation, no writes.
    output["act3_recall"] = await run_recall_act(user)

    print(json.dumps(output, indent=2))


async def run_distillation_act(user) -> dict:
    """Run the same agent-context extraction/distillation pieces that improve() orchestrates."""
    from cognee.infrastructure.session.agent_context_extraction import extract_pending_agent_context
    from cognee.modules.session_distillation import distill_session

    try:
        touched_ids = await extract_pending_agent_context(
            session_manager=get_session_manager(),
            user_id=str(user.id),
            session_id=SESSION_ID,
            min_new_traces=1,
        )
        extracted = await snapshot(user)
        result = await distill_session(SESSION_ID, dataset=DATASET_NAME, user=user)
        return {
            "batch_touched_entry_ids": touched_ids,
            "active_context_entries_after_batch": extracted["agent_lessons"],
            "distillation_status": result.status,
            "cognified_documents": result.documents,
        }
    except Exception as exc:
        progress(f"Act 2 failed: {exc}")
        return {
            "batch_touched_entry_ids": [],
            "active_context_entries_after_batch": await agent_lessons_for(user),
            "distillation_status": "failed",
            "cognified_documents": [],
            "error": str(exc),
        }


async def run_recall_act(user) -> dict:
    progress("Act 3: read-only recall (agent vs qa profile, trace evidence, no writes).")
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
    progress(
        "Act 3 done: agent block "
        f"{'present' if result['agent_profile_block'] else 'empty'}, "
        f"qa results={result['qa_profile_result_count']}, "
        f"trace evidence={result['trace_evidence_count']}, "
        f"no_writes={result['recall_made_no_writes']}."
    )
    print_recall(result)
    return result


# --------------------------------------------------------------------------- evidence


async def snapshot(user) -> dict:
    traces = await get_session_manager().get_agent_trace_session(
        user_id=str(user.id), session_id=SESSION_ID
    )
    return {"trace_count": len(traces), "agent_lessons": await agent_lessons_for(user)}


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


def print_phase_overview():
    print("", file=sys.stderr)
    print("--- Demo map ---", file=sys.stderr)
    print("1. Store traces. Print every extracted active context entry.", file=sys.stderr)
    print(
        "   Expect one entry: the deterministic failure lesson from the errored trace.",
        file=sys.stderr,
    )
    print(
        "2. Run batch extraction and distillation. Print the exact documents add+cognify wrote.",
        file=sys.stderr,
    )
    print(
        "3. Recall. Print what should be present, what should be absent, and why.",
        file=sys.stderr,
    )


def print_extracted_entries(title: str, snap: dict, *, note: str):
    print("", file=sys.stderr)
    print(f"--- {title} ---", file=sys.stderr)
    print(note, file=sys.stderr)
    print(f"raw traces stored: {snap['trace_count']}", file=sys.stderr)
    if not snap["agent_lessons"]:
        print("extracted active context entries: none", file=sys.stderr)
        return
    print(f"extracted active context entries ({len(snap['agent_lessons'])}):", file=sys.stderr)
    print_agent_lessons(snap["agent_lessons"])


def print_distillation(result: dict):
    print("", file=sys.stderr)
    print("--- Act 2 — distilled and cognified graph documents ---", file=sys.stderr)
    print(
        "Before distillation, the batch extractor may add richer active context entries from all traces.",
        file=sys.stderr,
    )
    print(
        f"batch touched entries: {len(result['batch_touched_entry_ids'])}",
        file=sys.stderr,
    )
    print(
        f"active context entries available to distillation: "
        f"{len(result['active_context_entries_after_batch'])}",
        file=sys.stderr,
    )
    print_agent_lessons(result["active_context_entries_after_batch"])
    print("", file=sys.stderr)
    print(
        "Distillation rewrites accepted entries into standalone markdown documents and "
        "add+cognifies them into node_set='session_learnings'.",
        file=sys.stderr,
    )
    print(f"distillation status: {result['distillation_status']}", file=sys.stderr)
    documents = result["cognified_documents"]
    if not documents:
        print("cognified documents: none", file=sys.stderr)
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
        return
    print(f"cognified documents ({len(documents)}):", file=sys.stderr)
    for index, document in enumerate(documents, start=1):
        print(f"  document {index}:", file=sys.stderr)
        print(indent_block(document.strip(), prefix="    "), file=sys.stderr)


def print_agent_lessons(agent_lessons: list[dict]):
    if not agent_lessons:
        print("  (none)", file=sys.stderr)
        return
    for lesson in agent_lessons:
        provenance = "live trace" if lesson["source_trace_ids"] else "batch LLM"
        print(
            f"  - [{lesson['section']}] {lesson['content']} "
            f"(confidence={lesson['confidence']}, source={provenance})",
            file=sys.stderr,
        )


def print_recall(result: dict):
    print("", file=sys.stderr)
    print("--- Act 3: read-only recall ---", file=sys.stderr)
    print(
        "Expected: agent profile has guidance, because prior acts wrote agent-profile entries.",
        file=sys.stderr,
    )
    print("agent profile block:", file=sys.stderr)
    print((result["agent_profile_block"] or "  (empty)"), file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Expected: QA profile has 0 results, because this demo never wrote QA-profile context.",
        file=sys.stderr,
    )
    print(f"qa profile results: {result['qa_profile_result_count']}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Expected: raw trace evidence is still recallable separately.", file=sys.stderr)
    print(f"raw trace evidence results: {result['trace_evidence_count']}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Expected: recall is read-only and does not update last_served_at.", file=sys.stderr)
    print(f"recall made no writes: {result['recall_made_no_writes']}", file=sys.stderr)


def indent_block(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


# --------------------------------------------------------------------------- storage setup


def clear_cache_if_available(fn):
    cache_clear = getattr(fn, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


async def configure_demo_storage():
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_config
    from cognee.infrastructure.databases.relational.create_db_and_tables import create_db_and_tables
    from cognee.infrastructure.databases.vector import get_vectordb_config
    from cognee.infrastructure.databases.vector.get_vector_engine import create_vector_engine

    progress(f"Using isolated demo root: {DEMO_ROOT}")
    shutil.rmtree(DEMO_ROOT, ignore_errors=True)
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_ROOT_DIRECTORY"] = str(DEMO_ROOT / "data")
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(DEMO_ROOT / "system")
    os.environ["CACHE_ROOT_DIRECTORY"] = str(DEMO_ROOT / "cache")
    for fn in (
        get_base_config,
        get_relational_config,
        get_graph_config,
        get_vectordb_config,
        get_cache_config,
        create_graph_engine,
        create_vector_engine,
        create_cache_engine,
    ):
        clear_cache_if_available(fn)
    cognee.config.data_root_directory(str(DEMO_ROOT / "data"))
    cognee.config.system_root_directory(str(DEMO_ROOT / "system"))
    progress("Creating database tables.")
    await create_db_and_tables()
    # A no-LLM add gives the demo user a dataset grant so trace storage, recall, and improve()
    # resolve cleanly (and no permission-denied noise is logged).
    progress("Creating the demo dataset.")
    await cognee.add(["Agentic session-context demo dataset."], dataset_name=DATASET_NAME)


async def reset_demo_session(user):
    progress(f"Resetting demo session: {SESSION_ID}")
    await get_session_manager().delete_session(user_id=str(user.id), session_id=SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
