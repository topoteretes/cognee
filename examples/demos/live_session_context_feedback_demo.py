"""Live session-context feedback demo.

Run with:

    uv run python examples/demos/live_session_context_feedback_demo.py

Reuse already-ingested demo data and start a fresh session:

    uv run python examples/demos/live_session_context_feedback_demo.py --no-ingest

This demo uses the real public ingestion and recall path. It ingests a small imagined company
dataset, asks questions in one session, sends feedback as user messages, and prints JSON evidence
showing QA storage, feedback persistence, and session-context growth.

Requires a configured LLM provider because ingestion, answer generation, and feedback detection are
live. Exact answer wording and learned guidance text can vary by model.
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
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee
from cognee import SearchType
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.methods import get_default_user

DATASET_NAME = "northstar_labs_live_session_demo"
SESSION_ID = "northstar_live_session"
DEMO_ROOT = Path(__file__).resolve().parents[2] / "temp" / "live_session_context_feedback_demo"

DOCUMENTS = [
    "Northstar Labs runs the Berlin office, the Lisbon office, the Toronto office, "
    "and the Singapore office; each office owns one logistics intelligence project.",
    "The Berlin office owns RoutePulse, a project that predicts delivery delays for "
    "European freight operators.",
    "The Lisbon office owns HarborLens, a project that monitors port congestion and "
    "recommends alternate unloading windows.",
    "The Toronto office owns FrostLine, a project that helps cold-chain teams track "
    "temperature risk during winter shipments.",
    "The Singapore office owns SkyBridge, a project that coordinates air-cargo handoffs "
    "between regional carriers.",
    "RoutePulse uses traffic feeds, weather alerts, and customs delay reports to estimate "
    "arrival risk.",
    "HarborLens uses vessel schedules, berth availability, and labor notices to forecast "
    "port bottlenecks.",
    "FrostLine uses sensor readings, weather forecasts, and route duration to warn about "
    "spoiled-goods risk.",
    "SkyBridge uses flight status, warehouse capacity, and customs clearance events to "
    "recommend cargo transfer plans.",
    "Northstar Labs asks customer-facing teams to explain project details in concise "
    "operational language.",
    "The Berlin office audit window is Monday morning, and the Berlin office audit should "
    "review RoutePulse traffic feeds, weather alerts, and customs delay reports.",
    "The Lisbon office audit window is Tuesday afternoon, and the Lisbon office audit should "
    "review HarborLens vessel schedules, berth availability, and labor notices.",
    "The Singapore office audit lead is Priya Tan, and Priya Tan is available Wednesday "
    "morning for the Singapore office SkyBridge audit.",
    "The Toronto office audit lead is Mateo Reed, and Mateo Reed is available Thursday "
    "afternoon for the Toronto office FrostLine audit.",
    "Northstar Labs audit trips should avoid unnecessary backtracking while still respecting "
    "local office availability windows.",
]

TURNS = [
    {
        "label": "initial_audit_question",
        "message": (
            "I'm planning an audit trip across Northstar Labs offices. Which offices, "
            "projects, and audit topics should I include?"
        ),
    },
    {
        "label": "goal_and_order_preference",
        "message": (
            "That helps. My goal is to create a practical audit itinerary, and I prefer "
            "visiting Berlin and Lisbon before Singapore and Toronto. For now, answer with "
            "2 informative bullet points."
        ),
    },
    {
        "label": "route_question_after_preference",
        "message": (
            "Given that preference, what visit order would you suggest and what should "
            "I audit in each office?"
        ),
    },
    {
        "label": "correction_to_route",
        "message": (
            "Wait, Singapore can't be flexible. Priya is only free Wednesday morning, "
            "so Singapore needs to happen before Toronto."
        ),
    },
    {
        "label": "priya_context_lesson",
        "message": (
            "Actually, from past audits, Priya usually has the context Mateo needs. "
            "It would be useful to talk to Priya before Mateo."
        ),
    },
    {
        "label": "lisbon_video_call_lesson",
        "message": (
            "Also, I know Lisbon is running a good operation, so I probably don't need "
            "the full site visit there. A video call should be enough unless something "
            "looks risky."
        ),
    },
    {
        "label": "updated_route_question",
        "message": (
            "Can you revise the trip plan with the right order, the Lisbon video call, "
            "and the audit focus for each stop?"
        ),
    },
    {
        "label": "communication_preference_update",
        "message": (
            "Actually, change my communication preference: I now prefer 4 concise bullet "
            "points instead of 2 informative bullet points."
        ),
    },
    {
        "label": "customer_facing_style_rule",
        "message": (
            "Good. Also remember that customer-facing audit notes should be operational, "
            "not technical."
        ),
    },
    {
        "label": "final_summary_question",
        "message": "Draft the final customer-facing audit trip summary.",
    },
]


def force_demo_environment():
    os.environ["CACHING"] = "true"
    os.environ["CACHE_BACKEND"] = "fs"
    os.environ["AUTO_FEEDBACK"] = "true"
    os.environ["DATA_ROOT_DIRECTORY"] = str(DEMO_ROOT / "data")
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(DEMO_ROOT / "system")
    os.environ["CACHE_ROOT_DIRECTORY"] = str(DEMO_ROOT / "cache")


def clear_cache_if_available(fn):
    cache_clear = getattr(fn, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def progress(message: str):
    print(f"[live-session-demo] {message}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "Reuse the existing isolated demo dataset, skip forget/remember, "
            "and only delete the demo session before running."
        ),
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.no_ingest:
        progress("Preparing isolated demo storage without ingesting data.")
        await setup_existing_demo_data()
    else:
        progress("Preparing isolated demo data.")
        await setup_demo_data()

    progress("Loading default user.")
    user = await get_default_user()
    session_was_deleted = await reset_demo_session(user)

    output = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "no_ingest": args.no_ingest,
        "session_was_deleted": session_was_deleted,
        "context_only_probe": await run_context_only_probe(user),
        "turns": [],
    }

    for index, turn in enumerate(TURNS, start=1):
        progress(f"Turn {index}: {turn['label']}")
        response = await ask(turn["message"], user=user, only_context=False)
        evidence = await session_evidence(user)
        print_turn_snapshot(
            turn_number=index,
            label=turn["label"],
            user_message=turn["message"],
            response=response,
            evidence=evidence,
        )
        output["turns"].append(
            {
                "turn": index,
                "label": turn["label"],
                "user_message": turn["message"],
                "assistant_response": serialize_response(response),
                "evidence": evidence,
            }
        )

    print(json.dumps(output, indent=2))


async def setup_demo_data():
    await configure_demo_storage(reset_storage=True)
    progress("Clearing previous demo state.")
    await cognee.forget(everything=True)
    progress(f"Ingesting {len(DOCUMENTS)} Northstar Labs facts.")
    await cognee.remember(DOCUMENTS, dataset_name=DATASET_NAME, self_improvement=False)
    progress("Ingestion complete.")


async def setup_existing_demo_data():
    await configure_demo_storage(reset_storage=False)
    progress("Skipping ingestion; reusing existing isolated demo data.")


async def configure_demo_storage(*, reset_storage: bool):
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_config
    from cognee.infrastructure.databases.vector import get_vectordb_config
    from cognee.infrastructure.databases.vector.get_vector_engine import create_vector_engine

    force_demo_environment()
    progress(f"Using isolated demo root: {DEMO_ROOT}")
    if reset_storage:
        shutil.rmtree(DEMO_ROOT, ignore_errors=True)
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    clear_cache_if_available(get_base_config)
    clear_cache_if_available(get_relational_config)
    clear_cache_if_available(get_graph_config)
    clear_cache_if_available(get_vectordb_config)
    clear_cache_if_available(get_cache_config)
    cognee.config.data_root_directory(str(DEMO_ROOT / "data"))
    cognee.config.system_root_directory(str(DEMO_ROOT / "system"))
    clear_cache_if_available(create_graph_engine)
    clear_cache_if_available(create_vector_engine)
    clear_cache_if_available(create_cache_engine)
    progress("Creating database tables.")
    await create_db_and_tables()


async def reset_demo_session(user) -> bool:
    progress(f"Deleting old demo session: {SESSION_ID}")
    deleted = await get_session_manager().delete_session(
        user_id=str(user.id),
        session_id=SESSION_ID,
    )
    if deleted:
        progress("Old demo session deleted.")
    else:
        progress("No previous demo session found; starting clean.")
    return deleted


async def run_context_only_probe(user) -> dict:
    progress("Running context-only probe; this should not register a QA entry.")
    before = await session_evidence(user)
    context = await ask(
        "Which Northstar offices are mentioned?",
        user=user,
        only_context=True,
    )
    after = await session_evidence(user)
    progress(
        "Context-only probe complete: "
        f"QA count before={before['qa_count']}, after={after['qa_count']}."
    )
    return {
        "message": "Which Northstar offices are mentioned?",
        "only_context": True,
        "returned_context": serialize_response(context),
        "qa_count_before": before["qa_count"],
        "qa_count_after": after["qa_count"],
        "qa_was_registered": after["qa_count"] > before["qa_count"],
    }


def print_turn_snapshot(
    *,
    turn_number: int,
    label: str,
    user_message: str,
    response: Any,
    evidence: dict,
):
    print("", file=sys.stderr)
    print(f"--- Turn {turn_number}: {label} ---", file=sys.stderr)
    print(f"user: {preview_text(user_message, max_chars=700)}", file=sys.stderr)
    for text in response_texts(response):
        print(f"assistant: {preview_text(text, max_chars=700)}", file=sys.stderr)

    latest_qa = evidence["latest_qa"]
    print(f"qa_count: {evidence['qa_count']}", file=sys.stderr)
    if latest_qa is None:
        print("latest_qa: none", file=sys.stderr)
    else:
        print(f"latest_qa.question: {latest_qa['question']}", file=sys.stderr)
        print(
            f"latest_qa.used_session_context_ids: {latest_qa['used_session_context_ids']}",
            file=sys.stderr,
        )

    print_session_context(evidence["session_context_entries"])


def print_session_context(entries: list[dict]):
    if not entries:
        print("session_context: empty", file=sys.stderr)
        return

    print("session_context:", file=sys.stderr)
    for entry in entries:
        print(
            "  "
            f"- [{entry['section']}] {entry['content']} "
            f"(helpful={entry['helpful_count']}, harmful={entry['harmful_count']})",
            file=sys.stderr,
        )


def response_texts(response: Any) -> list[str]:
    if isinstance(response, list):
        texts = []
        for item in response:
            texts.extend(response_texts(item))
        return texts
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        return [str(data.get("text") or data.get("content") or data)]
    return [str(response)]


async def ask(message: str, *, user, only_context: bool) -> Any:
    return await cognee.recall(
        query_text=message,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        session_id=SESSION_ID,
        user=user,
        only_context=only_context,
    )


async def session_evidence(user) -> dict:
    qa_entries = await cognee.session.get_session(session_id=SESSION_ID, user=user)
    context_entries = await get_session_manager().get_session_context_entries(
        user_id=str(user.id),
        session_id=SESSION_ID,
    )
    return {
        "qa_count": len(qa_entries),
        "latest_qa": serialize_latest_qa(qa_entries),
        "session_context_entries": serialize_context_entries(context_entries),
    }


def serialize_response(response: Any) -> Any:
    if isinstance(response, list):
        return [serialize_response(item) for item in response]
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        text = data.get("text") or data.get("content") or str(data)
        return {
            "source": data.get("source"),
            "kind": data.get("kind"),
            "text": preview_text(text),
            "text_length": len(text),
        }
    if isinstance(response, str):
        return {"text": preview_text(response), "text_length": len(response)}
    return response


def preview_text(text: str, max_chars: int = 1200) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def serialize_latest_qa(qa_entries: list) -> dict | None:
    if not qa_entries:
        return None
    latest = qa_entries[-1]
    return {
        "qa_id": latest.qa_id,
        "question": latest.question,
        "answer": latest.answer,
        "used_session_context_ids": latest.used_session_context_ids,
    }


def serialize_context_entries(entries: list[dict]) -> list[dict]:
    visible_entries = []
    for entry in entries:
        if entry.get("kind", "context") != "context":
            continue
        visible_entries.append(
            {
                "id": entry.get("id"),
                "section": entry.get("section"),
                "content": entry.get("content"),
                "helpful_count": entry.get("helpful_count", 0),
                "harmful_count": entry.get("harmful_count", 0),
                "source_feedback_ids": entry.get("source_feedback_ids", []),
            }
        )
    return visible_entries


if __name__ == "__main__":
    asyncio.run(main())
