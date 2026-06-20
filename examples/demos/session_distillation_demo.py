"""Session semantic retrieval + distillation demo.

Run with:

    uv run python examples/demos/session_distillation_demo.py

This demo exercises both halves of the session memory story on the real public path:

1. While the session runs, QA turns are indexed for vector recall and learned guidance is
   stored as active working memory.
2. After the session, ``cognee.session.distill_session`` gates the learned guidance,
   curates it against the existing graph, rewrites surviving lessons with entity
   anchoring, and cognifies the rendered document into the dataset (long-term memory).

Requires a configured LLM provider. Exact wording of answers, learned guidance, and the
distilled document varies by model.
"""

import asyncio
import os
import sys

os.environ["AUTO_FEEDBACK"] = "true"
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee
from cognee import SearchType
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_embeddings import (
    search_session_qa_ids,
    select_hybrid_qa_entries,
)
from cognee.modules.users.methods import get_default_user

DATASET_NAME = "aurora_robotics_distillation_demo"
SESSION_ID = "aurora_distillation_session"

DOCUMENTS = [
    "Aurora Robotics builds two products: the VoltaArm industrial gripper and the "
    "TerraScout warehouse rover.",
    "The VoltaArm gripper uses firmware version 4 and a calibration routine that maps "
    "joint torque to grip strength.",
    "The TerraScout rover navigates warehouses using lidar maps and charging dock beacons.",
    "Aurora Robotics releases firmware through the HALT test suite, a hardware abuse "
    "test that runs overnight.",
    "Dana Voss leads the VoltaArm firmware team at Aurora Robotics.",
    "Calibration data for the VoltaArm gripper is stored in a battery-backed memory bank.",
]

MESSAGES = [
    # Orientation question.
    "What products does Aurora Robotics build and who leads VoltaArm firmware?",
    # Durable lesson.
    "Flashing VoltaArm firmware wipes calibration data, so calibration must be re-run.",
    # Durable rule.
    "Always run the HALT test suite before a VoltaArm firmware release.",
    # Session-local preference.
    "For the rest of this chat, keep answers under three bullet points.",
    # Reworded lesson; exact-only active guidance keeps it separate.
    "After a VoltaArm firmware flash, redo calibration because the flash erases it.",
    # Ordinary question; helps push the VoltaArm lesson out of the recency window.
    "How does the TerraScout rover navigate warehouses?",
    # Ordinary question; helps semantic recall stand apart from recency.
    "What is the HALT test suite and when does it run?",
    # Application question.
    "Draft the steps a technician should follow for a VoltaArm firmware update.",
]


def progress(message: str):
    print(f"[distillation-demo] {message}", file=sys.stderr, flush=True)


async def setup_demo_data():
    progress("Clearing previous demo state.")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    progress(f"Ingesting {len(DOCUMENTS)} Aurora Robotics facts.")
    await cognee.add(DOCUMENTS, dataset_name=DATASET_NAME)
    await cognee.cognify(datasets=[DATASET_NAME])
    progress("Ingestion complete.")


async def reset_demo_session(user):
    deleted = await get_session_manager().delete_session(
        user_id=str(user.id), session_id=SESSION_ID
    )
    progress("Old demo session deleted." if deleted else "No previous demo session found.")


async def ask(message: str, user):
    return await cognee.recall(
        query_text=message,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        session_id=SESSION_ID,
        user=user,
    )


async def print_session_evidence(user):
    """Show that QA turns and exact active-guidance entries are stored."""
    session_manager = get_session_manager()
    qa_entries = await cognee.session.get_session(session_id=SESSION_ID, user=user)
    context_rows = await session_manager.get_session_context_entries(
        user_id=str(user.id), session_id=SESSION_ID
    )
    guidance = [row for row in context_rows if row.get("kind", "context") == "context"]

    print(f"  qa_count={len(qa_entries)}", file=sys.stderr)
    print(f"  guidance_entries={len(guidance)}", file=sys.stderr)
    for row in guidance:
        merged_sources = len(row.get("source_feedback_ids") or [])
        print(
            f"    [{row.get('section')}] {row.get('content')!r} (sources={merged_sources})",
            file=sys.stderr,
        )


async def show_semantic_recall(user):
    """Show hybrid history selection: an old on-topic turn outside the recency window is
    semantically recalled, while older off-topic turns are not."""
    qa_entries = await cognee.session.get_session(session_id=SESSION_ID, user=user)
    query = "What happens to VoltaArm calibration when firmware is flashed?"
    semantic_qa_ids = await search_session_qa_ids(
        user_id=str(user.id),
        session_id=SESSION_ID,
        query_text=query,
    )
    selected = select_hybrid_qa_entries(qa_entries, semantic_qa_ids, last_n=2)

    selected_qa_ids = {entry.qa_id for entry in selected}
    recent_qa_ids = {entry.qa_id for entry in qa_entries[-2:]}
    progress(f"Hybrid history for {query!r} with a recency window of 2:")
    for entry in qa_entries:
        if entry.qa_id in recent_qa_ids:
            verdict = "recent"
        else:
            recalled = entry.qa_id in selected_qa_ids
            verdict = "vector recalled" if recalled else "not recalled"
        print(f"  [{verdict}] {entry.question[:90]}", file=sys.stderr)


async def run_scripted_session(user):
    for index, message in enumerate(MESSAGES, start=1):
        progress(f"Message {index}")
        await ask(message, user)
        await print_session_evidence(user)


async def distill_and_verify(user):
    progress("Distilling the session into the knowledge graph.")
    result = await cognee.session.distill_session(SESSION_ID, dataset=DATASET_NAME, user=user)
    progress(f"Distillation status={result.status} documents={len(result.documents)}")
    if result.documents:
        print(
            f"\n----- {len(result.documents)} distilled lesson documents -----\n", file=sys.stderr
        )
        for doc in result.documents:
            print(doc, file=sys.stderr)
            print("---", file=sys.stderr)

    progress("Asking the graph (fresh session) what it now knows about the lesson.")
    answer = await cognee.recall(
        query_text="What must be done after flashing VoltaArm firmware, and why?",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        session_id="verification_session",
        user=user,
    )
    print("\n----- Post-distillation graph answer -----\n", file=sys.stderr)
    print(answer, file=sys.stderr)


async def main():
    await setup_demo_data()

    user = await get_default_user()
    await reset_demo_session(user)

    await run_scripted_session(user)
    await show_semantic_recall(user)
    await distill_and_verify(user)


if __name__ == "__main__":
    asyncio.run(main())
