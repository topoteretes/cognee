"""Session semantic retrieval + distillation demo.

Run with:

    uv run python examples/demos/session_distillation_demo.py

Reuse already-ingested demo data and start a fresh session:

    uv run python examples/demos/session_distillation_demo.py --no-ingest

This demo exercises both halves of the session memory story on the real public path:

1. While the session runs, QA turns and learned guidance entries are embedded at write
   time; guidance is ranked semantically and similar candidates merge instead of
   duplicating (working memory).
2. After the session, ``cognee.session.distill_session`` gates the learned guidance,
   curates it against the existing graph, rewrites surviving lessons with entity
   anchoring, and cognifies the rendered document into the dataset (long-term memory).

Requires a configured LLM provider. Exact wording of answers, learned guidance, and the
distilled document varies by model.
"""

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

os.environ["CACHING"] = "true"
os.environ["CACHE_BACKEND"] = "fs"
os.environ["AUTO_FEEDBACK"] = "true"
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee
from cognee import SearchType
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_embeddings import (
    MIN_QA_SIMILARITY,
    cosine_similarity,
    embed_text_safe,
    select_hybrid_qa_entries,
)
from cognee.modules.users.methods import get_default_user

DATASET_NAME = "aurora_robotics_distillation_demo"
SESSION_ID = "aurora_distillation_session"
DEMO_ROOT = Path(__file__).resolve().parents[2] / "temp" / "session_distillation_demo"

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

TURNS = [
    {
        "label": "orientation_question",
        "message": "What products does Aurora Robotics build and who leads VoltaArm firmware?",
    },
    {
        "label": "durable_lesson",
        "message": (
            "Important thing we discovered today: flashing VoltaArm firmware wipes the "
            "calibration data, so the calibration routine must be re-run after every "
            "firmware flash."
        ),
    },
    {
        "label": "durable_rule",
        "message": (
            "Going forward, always recommend running the HALT test suite before any "
            "VoltaArm firmware release."
        ),
    },
    {
        "label": "session_local_preference",
        "message": "For the rest of this chat, keep answers under three bullet points.",
    },
    {
        "label": "near_duplicate_lesson",
        "message": (
            "Just to repeat the key finding differently: after you flash firmware on the "
            "VoltaArm, calibration has to be done again because the flash erases it."
        ),
    },
    # Two ordinary questions on other topics. Guidance-only turns above store no QA entry,
    # so these create the QA history that pushes the VoltaArm turn out of the recency
    # window — letting show_semantic_recall demonstrate an actual semantic recall.
    {
        "label": "terrascout_question",
        "message": "How does the TerraScout rover navigate warehouses?",
    },
    {
        "label": "halt_question",
        "message": "What is the HALT test suite and when does it run?",
    },
    {
        "label": "apply_the_lesson",
        "message": "Draft the steps a technician should follow for a VoltaArm firmware update.",
    },
]


def progress(message: str):
    print(f"[distillation-demo] {message}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Reuse the existing isolated demo dataset; only delete the demo session.",
    )
    return parser.parse_args()


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


async def configure_demo_storage(*, reset_storage: bool):
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_config
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )
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
    await create_db_and_tables()


async def setup_demo_data(no_ingest: bool):
    await configure_demo_storage(reset_storage=not no_ingest)
    if no_ingest:
        progress("Skipping ingestion; reusing existing isolated demo data.")
        return
    progress("Clearing previous demo state.")
    await cognee.forget(everything=True)
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
    """Show that QA turns and guidance entries carry embeddings and merge near-dups."""
    session_manager = get_session_manager()
    qa_entries = await cognee.session.get_session(session_id=SESSION_ID, user=user)
    context_rows = await session_manager.get_session_context_entries(
        user_id=str(user.id), session_id=SESSION_ID
    )
    guidance = [row for row in context_rows if row.get("kind", "context") == "context"]

    embedded_qa = sum(1 for entry in qa_entries if entry.embedding)
    print(f"  qa_count={len(qa_entries)} (with embeddings: {embedded_qa})", file=sys.stderr)
    print(f"  guidance_entries={len(guidance)}", file=sys.stderr)
    for row in guidance:
        merged_sources = len(row.get("source_feedback_ids") or [])
        has_embedding = bool(row.get("embedding"))
        print(
            f"    [{row.get('section')}] {row.get('content')!r} "
            f"(sources={merged_sources}, embedded={has_embedding})",
            file=sys.stderr,
        )


async def show_semantic_recall(user):
    """Show hybrid history selection: an old on-topic turn outside the recency window is
    semantically recalled, while older off-topic turns are not."""
    qa_entries = await cognee.session.get_session(session_id=SESSION_ID, user=user)
    query = "What happens to VoltaArm calibration when firmware is flashed?"
    query_embedding = await embed_text_safe(query)
    selected = select_hybrid_qa_entries(qa_entries, query_embedding, last_n=2)

    selected_qa_ids = {entry.qa_id for entry in selected}
    recent_qa_ids = {entry.qa_id for entry in qa_entries[-2:]}
    progress(
        f"Hybrid history for {query!r} with a recency window of 2 "
        f"(similarity floor {MIN_QA_SIMILARITY}):"
    )
    for entry in qa_entries:
        if entry.qa_id in recent_qa_ids:
            verdict = "recent"
        else:
            similarity = cosine_similarity(query_embedding or [], entry.embedding or [])
            recalled = entry.qa_id in selected_qa_ids
            verdict = (
                f"{'semantically recalled' if recalled else 'not recalled'}, "
                f"similarity={similarity:.3f}"
            )
        print(f"  [{verdict}] {entry.question[:90]}", file=sys.stderr)


async def main():
    args = parse_args()
    await setup_demo_data(args.no_ingest)

    user = await get_default_user()
    await reset_demo_session(user)

    for index, turn in enumerate(TURNS, start=1):
        progress(f"Turn {index}: {turn['label']}")
        await ask(turn["message"], user)
        await print_session_evidence(user)

    await show_semantic_recall(user)

    progress("Distilling the session into the knowledge graph.")
    result = await cognee.session.distill_session(SESSION_ID, dataset=DATASET_NAME, user=user)
    progress(
        f"Distillation status={result.status} gated={result.gated_entry_count} "
        f"lessons={result.lesson_count} already_known={result.skipped_already_known}"
    )
    if result.document:
        print("\n----- Distilled document -----\n", file=sys.stderr)
        print(result.document, file=sys.stderr)

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


if __name__ == "__main__":
    asyncio.run(main())
