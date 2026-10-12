"""Ingest ONE LoCoMo conversation through cognee's memory API: sessions first, then improve().

Flow (per conversation, meant to run in its own process with its own DATA/SYSTEM roots):

1. ``remember(overview)`` — a short permanent document with the speakers and the session
   timeline. Creates the dataset.
2. For every LoCoMo session ``n``: the dialogue is split into dated windows and each window is
   written to the session cache with ``remember(text, session_id=<conv>_s<n>)``. Optionally
   every window is also run through the session-context analyzer (the same
   ``analyze_turn_for_session_context`` that powers automatic feedback) so durable
   preferences / facts become gated session guidance.
3. ``improve(dataset, session_ids=[all sessions], build_global_context_index=True)`` — the
   bridge into permanent memory: persists the session windows into the graph (add + cognify
   under the ``user_sessions_from_cache`` node set), distills the gated guidance into curated
   lessons, updates preference weights, runs the default enrichment (triplet embeddings) and
   builds the global context index.

Nothing in the dialogue reaches the graph except through ``improve()``, so the graph is an
honest picture of what the session → improve path produces.

Usage (normally driven by ``run_locomo_eval.py``)::

    uv run python -m cognee.eval_framework.locomo.ingest --conversation-index 0 \
        --run-dir temp/locomo_runs/<run>/conv_00
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Env that must be in place before cognee is imported (settings are read at import time).
os.environ.setdefault("CACHING", "true")
os.environ.setdefault("AUTO_FEEDBACK", "true")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("COGNEE_LOG_FILE", "false")

from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter
from cognee.eval_framework.locomo.model_registry import ensure_model_registered
from cognee.eval_framework.locomo.preprocess import (
    DEFAULT_WINDOW_TURNS,
    build_conversation_bundle,
    write_conversation_files,
)
from cognee.eval_framework.reporting.io import write_json


def print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def analyze_windows(
    windows: list[dict[str, Any]],
    *,
    session_manager,
    user_id: str,
    session_id: str,
    concurrency: int,
) -> int:
    """Run the session-context analyzer over every window; apply results in order.

    The analyzer expects a "user message" and the previous exchange. Consecutive dialogue
    windows play those roles: the previous window is the prior turn. Ratings need a
    ``previous_qa_id`` and personalization, so only candidate context updates land here —
    exactly the entries ``improve()`` later distills.
    """
    from cognee.infrastructure.session.feedback_detection import analyze_turn_for_session_context
    from cognee.infrastructure.session.session_turn import apply_session_turn_analysis

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def analyze(index: int):
        previous = windows[index - 1]["text"] if index > 0 else None
        async with semaphore:
            try:
                return await analyze_turn_for_session_context(
                    windows[index]["text"], previous_question=previous, previous_answer=None
                )
            except Exception as error:  # fail-open: analysis is an enrichment, not the data
                print_step(f"  turn analysis failed for window {index}: {error}")
                return None

    analyses = await asyncio.gather(*(analyze(i) for i in range(len(windows))))

    touched = 0
    for window, analysis in zip(windows, analyses):
        if analysis is None:
            continue
        ids = await apply_session_turn_analysis(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            query=window["text"],
            analysis=analysis,
            previous_qa_id=None,
            served_ids=[],
        )
        touched += len(ids)
    return touched


async def ingest_conversation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "conversation_index": args.conversation_index,
        "answer_model": os.getenv("LLM_MODEL"),
        "window_turns": args.window_turns,
        "max_sessions": args.max_sessions,
        "turn_analysis": not args.skip_turn_analysis,
        "global_context_index": not args.skip_global_context_index,
        "data_root_directory": os.getenv("DATA_ROOT_DIRECTORY"),
        "system_root_directory": os.getenv("SYSTEM_ROOT_DIRECTORY"),
        "sessions": [],
    }

    if os.getenv("LLM_MODEL"):
        ensure_model_registered(os.environ["LLM_MODEL"])

    import cognee
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.users.methods import get_default_user

    adapter = LocomoAdapter(
        conversation_index=args.conversation_index,
        max_sessions=args.max_sessions,
        data_path=args.data_path,
    )
    conversation = adapter.load_conversation(args.conversation_index)
    bundle = build_conversation_bundle(conversation, args.window_turns)
    dataset_name = bundle["dataset_name"]
    report["dataset_name"] = dataset_name
    report["sample_id"] = bundle["sample_id"]

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_conversation_files(bundle, run_dir / "preprocessed")

    if args.prune_first:
        print_step("Pruning existing cognee state")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    await setup()
    user = await get_default_user()
    user_id = str(user.id)
    session_manager = get_session_manager()
    if not session_manager.is_available:
        raise RuntimeError("Session cache unavailable — CACHING must be true for this ingestion")

    # 1. Permanent overview document (also creates the dataset).
    print_step(f"{bundle['sample_id']}: remember() overview into dataset {dataset_name}")
    overview_started = time.monotonic()
    await cognee.remember(
        bundle["overview"], dataset_name=dataset_name, self_improvement=False, user=user
    )
    report["overview_seconds"] = time.monotonic() - overview_started

    # 2. Session memory, one cognee session per LoCoMo session.
    session_ids: list[str] = []
    total_windows = 0
    for position, session in enumerate(bundle["sessions"], start=1):
        session_id = session["session_id"]
        windows = session["windows"]
        session_started = time.monotonic()
        print_step(
            f"{bundle['sample_id']}: session {position}/{len(bundle['sessions'])} "
            f"({session['date_time']}, {session['turn_count']} turns, {len(windows)} windows) "
            f"-> remember(session_id={session_id})"
        )
        await session_manager.delete_session(user_id=user_id, session_id=session_id)
        for window in windows:
            await cognee.remember(
                window["text"],
                dataset_name=dataset_name,
                session_id=session_id,
                self_improvement=False,
                user=user,
            )
        write_seconds = time.monotonic() - session_started

        touched = 0
        analysis_seconds = 0.0
        if not args.skip_turn_analysis:
            analysis_started = time.monotonic()
            touched = await analyze_windows(
                windows,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                concurrency=args.analysis_concurrency,
            )
            analysis_seconds = time.monotonic() - analysis_started
            print_step(f"  turn analysis: {touched} context entries in {analysis_seconds:.1f}s")

        session_ids.append(session_id)
        total_windows += len(windows)
        report["sessions"].append(
            {
                "session_index": session["session_index"],
                "session_id": session_id,
                "date_time": session["date_time"],
                "turn_count": session["turn_count"],
                "window_count": len(windows),
                "session_write_seconds": write_seconds,
                "turn_analysis_seconds": analysis_seconds,
                "context_entries_touched": touched,
            }
        )

    report["session_count"] = len(session_ids)
    report["window_count"] = total_windows

    # 3. Bridge everything into permanent memory.
    print_step(
        f"{bundle['sample_id']}: improve(dataset={dataset_name}, {len(session_ids)} sessions, "
        f"global_context_index={not args.skip_global_context_index})"
    )
    improve_started = time.monotonic()
    await cognee.improve(
        dataset=dataset_name,
        session_ids=session_ids,
        build_global_context_index=not args.skip_global_context_index,
        user=user,
    )
    report["improve_seconds"] = time.monotonic() - improve_started
    print_step(f"{bundle['sample_id']}: improve done in {report['improve_seconds']:.1f}s")

    # Graph size, for the report.
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
        report["graph_nodes"] = len(nodes)
        report["graph_edges"] = len(edges)
    except Exception as error:
        report["graph_size_error"] = str(error)

    report["total_seconds"] = time.monotonic() - started
    report["finished_at"] = utc_now()
    report["status"] = "completed"
    write_json(str(run_dir / "ingestion_report.json"), report)
    print_step(
        f"{bundle['sample_id']}: ingestion complete in {report['total_seconds'] / 60:.1f} min "
        f"(nodes={report.get('graph_nodes')}, edges={report.get('graph_edges')})"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-index", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--window-turns", type=int, default=DEFAULT_WINDOW_TURNS)
    parser.add_argument("--analysis-concurrency", type=int, default=8)
    parser.add_argument("--skip-turn-analysis", action="store_true")
    parser.add_argument("--skip-global-context-index", action="store_true")
    parser.add_argument("--prune-first", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(ingest_conversation(args))
    except Exception as error:
        write_json(
            str(args.run_dir / "ingestion_report.json"),
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        print_step(f"INGESTION FAILED: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
