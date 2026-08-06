"""Sequentially ingest BEAM JSON-list sessions with parallel session distillation.

Given one conversation folder created by the preprocessing stage, this script ingests each
session/batch file one at a time:

1. write a limited copy of the session file into a run folder,
2. parse the known user/assistant turns for session-memory candidate detection,
3. concurrently:
   - ``cognee.add`` that one JSON file and ``cognee.cognify`` it with ``JsonListChunker``,
   - analyze all turns for session-memory candidates in parallel, then apply them sequentially,
4. distill the harvested session learnings into the same dataset,
5. optionally update the global context index.

The normal BEAM ingestion remains session-file oriented: each JSON-list file is added as
one document source, and ``JsonListChunker`` splits the list items into graph chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognee.eval_framework.beam.session_io import (
    distillation_report_payload,
    parse_turns,
    read_json_list,
    session_id_for,
    write_json,
)

# Keep Cognee's normal logging quiet; this script prints its own progress.
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["COGNEE_LOG_FILE"] = "false"
os.environ["COGNEE_CLI_MODE"] = "true"
os.environ.setdefault("CACHING", "true")
os.environ.setdefault("CACHE_BACKEND", "fs")
os.environ.setdefault("AUTO_FEEDBACK", "true")


def import_cognee_modules():
    if os.getenv("BEAM_INGEST_SHOW_COGNEE_LOGS") == "1":
        import cognee
        from cognee.infrastructure.session.feedback_detection import (
            analyze_turn_for_session_context,
        )
        from cognee.infrastructure.session.get_session_manager import get_session_manager
        from cognee.infrastructure.session.session_turn import apply_session_turn_analysis
        from cognee.memify_pipelines.global_context_index import global_context_index_pipeline
        from cognee.modules.chunking.JsonListChunker import JsonListChunker
        from cognee.modules.users.methods.get_default_user import get_default_user

        return (
            cognee,
            global_context_index_pipeline,
            JsonListChunker,
            get_default_user,
            analyze_turn_for_session_context,
            get_session_manager,
            apply_session_turn_analysis,
        )

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            import cognee
            from cognee.infrastructure.session.feedback_detection import (
                analyze_turn_for_session_context,
            )
            from cognee.infrastructure.session.get_session_manager import get_session_manager
            from cognee.infrastructure.session.session_turn import apply_session_turn_analysis
            from cognee.memify_pipelines.global_context_index import global_context_index_pipeline
            from cognee.modules.chunking.JsonListChunker import JsonListChunker
            from cognee.modules.users.methods.get_default_user import get_default_user

    return (
        cognee,
        global_context_index_pipeline,
        JsonListChunker,
        get_default_user,
        analyze_turn_for_session_context,
        get_session_manager,
        apply_session_turn_analysis,
    )


(
    cognee,
    global_context_index_pipeline,
    JsonListChunker,
    get_default_user,
    analyze_turn_for_session_context,
    get_session_manager,
    apply_session_turn_analysis,
) = import_cognee_modules()

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "temp" / "beam_session_distillation_ingestion_runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def configure_quiet_logging() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ("cognee", "cognify", "global_context_index", "session_distillation"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def selected_session_files(conversation_folder: Path, max_sessions: int | None) -> list[Path]:
    files = sorted(path for path in conversation_folder.glob("*.json") if path.is_file())
    if max_sessions is None:
        return files
    return files[: max(0, max_sessions)]


def default_dataset_name(conversation_folder: Path) -> str:
    return f"beam_{conversation_folder.parent.name}_{conversation_folder.name}_{timestamp()}"


def run_dir_for(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    return DEFAULT_RUNS_DIR / dataset_name


def limited_session_copy(
    *,
    source_path: Path,
    run_dir: Path,
    max_turns_per_session: int | None,
) -> dict[str, Any]:
    items = read_json_list(source_path)
    limited_items = items
    if max_turns_per_session is not None:
        limited_items = items[: max(0, max_turns_per_session)]

    target_path = run_dir / "session_inputs" / source_path.name
    write_json(target_path, limited_items)
    item_sizes = [len(str(item).split()) for item in limited_items]
    return {
        "source_path": str(source_path),
        "ingested_path": str(target_path),
        "source_turn_count": len(items),
        "ingested_turn_count": len(limited_items),
        "max_rough_chunk_size": max(item_sizes, default=0),
    }


async def maybe_prune(args: argparse.Namespace) -> None:
    if not args.prune_first:
        return

    print_step("Pruning existing Cognee data/system state before ingestion")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print_step("Prune complete")


async def ensure_database_setup() -> None:
    from cognee.modules.engine.operations.setup import setup

    print_step("Ensuring Cognee database tables exist")
    await setup()


async def build_global_context_index(args: argparse.Namespace, dataset_name: str) -> None:
    user = await get_default_user()
    await global_context_index_pipeline(
        user=user,
        dataset=dataset_name,
        run_in_background=False,
        bucketing_strategy=args.global_context_bucketing_strategy,
        max_bucket_size=args.global_context_max_bucket_size,
        rebuild=args.global_context_rebuild,
    )


async def ingest_session_file(
    *,
    session_index: int,
    session_count: int,
    session_info: dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    ingested_path = session_info["ingested_path"]
    print_step(
        f"Session {session_index}/{session_count}: add {Path(ingested_path).name} "
        f"({session_info['ingested_turn_count']} turns, "
        f"max rough chunk {session_info['max_rough_chunk_size']})"
    )

    add_started = time.monotonic()
    await cognee.add(
        ingested_path,
        dataset_name=dataset_name,
        data_per_batch=args.data_per_batch,
    )
    add_seconds = time.monotonic() - add_started
    print_step(f"Session {session_index}/{session_count}: add done in {add_seconds:.1f}s")

    cognify_started = time.monotonic()
    await cognee.cognify(
        datasets=[dataset_name],
        chunker=JsonListChunker,
        chunk_size=args.chunk_size,
        data_per_batch=args.data_per_batch,
        chunks_per_batch=args.chunks_per_batch,
    )
    cognify_seconds = time.monotonic() - cognify_started
    print_step(f"Session {session_index}/{session_count}: cognify done in {cognify_seconds:.1f}s")

    return {
        "add_seconds": add_seconds,
        "cognify_seconds": cognify_seconds,
        "ingestion_seconds": time.monotonic() - started,
    }


async def analyze_turns_parallel(turns: list[dict[str, str]], concurrency: int) -> list[Any]:
    if concurrency < 1:
        raise ValueError("--candidate-concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)

    async def analyze(index: int) -> Any:
        previous = turns[index - 1] if index > 0 else {}
        async with semaphore:
            return await analyze_turn_for_session_context(
                turns[index]["user"],
                previous_question=previous.get("user"),
                previous_answer=previous.get("assistant"),
            )

    return await asyncio.gather(*(analyze(index) for index in range(len(turns))))


async def harvest_session_candidates(
    *,
    turns: list[dict[str, str]],
    user: Any,
    session_id: str,
    concurrency: int,
) -> dict[str, Any]:
    started = time.monotonic()
    session_manager = get_session_manager()
    user_id = str(user.id)
    await session_manager.delete_session(user_id=user_id, session_id=session_id)

    if not turns:
        return {
            "session_id": session_id,
            "turn_count": 0,
            "context_entry_count": 0,
            "candidate_seconds": time.monotonic() - started,
        }

    print_step(
        f"Session memory {session_id}: analyzing {len(turns)} turns (concurrency={concurrency})"
    )
    analysis_started = time.monotonic()
    analyses = await analyze_turns_parallel(turns, concurrency)
    analysis_seconds = time.monotonic() - analysis_started

    apply_started = time.monotonic()
    previous_qa_id = None
    for turn, analysis in zip(turns, analyses):
        qa_id = await session_manager.add_qa(
            user_id=user_id,
            question=turn["user"],
            context="",
            answer=turn["assistant"],
            session_id=session_id,
        )
        await apply_session_turn_analysis(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            query=turn["user"],
            analysis=analysis,
            previous_qa_id=previous_qa_id,
            served_ids=[],
        )
        previous_qa_id = qa_id

    context_rows = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    context_entry_count = sum(
        1
        for row in context_rows
        if isinstance(row, dict) and row.get("kind", "context") == "context"
    )
    apply_seconds = time.monotonic() - apply_started
    print_step(f"Session memory {session_id}: harvested {context_entry_count} context entries")

    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "context_entry_count": context_entry_count,
        "analysis_seconds": analysis_seconds,
        "apply_seconds": apply_seconds,
        "candidate_seconds": time.monotonic() - started,
    }


async def distill_session(
    *,
    session_id: str,
    dataset_name: str,
    user: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    print_step(f"Session memory {session_id}: distilling into dataset {dataset_name}")
    result = await cognee.session.distill_session(
        session_id,
        dataset=dataset_name,
        user=user,
    )
    seconds = time.monotonic() - started
    payload = distillation_report_payload(result, seconds)
    print_step(
        f"Session memory {session_id}: distill status={payload['status']} "
        f"gated={payload['gated_entry_count']} batches={payload['batch_count']} "
        f"proposed={payload['proposed_lesson_count']} "
        f"accepted={payload['accepted_lesson_count']} "
        f"rejected={payload['rejected_lesson_count']} "
        f"documents={payload['document_count']} in {seconds:.1f}s"
    )
    return payload


async def run_post_ingestion_steps(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    session_id: str,
    user: Any,
    run_dir: Path,
) -> tuple[dict[str, Any], float | None]:
    if args.parallel_post_processing and not args.skip_global_context_index:
        print_step(
            f"Session memory {session_id}: distilling and building global context index in parallel"
        )
        global_started = time.monotonic()
        distill_result, _ = await asyncio.gather(
            distill_session(
                session_id=session_id,
                dataset_name=dataset_name,
                user=user,
            ),
            build_global_context_index(args, dataset_name),
        )
        return distill_result, time.monotonic() - global_started

    distill_result = await distill_session(
        session_id=session_id,
        dataset_name=dataset_name,
        user=user,
    )

    if args.skip_global_context_index:
        return distill_result, None

    print_step(f"Session memory {session_id}: building global context index")
    global_started = time.monotonic()
    await build_global_context_index(args, dataset_name)
    global_context_seconds = time.monotonic() - global_started
    print_step(
        f"Session memory {session_id}: global context index done in {global_context_seconds:.1f}s"
    )
    return distill_result, global_context_seconds


async def process_session_file(
    *,
    session_index: int,
    session_count: int,
    source_path: Path,
    dataset_name: str,
    run_dir: Path,
    user: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    session_info = limited_session_copy(
        source_path=source_path,
        run_dir=run_dir,
        max_turns_per_session=args.max_turns_per_session,
    )
    turns = parse_turns(Path(session_info["ingested_path"]))
    session_id = session_id_for(dataset_name, source_path)

    ingestion_result, candidate_result = await asyncio.gather(
        ingest_session_file(
            session_index=session_index,
            session_count=session_count,
            session_info=session_info,
            dataset_name=dataset_name,
            args=args,
        ),
        harvest_session_candidates(
            turns=turns,
            user=user,
            session_id=session_id,
            concurrency=args.candidate_concurrency,
        ),
    )

    distillation_result, global_context_seconds = await run_post_ingestion_steps(
        args=args,
        dataset_name=dataset_name,
        session_id=session_id,
        user=user,
        run_dir=run_dir,
    )

    return {
        **session_info,
        "session_index": session_index,
        "session_id": session_id,
        "parsed_turn_count": len(turns),
        **ingestion_result,
        "candidate_detection": candidate_result,
        "distillation": distillation_result,
        "global_context_seconds": global_context_seconds,
        "total_seconds": time.monotonic() - started,
    }


def build_report(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    run_dir: Path,
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": utc_now(),
            "conversation_folder": str(args.conversation_folder),
            "dataset_name": dataset_name,
            "run_dir": str(run_dir),
            "max_sessions": args.max_sessions,
            "max_turns_per_session": args.max_turns_per_session,
            "chunk_size": args.chunk_size,
            "data_per_batch": args.data_per_batch,
            "chunks_per_batch": args.chunks_per_batch,
            "candidate_concurrency": args.candidate_concurrency,
            "skip_global_context_index": args.skip_global_context_index,
            "parallel_post_processing": args.parallel_post_processing,
            "global_context_bucketing_strategy": args.global_context_bucketing_strategy,
            "global_context_max_bucket_size": args.global_context_max_bucket_size,
            "global_context_rebuild": args.global_context_rebuild,
            "prune_first": args.prune_first,
        },
        "summary": {
            "session_count": len(sessions),
            "turn_count": sum(session["ingested_turn_count"] for session in sessions),
            "parsed_turn_count": sum(session["parsed_turn_count"] for session in sessions),
            "distilled_lesson_count": sum(
                session["distillation"].get("accepted_lesson_count") or 0 for session in sessions
            ),
            "session_context_entry_count": sum(
                session["candidate_detection"].get("context_entry_count") or 0
                for session in sessions
            ),
            "max_rough_chunk_size": max(
                (session["max_rough_chunk_size"] for session in sessions),
                default=0,
            ),
        },
        "sessions": sessions,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    configure_quiet_logging()
    if not args.conversation_folder.exists():
        raise ValueError(f"Conversation folder does not exist: {args.conversation_folder}")
    if args.max_sessions is not None and args.max_sessions < 0:
        raise ValueError("--max-sessions cannot be negative")
    if args.max_turns_per_session is not None and args.max_turns_per_session < 0:
        raise ValueError("--max-turns-per-session cannot be negative")
    if args.candidate_concurrency < 1:
        raise ValueError("--candidate-concurrency must be at least 1")

    dataset_name = args.dataset_name or default_dataset_name(args.conversation_folder)
    run_dir = run_dir_for(args, dataset_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    session_files = selected_session_files(args.conversation_folder, args.max_sessions)
    session_count = len(session_files)
    print_step(f"Conversation folder: {args.conversation_folder}")
    print_step(f"Dataset: {dataset_name}")
    print_step(f"Run dir: {run_dir}")
    print_step(f"Selected session files: {session_count}")

    await ensure_database_setup()
    await maybe_prune(args)
    if args.prune_first:
        await ensure_database_setup()
    user = await get_default_user()

    session_results: list[dict[str, Any]] = []
    for session_index, source_path in enumerate(session_files, start=1):
        session_result = await process_session_file(
            session_index=session_index,
            session_count=session_count,
            source_path=source_path,
            dataset_name=dataset_name,
            run_dir=run_dir,
            user=user,
            args=args,
        )
        session_results.append(session_result)
        write_json(
            run_dir / "ingestion_report.json",
            build_report(
                args=args,
                dataset_name=dataset_name,
                run_dir=run_dir,
                sessions=session_results,
            ),
        )

    report = build_report(
        args=args,
        dataset_name=dataset_name,
        run_dir=run_dir,
        sessions=session_results,
    )
    write_json(run_dir / "ingestion_report.json", report)
    print_step(
        f"Done: {report['summary']['session_count']} sessions, "
        f"{report['summary']['turn_count']} turns, "
        f"{report['summary']['distilled_lesson_count']} distilled lessons"
    )
    print_step(f"Report: {run_dir / 'ingestion_report.json'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially add+cognify BEAM JSON-list sessions with distillation."
    )
    parser.add_argument("conversation_folder", type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns-per-session", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--data-per-batch", type=int, default=100)
    parser.add_argument("--chunks-per-batch", type=int, default=1000)
    parser.add_argument(
        "--candidate-concurrency",
        type=int,
        default=50,
        help="Max concurrent candidate-analysis LLM calls per session.",
    )
    parser.add_argument(
        "--skip-global-context-index",
        action="store_true",
        help="Skip global context index updates after each distilled session.",
    )
    parser.add_argument(
        "--parallel-post-processing",
        action="store_true",
        help=(
            "Run distillation and global context indexing concurrently. Faster, but the "
            "index may not include the distillate produced by that same session."
        ),
    )
    parser.add_argument("--global-context-bucketing-strategy", default="graph")
    parser.add_argument("--global-context-max-bucket-size", type=int, default=4)
    parser.add_argument("--global-context-rebuild", action="store_true")
    parser.add_argument("--prune-first", action="store_true")
    return parser


def main() -> None:
    try:
        asyncio.run(main_async(build_parser().parse_args()))
    except KeyboardInterrupt:
        print_step("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
