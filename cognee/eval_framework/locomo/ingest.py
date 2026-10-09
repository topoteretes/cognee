"""Ingest ONE LoCoMo conversation into cognee (meant to run in its own process/roots).

The conversation becomes a short overview document (speakers and session timeline) plus one
document per window of consecutive turns (``--window-turns``, default 6, each with a dated header;
see ``preprocess.py``). All of them go through a single ``remember(documents, dataset_name=...)``
= ``add`` + ``cognify`` + ``improve()``, the way any user builds a memory. ``--chunk-size``
(default 4096 tokens) is above the largest window, so every document stays one chunk; the
ingestion report checks ``chunk_count == document_count``.

Usage (normally driven by ``run_locomo_eval.py``)::

    uv run python -m cognee.eval_framework.locomo.ingest --conversation-index 1 \
        --run-dir temp/locomo_runs/<run>/conv_01
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Env that must be in place before cognee is imported (settings are read at import time).
os.environ.setdefault("CACHING", "true")
os.environ.setdefault("AUTO_FEEDBACK", "false")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("COGNEE_LOG_FILE", "false")

from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter
from cognee.eval_framework.locomo.preprocess import (
    DEFAULT_WINDOW_TURNS,
    build_conversation_windows,
    conversation_overview_text,
    dataset_name_for,
)
from cognee.eval_framework.reporting.io import write_json
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Max tokens per chunk. The largest LoCoMo session is ~1.6k tokens (cl100k), so every window
# stays one chunk; fixed so it does not depend on the embedding model's limit (the
# auto-calculated default).
DEFAULT_CHUNK_SIZE = 4096


def print_step(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def ingest_conversation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "conversation_index": args.conversation_index,
        "llm_model": os.getenv("LLM_MODEL"),
        "window_turns": args.window_turns,
        "chunk_size": args.chunk_size,
        "max_sessions": args.max_sessions,
        "data_root_directory": os.getenv("DATA_ROOT_DIRECTORY"),
        "system_root_directory": os.getenv("SYSTEM_ROOT_DIRECTORY"),
    }

    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config

    embedding_config = get_embedding_config()
    report["embedding_provider"] = embedding_config.embedding_provider
    report["embedding_model"] = embedding_config.embedding_model

    import cognee
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.users.methods import get_default_user

    adapter = LocomoAdapter(max_sessions=args.max_sessions, data_path=args.data_path)
    conversation = adapter.load_conversation(args.conversation_index)
    dataset_name = dataset_name_for(conversation)
    report["dataset_name"] = dataset_name
    report["sample_id"] = conversation.sample_id

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.prune_first:
        print_step("Pruning existing cognee state")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    await setup()
    user = await get_default_user()

    windows = build_conversation_windows(conversation, args.window_turns)
    texts = [conversation_overview_text(conversation)] + [window.text for window in windows]
    report["window_count"] = len(windows)
    report["document_count"] = len(texts)
    report["window_map"] = [
        {
            "session": window.session_index,
            "part": window.part,
            "parts": window.parts,
            "first_dia_id": window.first_dia_id,
            "last_dia_id": window.last_dia_id,
            "words": window.word_count,
        }
        for window in windows
    ]

    print_step(
        f"{conversation.sample_id}: remember() {len(texts)} documents ({len(windows)} windows of "
        f"{args.window_turns} turns + overview) into dataset {dataset_name}"
    )
    remember_started = time.monotonic()
    await cognee.remember(texts, dataset_name=dataset_name, user=user, chunk_size=args.chunk_size)
    report["remember_seconds"] = time.monotonic() - remember_started
    print_step(f"{conversation.sample_id}: remember done in {report['remember_seconds']:.1f}s")

    # Graph size, for the report; every document must stay a single chunk.
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
        report["graph_nodes"] = len(nodes)
        report["graph_edges"] = len(edges)
        chunk_count = sum(1 for _, props in nodes if props.get("type") == "DocumentChunk")
        report["chunk_count"] = chunk_count
        report["one_chunk_per_document"] = chunk_count == report["document_count"]
        if chunk_count != report["document_count"]:
            print_step(
                f"WARNING: {chunk_count} chunks for {report['document_count']} documents "
                f"(chunk_size={args.chunk_size})"
            )
    except Exception as error:
        logger.exception("Could not read the graph size for the ingestion report")
        report["graph_size_error"] = str(error)

    report["total_seconds"] = time.monotonic() - started
    report["finished_at"] = utc_now()
    report["status"] = "completed"
    write_json(str(run_dir / "ingestion_report.json"), report)
    print_step(
        f"{conversation.sample_id}: ingestion complete in {report['total_seconds'] / 60:.1f} min "
        f"(nodes={report.get('graph_nodes')}, edges={report.get('graph_edges')})"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-index", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--window-turns", type=int, default=DEFAULT_WINDOW_TURNS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-sessions", type=int, default=None)
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
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
