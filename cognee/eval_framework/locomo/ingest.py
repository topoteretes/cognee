"""Ingest ONE LoCoMo conversation into cognee (meant to run in its own process/roots).

The conversation becomes a short overview document (speakers and session timeline) plus one
document per window of consecutive turns (``--window-turns``, default 6, each with a dated header;
see ``preprocess.py``). All of them go through a single ``remember(documents, dataset_name=...)``
= ``add`` + ``cognify`` + ``improve()``, the way any user builds a memory. ``--chunk-size``
(default 4096 tokens) is above the largest window, so every document stays one chunk; the
ingestion report checks ``chunk_count == document_count``.

Four optional changes to how the memory is built (all off by default; together they are the
top-k 5 configuration described in the README):

- ``--resolve-dates``: relative time expressions in the windows get the absolute date they
  refer to before ingestion (``dates.py``).
- ``--extraction-prompt FILE``: graph extraction prompt passed to ``remember(custom_prompt=...)``
  instead of cognee's default.
- ``--summary-prompt FILE``: chunk summary prompt used instead of cognee's
  ``summarize_content.txt`` for the duration of the ingestion.
- ``--global-context-index``: cognee's global context index is built over the memory after
  ``remember()``.

Usage (normally driven by ``run_locomo_eval.py``)::

    uv run python -m cognee.eval_framework.locomo.ingest --conversation-index 1 \
        --run-dir temp/locomo_runs/<run>/conv_01
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


SUMMARY_PROMPT_NAME = "summarize_content.txt"
# The package re-exports the function under the module's name, so the module is looked up by
# its dotted path rather than imported by name.
SUMMARY_MODULE = "cognee.infrastructure.llm.extraction.extract_summary"


class summary_prompt_override:
    """Use ``prompt_text`` as the chunk summary prompt while the block runs.

    cognee reads that prompt by file name (``read_query_prompt("summarize_content.txt")`` in
    ``SUMMARY_MODULE``) and has no setting to point it at another file, so for the duration of
    the ingestion this answers that one lookup with our text; every other prompt is read as
    usual. ``calls`` counts how often the summary prompt was requested: the caller checks it is
    not zero, so a cognee change in how the prompt is loaded fails the run instead of silently
    summarizing with the default prompt.
    """

    def __init__(self, prompt_text: str) -> None:
        self.prompt_text = prompt_text
        self.calls = 0

    def __enter__(self):
        module = importlib.import_module(SUMMARY_MODULE)
        original = getattr(module, "read_query_prompt", None)
        if not callable(original):
            raise RuntimeError(
                "cognee no longer loads the chunk summary prompt through "
                "extract_summary.read_query_prompt; update summary_prompt_override"
            )
        self._module, self._original = module, original

        def read_query_prompt(name: str, *args: Any, **kwargs: Any) -> str:
            if name == SUMMARY_PROMPT_NAME:
                self.calls += 1
                return self.prompt_text
            return original(name, *args, **kwargs)

        module.read_query_prompt = read_query_prompt
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._module.read_query_prompt = self._original


async def ingest_conversation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "conversation_index": args.conversation_index,
        "llm_model": os.getenv("LLM_MODEL"),
        "window_turns": args.window_turns,
        "chunk_size": args.chunk_size,
        "max_sessions": args.max_sessions,
        "resolve_dates": args.resolve_dates,
        "extraction_prompt": str(args.extraction_prompt) if args.extraction_prompt else None,
        "summary_prompt": str(args.summary_prompt) if args.summary_prompt else None,
        "global_context_index": args.global_context_index,
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
    report["window_count"] = len(windows)
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

    if args.resolve_dates:
        from cognee.eval_framework.locomo.dates import resolve_window_dates
        from cognee.infrastructure.llm.config import get_llm_config

        model = os.getenv("LLM_MODEL") or get_llm_config().llm_model
        print_step(f"{conversation.sample_id}: resolving relative dates in {len(windows)} windows")
        dates_started = time.monotonic()
        windows, stats = await resolve_window_dates(
            conversation,
            windows,
            model=model,
            api_key=get_llm_config().llm_api_key,
            cache_path=run_dir / "date_annotations.json",
        )
        stats["seconds"] = time.monotonic() - dates_started
        report["resolved_dates"] = stats
        print_step(
            f"{conversation.sample_id}: dates resolved in {stats['seconds']:.1f}s "
            f"(annotated {stats['annotated']}, unchanged {stats['unchanged']}, "
            f"rejected {stats['rejected']}, errors {stats['error']}, reused {stats['reused']})"
        )

    texts = [conversation_overview_text(conversation)] + [window.text for window in windows]
    report["document_count"] = len(texts)

    remember_kwargs: dict[str, Any] = {}
    if args.extraction_prompt:
        remember_kwargs["custom_prompt"] = args.extraction_prompt.read_text(encoding="utf-8")
    summary_override = (
        summary_prompt_override(args.summary_prompt.read_text(encoding="utf-8"))
        if args.summary_prompt
        else None
    )

    print_step(
        f"{conversation.sample_id}: remember() {len(texts)} documents ({len(windows)} windows of "
        f"{args.window_turns} turns + overview) into dataset {dataset_name}"
    )
    remember_started = time.monotonic()
    with summary_override or nullcontext():
        await cognee.remember(
            texts,
            dataset_name=dataset_name,
            user=user,
            chunk_size=args.chunk_size,
            **remember_kwargs,
        )
    report["remember_seconds"] = time.monotonic() - remember_started
    print_step(f"{conversation.sample_id}: remember done in {report['remember_seconds']:.1f}s")
    if summary_override is not None:
        report["summary_prompt_calls"] = summary_override.calls
        if summary_override.calls == 0:
            raise RuntimeError(
                "the chunk summary prompt override was never used: cognee did not read "
                f"{SUMMARY_PROMPT_NAME} during remember(), so the summaries were not built with "
                f"{args.summary_prompt}"
            )

    if args.global_context_index:
        from cognee.memify_pipelines.global_context_index import global_context_index_pipeline

        print_step(f"{conversation.sample_id}: building the global context index")
        index_started = time.monotonic()
        await global_context_index_pipeline(user=user, dataset=dataset_name)
        report["global_context_index_seconds"] = time.monotonic() - index_started
        print_step(
            f"{conversation.sample_id}: global context index built in "
            f"{report['global_context_index_seconds']:.1f}s"
        )

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
    parser.add_argument(
        "--resolve-dates",
        action="store_true",
        help="annotate relative time expressions with absolute dates before ingestion",
    )
    parser.add_argument(
        "--extraction-prompt",
        type=Path,
        default=None,
        help="graph extraction prompt file, passed to remember(custom_prompt=...)",
    )
    parser.add_argument(
        "--summary-prompt",
        type=Path,
        default=None,
        help="chunk summary prompt file, used instead of cognee's summarize_content.txt",
    )
    parser.add_argument(
        "--global-context-index",
        action="store_true",
        help="build cognee's global context index over the memory after remember()",
    )
    args = parser.parse_args()
    for option in ("extraction_prompt", "summary_prompt"):
        path = getattr(args, option)
        if path is not None and not path.is_file():
            parser.error(f"--{option.replace('_', '-')}: no such file: {path}")
    return args


def _apply_env_defaults() -> None:
    """Benchmark settings for a standalone run, set before cognee reads its (cached) config.

    Only the entry point applies them: setting them at import time would change the settings of
    every process that imports this module, the test suite included, and turn off access control
    for it. ``run_locomo_eval.py`` passes the same values to its child processes explicitly.
    """
    os.environ.setdefault("CACHING", "true")
    os.environ.setdefault("AUTO_FEEDBACK", "false")
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")


def main() -> None:
    _apply_env_defaults()
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
