"""Run BEAM-10M evaluation with preprocessed incremental ingestion.

Usage:
    uv run python cognee/eval_framework/run_beam_10m_preprocessed_eval.py --conversation-index 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from cognee.eval_framework.beam.preprocessed_10m_runtime import (
    DEFAULT_PREPROCESSED_10M_CHUNKS_PER_BATCH,
    DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    run_beam_10m_preprocessed_conversation,
)
from cognee.eval_framework.beam.runtime import make_timestamped_output_dir
from cognee.eval_framework.reporting.io import write_json
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _parse_csv(value: str) -> Optional[list[str]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _build_run_info(
    *,
    output_dir: Path,
    conversation_index: int,
    plans: Optional[list[str]],
    question_types: Optional[list[str]],
    max_batches_per_plan: Optional[int],
    docs_per_add_batch: int,
    preprocessed_max_chunk_size: int,
    cognify_chunk_size: int,
    chunks_per_batch: int,
) -> None:
    payload: dict[str, Any] = {
        "entrypoint": "cognee/eval_framework/run_beam_10m_preprocessed_eval.py",
        "conversation_index": conversation_index,
        "plans": plans,
        "question_types": question_types,
        "max_batches_per_plan": max_batches_per_plan,
        "docs_per_add_batch": docs_per_add_batch,
        "preprocessed_max_chunk_size": preprocessed_max_chunk_size,
        "cognify_chunk_size": cognify_chunk_size,
        "chunks_per_batch": chunks_per_batch,
        "output_dir": str(output_dir),
        "benchmark": "BEAM-10M",
        "ingestion_mode": "batched_preprocessed_10m",
        "chunker": "TextChunker",
        "adapter": "BEAM10MPreprocessedAdapter",
    }
    write_json(str(output_dir / "run_info.json"), payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversation-index",
        type=int,
        default=0,
        help="BEAM-10M conversation index to run.",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default="",
        help="Comma-separated list of plans to ingest (empty = all plans).",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="",
        help="Comma-separated list of question types to answer/evaluate (empty = all).",
    )
    parser.add_argument(
        "--max-batches-per-plan",
        type=int,
        default=0,
        help="Limit the number of native batches ingested per plan (0 = all).",
    )
    parser.add_argument(
        "--docs-per-add-batch",
        type=int,
        default=DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
        help="Number of preprocessed documents to add before each cognify call.",
    )
    parser.add_argument(
        "--preprocessed-max-chunk-size",
        type=int,
        default=DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
        help="Maximum estimated token size for each preprocessed document.",
    )
    parser.add_argument(
        "--cognify-chunk-size",
        type=int,
        default=DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
        help="Chunk size passed to TextChunker during cognify.",
    )
    parser.add_argument(
        "--chunks-per-batch",
        type=int,
        default=DEFAULT_PREPROCESSED_10M_CHUNKS_PER_BATCH,
        help="Chunk batches passed to cognify.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory for artifacts (default = temp timestamped folder).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    plans = _parse_csv(args.plans)
    question_types = _parse_csv(args.question_types)
    max_batches_per_plan = args.max_batches_per_plan or None

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else make_timestamped_output_dir(prefix="beam10m_preprocessed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _build_run_info(
        output_dir=output_dir,
        conversation_index=args.conversation_index,
        plans=plans,
        question_types=question_types,
        max_batches_per_plan=max_batches_per_plan,
        docs_per_add_batch=args.docs_per_add_batch,
        preprocessed_max_chunk_size=args.preprocessed_max_chunk_size,
        cognify_chunk_size=args.cognify_chunk_size,
        chunks_per_batch=args.chunks_per_batch,
    )

    logger.info("Writing BEAM-10M preprocessed artifacts to %s", output_dir)
    aggregate = await run_beam_10m_preprocessed_conversation(
        conversation_index=args.conversation_index,
        output_dir=output_dir,
        plans=plans,
        max_batches_per_plan=max_batches_per_plan,
        question_types=question_types,
        docs_per_add_batch=args.docs_per_add_batch,
        preprocessed_max_chunk_size=args.preprocessed_max_chunk_size,
        cognify_chunk_size=args.cognify_chunk_size,
        chunks_per_batch=args.chunks_per_batch,
    )

    if not aggregate:
        logger.warning("No aggregate metrics produced for this run")
        return

    logger.info("=== BEAM-10M preprocessed results ===")
    for metric_name, score in aggregate.items():
        if isinstance(score, (int, float)):
            logger.info("  %s: %.3f", metric_name, score)
        else:
            logger.info("  %s: %s", metric_name, score)

    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
