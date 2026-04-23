"""Run a non-Modal BEAM multi-retriever sweep locally (CLI + BEAM corpus wiring)."""

import argparse
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

from cognee.eval_framework.beam.presets import (
    BEAM_DEFAULT_SPLIT,
    apply_beam_prompt_policy,
    get_default_beam_sweep_configs,
)
from cognee.eval_framework.beam.runtime import make_timestamped_output_dir, prepare_beam_questions
from cognee.eval_framework.reporting.aggregations import (
    build_best_retriever_by_question_type_report,
    build_combined_summary,
    build_empty_conversation_summary,
)
from cognee.eval_framework.reporting.io import write_json
from cognee.eval_framework.sweeps.retriever_sweep_runner import (
    RetrieverSweepSettings,
    run_retriever_sweep_for_questions,
    validate_retriever_configs,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

NUM_CONVERSATIONS = 20
NUM_RUNS = 1
BEAM_MAX_BATCHES: Optional[int] = None
MAX_CONCURRENT_QUESTIONS = 10
BEAM_COMBINED_SUMMARY = "beam_sweep_combined.json"
BEAM_BEST_BY_QUESTION_TYPE_SUMMARY = "beam_sweep_best_by_question_type.json"
BEAM_RUBRIC_METRIC = "beam_rubric"
BEAM_ARTIFACT_PREFIX = "beam"
BEAM_RUN_INFO_FILENAME = "run_info.json"


@dataclass(frozen=True)
class BeamSweepRunParams:
    """CLI-derived BEAM sweep: dataset selection plus shared retriever sweep settings."""

    split: str
    num_conversations: int
    max_batches: Optional[int]
    conversation_json: Optional[str]
    sweep: RetrieverSweepSettings
    cli_args: tuple[str, ...]


def _serialize_run_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_run_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_run_value(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    return str(value)


def _build_run_info_payload(
    params: BeamSweepRunParams,
    retriever_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    sweep = params.sweep
    conversation_json = (
        str(Path(params.conversation_json).resolve()) if params.conversation_json else None
    )

    return {
        "entrypoint": "cognee/eval_framework/run_beam_retriever_sweep.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_cli_args": list(params.cli_args),
        "resolved_run": {
            "split": params.split,
            "num_conversations": params.num_conversations,
            "max_batches": params.max_batches,
            "conversation_json": conversation_json,
            "dataset_source": "local_json" if params.conversation_json else "beam_huggingface",
            "chunker": "ConversationChunker",
            "question_types": sweep.question_types,
            "num_runs": sweep.num_runs,
            "max_concurrent_questions": sweep.max_concurrent_questions,
            "primary_metric_name": sweep.primary_metric_name,
            "artifact_prefix": sweep.artifact_prefix,
            "combined_summary_filename": sweep.combined_summary_filename,
            "summary_tags": _serialize_run_value(sweep.summary_tags),
            "output_dir": str(sweep.output_dir),
        },
        "retriever_configs": [_serialize_run_value(config) for config in retriever_configs],
    }


def _write_run_info(
    output_dir: Path,
    params: BeamSweepRunParams,
    retriever_configs: list[dict[str, Any]],
) -> None:
    run_info_path = output_dir / BEAM_RUN_INFO_FILENAME
    run_info = _build_run_info_payload(params, retriever_configs)
    write_json(str(run_info_path), run_info)
    logger.info("Wrote run manifest to %s", run_info_path)


async def _run_beam_conversation(
    conversation_index: int,
    params: BeamSweepRunParams,
    retriever_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    sweep = params.sweep

    logger.info(
        "=== BEAM retriever sweep: conversation %s / %s ===",
        conversation_index,
        params.num_conversations - 1,
    )

    eval_params, questions = await prepare_beam_questions(
        conversation_index=conversation_index,
        output_dir=sweep.output_dir,
        split=params.split,
        max_batches=params.max_batches,
        question_types=sweep.question_types,
        conversation_json=params.conversation_json,
    )

    retriever_names = [config["name"] for config in retriever_configs]
    if not questions:
        logger.warning("[conv %s] No questions to evaluate after filtering", conversation_index)
        summary = build_empty_conversation_summary(conversation_index, sweep, retriever_names)
        summary_path = str(
            sweep.output_dir / f"{sweep.artifact_prefix}_sweep_conv{conversation_index}.json"
        )
        write_json(summary_path, summary)
        return summary

    return await run_retriever_sweep_for_questions(
        conversation_index=conversation_index,
        settings=sweep,
        retriever_configs=retriever_configs,
        base_eval_params=eval_params,
        questions=questions,
    )


async def run_beam_sweep(params: BeamSweepRunParams) -> dict[str, Any]:
    sweep = params.sweep
    sweep.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing sweep artifacts to %s", sweep.output_dir)

    retriever_configs = apply_beam_prompt_policy(get_default_beam_sweep_configs(params.split))
    validate_retriever_configs(retriever_configs)
    _write_run_info(sweep.output_dir, params, retriever_configs)

    conversation_summaries = []
    for conversation_index in range(params.num_conversations):
        conversation_summary = await _run_beam_conversation(
            conversation_index=conversation_index,
            params=params,
            retriever_configs=retriever_configs,
        )
        conversation_summaries.append(conversation_summary)

    combined_summary = build_combined_summary(
        conversation_summaries=conversation_summaries,
        settings=sweep,
        retriever_configs=retriever_configs,
        num_requested_conversations=params.num_conversations,
    )
    combined_summary_path = str(sweep.output_dir / sweep.combined_summary_filename)
    write_json(combined_summary_path, combined_summary)
    logger.info("Wrote combined summary to %s", combined_summary_path)

    best_by_question_type_report = build_best_retriever_by_question_type_report(
        conversation_summaries=conversation_summaries,
        settings=sweep,
        retriever_configs=retriever_configs,
    )
    best_by_question_type_path = str(sweep.output_dir / BEAM_BEST_BY_QUESTION_TYPE_SUMMARY)
    write_json(best_by_question_type_path, best_by_question_type_report)
    logger.info(
        "Wrote best-by-question-type report to %s",
        best_by_question_type_path,
    )
    return combined_summary


def _parse_args() -> BeamSweepRunParams:
    parser = argparse.ArgumentParser(
        description=__doc__ + "\n\nUsage:\n"
        "    uv run python cognee/eval_framework/run_beam_retriever_sweep.py\n"
        "    uv run python cognee/eval_framework/run_beam_retriever_sweep.py --num-conversations 1\n"
        "    uv run python cognee/eval_framework/run_beam_retriever_sweep.py --question-types summarization\n"
        "    uv run python cognee/eval_framework/run_beam_retriever_sweep.py "
        "--conversation-json examples/demos/beam_toy_conversation.json\n"
    )
    parser.add_argument(
        "--split",
        default=BEAM_DEFAULT_SPLIT,
        choices=["100K", "500K", "1M"],
        help="BEAM dataset split to evaluate.",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=NUM_CONVERSATIONS,
        help="Number of conversations to evaluate.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help="Number of answer-generation runs per retriever config.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit the number of BEAM conversation batches used for ingestion (0 = all).",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=MAX_CONCURRENT_QUESTIONS,
        help="Maximum number of questions to answer in parallel per batch.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="",
        help="Comma-separated list of question types to evaluate (empty = all).",
    )
    parser.add_argument(
        "--conversation-json",
        type=str,
        default="",
        help="Path to a local BEAM-shaped JSON file with `corpus` and `probing_questions`.",
    )

    args = parser.parse_args()
    question_types = [
        item.strip() for item in args.question_types.split(",") if item.strip()
    ] or None
    conversation_json = args.conversation_json.strip() or None

    if args.num_conversations < 1:
        raise ValueError("--num-conversations must be at least 1")
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1")
    if args.max_concurrent_questions < 1:
        raise ValueError("--max-concurrent-questions must be at least 1")
    if conversation_json and args.num_conversations != 1:
        raise ValueError("--conversation-json currently supports exactly one conversation")
    if conversation_json and args.max_batches > 0:
        raise ValueError("--max-batches cannot be used together with --conversation-json")
    if conversation_json and not Path(conversation_json).is_file():
        raise FileNotFoundError(f"Conversation JSON not found: {conversation_json}")

    output_dir = make_timestamped_output_dir(prefix=BEAM_ARTIFACT_PREFIX)
    sweep = RetrieverSweepSettings(
        output_dir=output_dir,
        num_runs=args.num_runs,
        max_concurrent_questions=args.max_concurrent_questions,
        question_types=question_types,
        primary_metric_name=BEAM_RUBRIC_METRIC,
        artifact_prefix=BEAM_ARTIFACT_PREFIX,
        combined_summary_filename=BEAM_COMBINED_SUMMARY,
        summary_tags={"split": args.split},
    )

    return BeamSweepRunParams(
        split=args.split,
        num_conversations=args.num_conversations,
        max_batches=(args.max_batches if args.max_batches > 0 else None),
        conversation_json=conversation_json,
        sweep=sweep,
        cli_args=tuple(sys.argv[1:]),
    )


def main() -> None:
    params = _parse_args()
    asyncio.run(run_beam_sweep(params))


if __name__ == "__main__":
    main()
