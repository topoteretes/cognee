"""Run BEAM answer/eval sweep against an already-ingested Cognee corpus."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional

# Keep Cognee's normal logging quiet; this script prints its own progress.
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["COGNEE_LOG_FILE"] = "false"
os.environ["COGNEE_CLI_MODE"] = "true"

from cognee.eval_framework.beam.eval.sweep import (  # noqa: E402
    build_beam_eval_params,
    build_registry_base_configs,
    filter_questions_by_type,
    load_and_annotate_questions,
    load_beam_sweep_payload_from_file,
    make_timestamped_output_dir,
    resolve_beam_sweep_config,
)
from cognee.eval_framework.benchmark_adapters.beam_adapter import (  # noqa: E402
    load_beam_row,
    parse_beam_probing_questions,
    truncate_beam_chat_batches,
)
from cognee.eval_framework.reporting.io import write_json  # noqa: E402
from cognee.eval_framework.sweeps.retriever_sweep_runner import (  # noqa: E402
    RetrieverSweepSettings,
    _run_retriever_all_runs,
    validate_retriever_configs,
)

DEFAULT_NUM_RUNS = 1
DEFAULT_MAX_CONCURRENT_QUESTIONS = 4
DEFAULT_ARTIFACT_PREFIX = "beam_existing_ingestion"
RUN_INFO_FILENAME = "run_info.json"
PRIMARY_METRIC_NAME = "beam_rubric"
BEAM_SUPPORTED_SPLITS = ("100K", "500K", "1M", "10M")


def print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def configure_quiet_logging() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ("cognee", "eval_framework", "evaluation", "retrieval"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.1f}s"


def summarize_question_types(questions: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for question in questions:
        question_type = question.get("question_type", "unknown")
        counts[question_type] = counts.get(question_type, 0) + 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def summarize_aggregate_metrics(aggregate_metrics: dict[str, Any]) -> str:
    parts = []
    for metric_name in ("beam_rubric", "kendall_tau"):
        metric = aggregate_metrics.get(metric_name)
        if not isinstance(metric, dict):
            continue
        mean = metric.get("mean")
        if isinstance(mean, (int, float)):
            parts.append(f"{metric_name}={mean:.3f}")
    return ", ".join(parts) if parts else "aggregate metrics written"


@dataclass(frozen=True)
class ExistingIngestionSweepParams:
    split: str
    conversation_index: int
    questions_path: Optional[Path]
    max_batches: Optional[int]
    max_questions: Optional[int]
    retriever_names: Optional[list[str]]
    config_json_path: Path
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


def _write_run_info(
    params: ExistingIngestionSweepParams,
    retriever_configs: list[dict[str, Any]],
) -> Path:
    output_dir = params.sweep.output_dir
    run_info_path = output_dir / RUN_INFO_FILENAME
    run_info = {
        "entrypoint": "cognee/eval_framework/beam/eval/run_sweep.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_cli_args": list(params.cli_args),
        "resolved_run": {
            "split": params.split,
            "conversation_index": params.conversation_index,
            "questions_path": str(params.questions_path) if params.questions_path else None,
            "max_batches": params.max_batches,
            "max_questions": params.max_questions,
            "retrievers": params.retriever_names,
            "config_json_path": str(params.config_json_path),
            "dataset_source": "existing_cognee_ingestion",
            "question_types": params.sweep.question_types,
            "num_runs": params.sweep.num_runs,
            "parallel_runs": params.sweep.parallel_runs,
            "max_concurrent_questions": params.sweep.max_concurrent_questions,
            "primary_metric_name": params.sweep.primary_metric_name,
            "artifact_prefix": params.sweep.artifact_prefix,
            "summary_tags": _serialize_run_value(params.sweep.summary_tags),
            "output_dir": str(output_dir),
        },
        "retriever_configs": [_serialize_run_value(config) for config in retriever_configs],
    }
    write_json(str(run_info_path), run_info)
    return run_info_path


def _load_questions(params: ExistingIngestionSweepParams) -> list[dict[str, Any]]:
    if params.questions_path is not None:
        questions = load_and_annotate_questions(str(params.questions_path))
        filtered_questions = filter_questions_by_type(questions, params.sweep.question_types)
        if params.max_questions is None:
            return filtered_questions
        return filtered_questions[: params.max_questions]

    row = load_beam_row(params.split, params.conversation_index)
    chat_batches = truncate_beam_chat_batches(row["chat"], params.max_batches)
    questions = parse_beam_probing_questions(
        row,
        chat_batches,
        conversation_index=params.conversation_index,
    )

    annotated_questions = []
    for index, question in enumerate(questions):
        annotated = dict(question)
        annotated["question_idx"] = index
        annotated.setdefault(
            "conversation_id",
            row.get("conversation_id", str(params.conversation_index)),
        )
        annotated_questions.append(annotated)

    filtered_questions = filter_questions_by_type(annotated_questions, params.sweep.question_types)
    if params.max_questions is None:
        return filtered_questions
    return filtered_questions[: params.max_questions]


def filter_retriever_configs(
    retriever_configs: list[dict[str, Any]],
    retriever_names: Optional[list[str]],
) -> list[dict[str, Any]]:
    if not retriever_names:
        return retriever_configs

    configs_by_name = {config["name"]: config for config in retriever_configs}
    missing_names = [name for name in retriever_names if name not in configs_by_name]
    if missing_names:
        available_names = ", ".join(sorted(configs_by_name))
        missing = ", ".join(missing_names)
        raise ValueError(f"Unsupported retriever(s): {missing}. Available: {available_names}")

    return [configs_by_name[name] for name in retriever_names]


async def run_retriever_sweep_with_progress(
    *,
    conversation_index: int,
    settings: RetrieverSweepSettings,
    retriever_configs: list[dict[str, Any]],
    base_eval_params: dict[str, Any],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    batch_results = []
    total_retrievers = len(retriever_configs)

    for retriever_index, config in enumerate(retriever_configs, start=1):
        retriever_name = config["name"]
        print_step(
            f"Retriever {retriever_index}/{total_retrievers}: {retriever_name} "
            f"running {settings.num_runs} repeat(s) "
            f"({len(questions)} questions each, concurrency {settings.max_concurrent_questions})"
        )

        retriever_started = time.monotonic()
        retriever_batch_results = await _run_retriever_all_runs(
            conversation_index=conversation_index,
            settings=settings,
            config=config,
            base_eval_params=base_eval_params,
            questions=questions,
        )
        batch_results.extend(retriever_batch_results)

        for batch_result in retriever_batch_results:
            run_number = batch_result["run_idx"] + 1
            print_step(
                f"Retriever {retriever_index}/{total_retrievers}: {retriever_name} "
                f"run {run_number}/{settings.num_runs} done "
                f"({summarize_aggregate_metrics(batch_result['aggregate_metrics'])})"
            )
        print_step(
            f"Retriever {retriever_index}/{total_retrievers}: {retriever_name} "
            f"finished all runs in {format_seconds(time.monotonic() - retriever_started)}"
        )

    raw_results_path = settings.output_dir / f"{settings.artifact_prefix}_raw_batch_results.json"
    write_json(str(raw_results_path), batch_results)
    print_step(f"Raw batch results: {raw_results_path}")
    return batch_results


async def run_existing_ingestion_sweep(
    params: ExistingIngestionSweepParams,
) -> list[dict[str, Any]]:
    configure_quiet_logging()
    started = time.monotonic()
    output_dir = params.sweep.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step("Starting BEAM existing-ingestion answer/eval sweep")
    print_step(f"Split: {params.split}")
    print_step(f"Conversation index: {params.conversation_index}")
    print_step(f"Config JSON: {params.config_json_path}")
    print_step(f"Output dir: {output_dir}")

    payload = load_beam_sweep_payload_from_file(params.config_json_path)
    retriever_configs = resolve_beam_sweep_config(payload, build_registry_base_configs())
    retriever_configs = filter_retriever_configs(retriever_configs, params.retriever_names)
    validate_retriever_configs(retriever_configs)
    run_info_path = _write_run_info(params, retriever_configs)
    print_step(f"Run info: {run_info_path}")
    print_step("Retrievers: " + ", ".join(config["name"] for config in retriever_configs))

    questions = _load_questions(params)
    if not questions:
        raise ValueError(
            "No questions to evaluate after loading/filtering "
            f"(questions_path={params.questions_path}, "
            f"question_types={params.sweep.question_types}, "
            f"max_questions={params.max_questions}). Refusing to run an empty sweep."
        )
    questions_path = (
        output_dir
        / f"{params.sweep.artifact_prefix}_questions_conv{params.conversation_index}.json"
    )
    write_json(str(questions_path), questions)
    print_step(f"Questions: {len(questions)} loaded")
    print_step(f"Question types: {summarize_question_types(questions)}")
    print_step(f"Questions file: {questions_path}")

    eval_params = build_beam_eval_params(
        conversation_index=params.conversation_index,
        output_dir=output_dir,
        answering_questions=False,
        qa_engine="existing_ingestion_sweep",
    )
    eval_params["building_corpus_from_scratch"] = False
    eval_params["questions_path"] = str(questions_path)

    batch_results = await run_retriever_sweep_with_progress(
        conversation_index=params.conversation_index,
        settings=params.sweep,
        retriever_configs=retriever_configs,
        base_eval_params=eval_params,
        questions=questions,
    )

    print_step(f"Done in {format_seconds(time.monotonic() - started)}")
    return batch_results


def _parse_csv(value: str) -> Optional[list[str]]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _parse_args() -> ExistingIngestionSweepParams:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="100K",
        choices=BEAM_SUPPORTED_SPLITS,
        help="BEAM split whose questions should be evaluated. Use --questions-path for 10M.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help=(
            "Optional pre-exported questions JSON. Required for split=10M; when provided, "
            "--split is only recorded as run metadata."
        ),
    )
    parser.add_argument(
        "--conversation-index",
        type=int,
        default=0,
        help="Conversation index whose probing questions should be evaluated.",
    )
    parser.add_argument(
        "--config-json-path",
        type=Path,
        required=True,
        help="Required sweep config JSON. No default BEAM sweep is bundled.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of answer-generation runs per retriever config.",
    )
    parser.add_argument(
        "--parallel-runs",
        action="store_true",
        help="Run repeats for each retriever in parallel.",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_QUESTIONS,
        help="Maximum number of questions to answer in parallel per retriever.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="",
        help="Comma-separated question types to evaluate. Empty means all.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Use questions as if only the first N batches were present. 0 means all.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Evaluate only the first N filtered questions. 0 means all.",
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="",
        help="Comma-separated resolved retriever variant names to run. Empty means all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for answer/eval artifacts. Defaults to temp timestamped dir.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=DEFAULT_ARTIFACT_PREFIX,
        help="Filename prefix for generated artifacts.",
    )

    args = parser.parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1")
    if args.max_concurrent_questions < 1:
        raise ValueError("--max-concurrent-questions must be at least 1")
    if not args.config_json_path.is_file():
        raise FileNotFoundError(f"Missing sweep config JSON: {args.config_json_path}")
    if args.questions_path is not None and not args.questions_path.is_file():
        raise FileNotFoundError(f"Missing questions JSON: {args.questions_path}")
    if args.split == "10M" and args.questions_path is None:
        raise ValueError("--split 10M requires --questions-path.")
    if args.questions_path is not None and args.max_batches > 0:
        raise ValueError("--max-batches cannot be used with --questions-path.")

    output_dir = args.output_dir or make_timestamped_output_dir(prefix=args.artifact_prefix)
    question_types = _parse_csv(args.question_types)
    max_batches = args.max_batches if args.max_batches > 0 else None
    max_questions = args.max_questions if args.max_questions > 0 else None
    retriever_names = _parse_csv(args.retrievers)

    sweep = RetrieverSweepSettings(
        output_dir=output_dir,
        num_runs=args.num_runs,
        parallel_runs=args.parallel_runs,
        max_concurrent_questions=args.max_concurrent_questions,
        question_types=question_types,
        primary_metric_name=PRIMARY_METRIC_NAME,
        artifact_prefix=args.artifact_prefix,
        summary_tags={
            "split": args.split,
            "conversation_index": args.conversation_index,
            "questions_path": str(args.questions_path) if args.questions_path else None,
            "config_json_path": str(args.config_json_path),
        },
    )
    return ExistingIngestionSweepParams(
        split=args.split,
        conversation_index=args.conversation_index,
        questions_path=args.questions_path,
        max_batches=max_batches,
        max_questions=max_questions,
        retriever_names=retriever_names,
        config_json_path=args.config_json_path,
        sweep=sweep,
        cli_args=tuple(sys.argv[1:]),
    )


def main() -> None:
    asyncio.run(run_existing_ingestion_sweep(_parse_args()))


if __name__ == "__main__":
    main()
