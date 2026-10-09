"""Answer and score one LoCoMo conversation's questions against an ingested memory.

Runs in the same DATA/SYSTEM roots the ingestion used (one process per conversation).
Reuses the benchmark-agnostic retriever sweep runner: every retriever variant in the sweep
config answers every question, ``LocomoEval`` scores with token-F1 and the LLM judge, and
the artifacts land next to the ingestion report.

Usage (normally driven by ``run_locomo_eval.py``)::

    uv run python -m cognee.eval_framework.locomo.answer --conversation-index 0 \
        --run-dir temp/locomo_runs/<run>/conv_00 \
        --config-json-path cognee/eval_framework/locomo/configs/locomo_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("CACHING", "true")
# No per-turn analysis while answering: it is one extra LLM call per question and the
# retrievers here run without a session_id anyway.
os.environ.setdefault("AUTO_FEEDBACK", "false")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("COGNEE_LOG_FILE", "false")

from cognee.eval_framework.beam.eval.sweep import (
    build_registry_base_configs,
    load_beam_sweep_payload_from_file,
    resolve_beam_sweep_config,
)
from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.locomo.model_registry import ensure_model_registered
from cognee.eval_framework.reporting.io import write_json
from cognee.eval_framework.sweeps.retriever_sweep_runner import (
    RetrieverSweepSettings,
    run_retriever_sweep_for_questions,
)

ARTIFACT_PREFIX = "locomo"


def print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def build_eval_params(output_dir: Path, conversation_index: int) -> dict[str, Any]:
    return EvalConfig(
        benchmark="LoCoMo",
        building_corpus_from_scratch=False,
        number_of_samples_in_corpus=0,
        qa_engine="existing_ingestion_sweep",
        answering_questions=True,
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="LocomoEval",
        evaluation_metrics=["f1", "llm_judge"],
        calculate_metrics=True,
        dashboard=False,
        questions_path=str(
            output_dir / f"{ARTIFACT_PREFIX}_questions_conv{conversation_index}.json"
        ),
        answers_path=str(output_dir / f"{ARTIFACT_PREFIX}_answers_conv{conversation_index}.json"),
        metrics_path=str(output_dir / f"{ARTIFACT_PREFIX}_metrics_conv{conversation_index}.json"),
        aggregate_metrics_path=str(
            output_dir / f"{ARTIFACT_PREFIX}_aggregate_metrics_conv{conversation_index}.json"
        ),
        dashboard_path=str(
            output_dir / f"{ARTIFACT_PREFIX}_dashboard_conv{conversation_index}.html"
        ),
    ).to_dict()


def select_questions(
    questions: list[dict[str, Any]],
    *,
    question_types: list[str] | None,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    selected = questions
    if question_types:
        wanted = set(question_types)
        selected = [q for q in selected if q.get("question_type") in wanted]
    if max_questions:
        # Keep the category mix when truncating: round-robin over types.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for question in selected:
            by_type.setdefault(question["question_type"], []).append(question)
        picked: list[dict[str, Any]] = []
        while len(picked) < max_questions and any(by_type.values()):
            for bucket in by_type.values():
                if bucket and len(picked) < max_questions:
                    picked.append(bucket.pop(0))
        selected = sorted(picked, key=lambda q: q["question_idx"])
    return selected


async def answer_conversation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if os.getenv("LLM_MODEL"):
        ensure_model_registered(os.environ["LLM_MODEL"])

    adapter = LocomoAdapter(
        conversation_index=args.conversation_index,
        max_sessions=args.max_sessions,
        include_adversarial=not args.drop_adversarial,
        data_path=args.data_path,
    )
    conversation = adapter.load_conversation(args.conversation_index)
    questions = adapter.questions_for(conversation, load_golden_context=True)
    questions = select_questions(
        questions,
        question_types=args.question_types,
        max_questions=args.max_questions,
    )
    if not questions:
        raise RuntimeError("No questions selected")

    output_dir: Path = args.run_dir / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_params = build_eval_params(output_dir, args.conversation_index)
    write_json(eval_params["questions_path"], questions)

    payload = load_beam_sweep_payload_from_file(args.config_json_path)
    configs = resolve_beam_sweep_config(payload, build_registry_base_configs())
    if args.retrievers:
        wanted = set(args.retrievers)
        configs = [config for config in configs if config["name"] in wanted]
        missing = wanted - {config["name"] for config in configs}
        if missing:
            raise ValueError(f"Unknown retriever variant(s): {', '.join(sorted(missing))}")

    type_counts: dict[str, int] = {}
    for question in questions:
        type_counts[question["question_type"]] = type_counts.get(question["question_type"], 0) + 1
    print_step(
        f"{conversation.sample_id}: {len(questions)} questions {json.dumps(type_counts)} x "
        f"{len(configs)} retriever(s) x {args.num_runs} run(s); "
        f"answer model={os.getenv('LLM_MODEL')} judge={os.getenv('LOCOMO_JUDGE_MODEL')}"
    )

    settings = RetrieverSweepSettings(
        output_dir=output_dir,
        num_runs=args.num_runs,
        parallel_runs=False,
        max_concurrent_questions=args.max_concurrent_questions,
        artifact_prefix=ARTIFACT_PREFIX,
        summary_tags={"benchmark": "LoCoMo", "conversation": conversation.sample_id},
    )
    batch_results = await run_retriever_sweep_for_questions(
        conversation_index=args.conversation_index,
        settings=settings,
        retriever_configs=configs,
        base_eval_params=eval_params,
        questions=questions,
    )

    summary = {
        "conversation_index": args.conversation_index,
        "sample_id": conversation.sample_id,
        "question_count": len(questions),
        "question_type_counts": type_counts,
        "answer_model": os.getenv("LLM_MODEL"),
        "judge_model": os.getenv("LOCOMO_JUDGE_MODEL"),
        "retrievers": [config["name"] for config in configs],
        "num_runs": args.num_runs,
        "batches": [
            {
                "retriever_name": batch["retriever_name"],
                "run_idx": batch["run_idx"],
                "metrics_path": batch["metrics_path"],
                "aggregate_metrics": batch["aggregate_metrics"],
            }
            for batch in batch_results
        ],
        "total_seconds": time.monotonic() - started,
        "status": "completed",
    }
    write_json(
        str(output_dir / f"{ARTIFACT_PREFIX}_qa_summary_conv{args.conversation_index}.json"),
        summary,
    )
    for batch in batch_results:
        aggregate = batch["aggregate_metrics"] or {}
        line = ", ".join(
            f"{metric}={values.get('mean'):.3f}"
            for metric, values in aggregate.items()
            if isinstance(values, dict) and isinstance(values.get("mean"), (int, float))
        )
        print_step(f"  {batch['retriever_name']} run{batch['run_idx']}: {line}")
    print_step(f"{conversation.sample_id}: QA complete in {summary['total_seconds'] / 60:.1f} min")
    return summary


def _parse_csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-index", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-json-path", type=Path, required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--question-types", type=_parse_csv, default=None)
    parser.add_argument("--retrievers", type=_parse_csv, default=None)
    parser.add_argument("--drop-adversarial", action="store_true")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-concurrent-questions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(answer_conversation(args))
    except Exception as error:
        write_json(
            str(
                args.run_dir
                / "qa"
                / f"{ARTIFACT_PREFIX}_qa_summary_conv{args.conversation_index}.json"
            ),
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        print_step(f"QA FAILED: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
