"""Generic multi-strategy answer + DeepEval scoring sweep."""

import asyncio
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from cognee.eval_framework.answer_generation.beam_router import get_beam_question_type_prompt
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.reporting.aggregations import (
    build_combined_summary,
    build_conversation_summary,
)
from cognee.eval_framework.reporting.io import read_json, write_json
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@dataclass(frozen=True)
class RetrieverSweepSettings:
    """Shared knobs for artifact paths, concurrency, and summary shaping."""

    output_dir: Path
    num_runs: int = 1
    max_concurrent_questions: int = 10
    question_types: Optional[list[str]] = None
    primary_metric_name: str = "score"
    artifact_prefix: str = "sweep"
    combined_summary_filename: str = "sweep_combined.json"
    summary_tags: dict[str, Any] = field(default_factory=dict)


def validate_retriever_configs(configs: list[dict[str, Any]]) -> None:
    names = set()
    for config in configs:
        name = config.get("name")
        mode = config.get("mode")

        if not name:
            raise ValueError("Retriever config is missing 'name'")
        if name in names:
            raise ValueError(f"Duplicate retriever config name: {name}")
        names.add(name)

        if mode not in {"router", "fixed_retriever"}:
            raise ValueError(f"Unsupported retriever config mode for '{name}': {mode}")

        if mode == "router":
            if "router_kwargs" not in config:
                raise ValueError(f"Router config '{name}' is missing 'router_kwargs'")
            if "router_cls" not in config:
                raise ValueError(f"Router config '{name}' is missing 'router_cls'")

        if mode == "fixed_retriever":
            if "retriever_cls" not in config:
                raise ValueError(f"Fixed retriever config '{name}' is missing 'retriever_cls'")
            if "retriever_kwargs" not in config:
                raise ValueError(f"Fixed retriever config '{name}' is missing 'retriever_kwargs'")


def slugify_config_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "retriever"


def get_batch_paths(
    output_dir: Path,
    artifact_prefix: str,
    conversation_index: int,
    retriever_name: str,
    run_idx: int,
) -> dict[str, str]:
    suffix = f"conv{conversation_index}_{slugify_config_name(retriever_name)}_run{run_idx}"
    return {
        "answers_path": str(output_dir / f"{artifact_prefix}_answers_{suffix}.json"),
        "metrics_path": str(output_dir / f"{artifact_prefix}_metrics_{suffix}.json"),
        "aggregate_metrics_path": str(
            output_dir / f"{artifact_prefix}_aggregate_metrics_{suffix}.json"
        ),
    }


def normalize_answer_text(search_results: Any) -> str:
    if isinstance(search_results, str):
        return search_results

    if isinstance(search_results, list):
        if not search_results:
            return ""
        first = search_results[0]
        return first if isinstance(first, str) else str(first)

    if search_results is None:
        return ""

    return str(search_results)


def build_answer_record(
    question: dict[str, Any],
    answer_text: str,
    retrieval_context: str,
    retriever_name: str,
    run_idx: int,
) -> dict[str, Any]:
    answer = {
        "conversation_id": question.get("conversation_id", "unknown"),
        "question_idx": question["question_idx"],
        "question": question["question"],
        "answer": answer_text,
        "golden_answer": question["answer"],
        "retrieval_context": retrieval_context,
        "question_type": question.get("question_type", "unknown"),
        "retriever_name": retriever_name,
        "run_idx": run_idx,
    }

    for optional_key in ("rubric", "difficulty", "golden_context"):
        if optional_key in question:
            answer[optional_key] = question[optional_key]

    return answer


async def _answer_single_fixed_retriever(
    question: dict[str, Any],
    config: dict[str, Any],
    run_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        retriever_name = config["name"]
        retriever_kwargs = dict(config.get("retriever_kwargs", {}))
        if config.get("use_beam_question_type_prompts"):
            retriever_kwargs["system_prompt"] = get_beam_question_type_prompt(
                question.get("question_type", "unknown")
            )
        else:
            system_prompt = config.get("system_prompt")
            if system_prompt is not None:
                retriever_kwargs["system_prompt"] = system_prompt

        retriever = config["retriever_cls"](**retriever_kwargs)

        try:
            retrieved_objects = await retriever.get_retrieved_objects(query=question["question"])
            retrieval_context = await retriever.get_context_from_objects(
                query=question["question"],
                retrieved_objects=retrieved_objects,
            )
            search_results = await retriever.get_completion_from_context(
                query=question["question"],
                retrieved_objects=retrieved_objects,
                context=retrieval_context,
            )
            answer_text = normalize_answer_text(search_results)
        except Exception as exc:
            logger.error(
                "[%s][run %s] Failed to answer question_idx=%s: %s",
                retriever_name,
                run_idx,
                question["question_idx"],
                exc,
            )
            answer_text = f"ERROR: {exc}"
            retrieval_context = ""

        return build_answer_record(
            question=question,
            answer_text=answer_text,
            retrieval_context=retrieval_context,
            retriever_name=retriever_name,
            run_idx=run_idx,
        )


async def answer_with_config(
    questions: list[dict[str, Any]],
    config: dict[str, Any],
    run_idx: int,
    max_concurrent: int,
) -> list[dict[str, Any]]:
    retriever_name = config["name"]

    if config["mode"] == "router":
        router = config["router_cls"](**config.get("router_kwargs", {}))
        router_answers = await router.answer_questions(questions, max_concurrent=max_concurrent)
        if len(router_answers) != len(questions):
            raise ValueError(
                f"Router '{retriever_name}' returned {len(router_answers)} answers for "
                f"{len(questions)} questions"
            )

        enriched_answers = []
        for question, answer in zip(questions, router_answers):
            enriched_answer = dict(answer)
            enriched_answer["conversation_id"] = question.get("conversation_id", "unknown")
            enriched_answer["question_idx"] = question["question_idx"]
            enriched_answer["retriever_name"] = retriever_name
            enriched_answer["run_idx"] = run_idx
            enriched_answers.append(enriched_answer)

        return enriched_answers

    if config["mode"] != "fixed_retriever":
        raise ValueError(f"Unsupported config mode: {config['mode']}")

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _answer_single_fixed_retriever(question, config, run_idx, semaphore)
        for question in questions
    ]
    return await asyncio.gather(*tasks)


async def evaluate_batch(
    base_params: dict[str, Any],
    output_dir: Path,
    artifact_prefix: str,
    conversation_index: int,
    retriever_name: str,
    run_idx: int,
    answers: list[dict[str, Any]],
    run_evaluation_fn: Callable[[dict[str, Any]], Awaitable[None]] = run_evaluation,
) -> dict[str, Any]:
    batch_paths = get_batch_paths(
        output_dir, artifact_prefix, conversation_index, retriever_name, run_idx
    )
    write_json(batch_paths["answers_path"], answers)

    batch_params = deepcopy(base_params)
    batch_params.update(batch_paths)
    await run_evaluation_fn(batch_params)

    metrics = read_json(batch_paths["metrics_path"])
    aggregate_metrics = read_json(batch_paths["aggregate_metrics_path"])

    return {
        "retriever_name": retriever_name,
        "run_idx": run_idx,
        **batch_paths,
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
    }


async def run_retriever_sweep_for_questions(
    conversation_index: int,
    settings: RetrieverSweepSettings,
    retriever_configs: list[dict[str, Any]],
    base_eval_params: dict[str, Any],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_results = []
    prefix = settings.artifact_prefix
    for config in retriever_configs:
        for run_idx in range(settings.num_runs):
            logger.info(
                "[conv %s] Answering with %s (run %s/%s)...",
                conversation_index,
                config["name"],
                run_idx + 1,
                settings.num_runs,
            )
            answers = await answer_with_config(
                questions=questions,
                config=config,
                run_idx=run_idx,
                max_concurrent=settings.max_concurrent_questions,
            )

            logger.info(
                "[conv %s] Evaluating %s (run %s/%s)...",
                conversation_index,
                config["name"],
                run_idx + 1,
                settings.num_runs,
            )
            batch_result = await evaluate_batch(
                base_params=base_eval_params,
                output_dir=settings.output_dir,
                artifact_prefix=prefix,
                conversation_index=conversation_index,
                retriever_name=config["name"],
                run_idx=run_idx,
                answers=answers,
            )
            batch_results.append(batch_result)

    summary = build_conversation_summary(
        conversation_index=conversation_index,
        settings=settings,
        batch_results=batch_results,
        retriever_configs=retriever_configs,
    )
    summary_path = str(settings.output_dir / f"{prefix}_sweep_conv{conversation_index}.json")
    write_json(summary_path, summary)
    logger.info("[conv %s] Wrote conversation summary to %s", conversation_index, summary_path)
    return summary


__all__ = [
    "RetrieverSweepSettings",
    "build_combined_summary",
    "build_conversation_summary",
    "get_batch_paths",
    "run_retriever_sweep_for_questions",
    "validate_retriever_configs",
]
