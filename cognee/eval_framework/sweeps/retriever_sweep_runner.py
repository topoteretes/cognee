"""Generic multi-strategy answer + scoring sweep."""

from __future__ import annotations

import asyncio
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from cognee.eval_framework.answer_generation.question_type_prompts import (
    get_question_type_prompt,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.reporting.io import read_json, write_json
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@dataclass(frozen=True)
class RetrieverSweepSettings:
    """Shared knobs for artifact paths and concurrency."""

    output_dir: Path
    num_runs: int = 1
    parallel_runs: bool = False
    max_concurrent_questions: int = 10
    question_types: Optional[list[str]] = None
    primary_metric_name: str = "score"
    artifact_prefix: str = "sweep"
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

        if mode != "fixed_retriever":
            raise ValueError(f"Unsupported retriever config mode for '{name}': {mode}")
        if "retriever_cls" not in config:
            raise ValueError(f"Fixed retriever config '{name}' is missing 'retriever_cls'")
        if "retriever_kwargs" not in config:
            raise ValueError(f"Fixed retriever config '{name}' is missing 'retriever_kwargs'")

        qa_prompt_paths = config.get("qa_prompt_paths")
        if qa_prompt_paths is not None and not isinstance(qa_prompt_paths, dict):
            raise ValueError(f"Retriever config '{name}' has non-dict 'qa_prompt_paths'")
        agentic_prompt_paths = config.get("agentic_prompt_paths")
        if agentic_prompt_paths is not None and not isinstance(agentic_prompt_paths, dict):
            raise ValueError(f"Retriever config '{name}' has non-dict 'agentic_prompt_paths'")


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


def _question_answer_cache_path(answer_cache_dir: Path, question: dict[str, Any]) -> Path:
    question_idx = question["question_idx"]
    return answer_cache_dir / f"question_{question_idx}.json"


def _load_cached_answer(
    answer_cache_dir: Optional[Path], question: dict[str, Any]
) -> Optional[dict]:
    if answer_cache_dir is None:
        return None

    answer_path = _question_answer_cache_path(answer_cache_dir, question)
    if not answer_path.exists():
        return None

    cached_answer = read_json(str(answer_path))
    if isinstance(cached_answer, dict):
        return cached_answer

    return None


def _write_cached_answer(answer_cache_dir: Optional[Path], answer: dict[str, Any]) -> None:
    if answer_cache_dir is None:
        return

    answer_text = answer.get("answer")
    if isinstance(answer_text, str) and answer_text.startswith("ERROR:"):
        return

    answer_cache_dir.mkdir(parents=True, exist_ok=True)
    answer_path = _question_answer_cache_path(answer_cache_dir, answer)
    write_json(str(answer_path), answer)


async def _answer_single_fixed_retriever(
    question: dict[str, Any],
    config: dict[str, Any],
    run_idx: int,
    semaphore: asyncio.Semaphore,
    answer_cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    cached_answer = _load_cached_answer(answer_cache_dir, question)
    if cached_answer is not None:
        logger.info(
            "[%s][run %s] Skipping cached question_idx=%s",
            config["name"],
            run_idx,
            question["question_idx"],
        )
        return cached_answer

    async with semaphore:
        retriever_name = config["name"]
        retriever_kwargs = dict(config.get("retriever_kwargs", {}))
        qa_prompt_paths = config.get("qa_prompt_paths")
        prompt = None
        if qa_prompt_paths is not None:
            prompt = get_question_type_prompt(
                qa_prompt_paths, question.get("question_type", "unknown")
            )

        if prompt is not None:
            retriever_kwargs["system_prompt"] = prompt
        else:
            system_prompt = config.get("system_prompt")
            if system_prompt is not None:
                retriever_kwargs["system_prompt"] = system_prompt

        agentic_prompt_paths = config.get("agentic_prompt_paths")
        if agentic_prompt_paths is not None:
            agentic_prompt = get_question_type_prompt(
                agentic_prompt_paths, question.get("question_type", "unknown")
            )
            if agentic_prompt is not None:
                retriever_kwargs["agentic_system_prompt"] = agentic_prompt

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

        answer = build_answer_record(
            question=question,
            answer_text=answer_text,
            retrieval_context=retrieval_context,
            retriever_name=retriever_name,
            run_idx=run_idx,
        )
        _write_cached_answer(answer_cache_dir, answer)
        return answer


async def answer_with_config(
    questions: list[dict[str, Any]],
    config: dict[str, Any],
    run_idx: int,
    max_concurrent: int,
    answer_cache_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    if config["mode"] != "fixed_retriever":
        raise ValueError(f"Unsupported config mode: {config['mode']}")

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _answer_single_fixed_retriever(question, config, run_idx, semaphore, answer_cache_dir)
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


async def _run_retriever_run(
    *,
    conversation_index: int,
    settings: RetrieverSweepSettings,
    config: dict[str, Any],
    base_eval_params: dict[str, Any],
    questions: list[dict[str, Any]],
    run_idx: int,
    run_evaluation_fn: Callable[[dict[str, Any]], Awaitable[None]] = run_evaluation,
) -> dict[str, Any]:
    answers = await answer_with_config(
        questions=questions,
        config=config,
        run_idx=run_idx,
        max_concurrent=settings.max_concurrent_questions,
    )
    return await evaluate_batch(
        base_params=base_eval_params,
        output_dir=settings.output_dir,
        artifact_prefix=settings.artifact_prefix,
        conversation_index=conversation_index,
        retriever_name=config["name"],
        run_idx=run_idx,
        answers=answers,
        run_evaluation_fn=run_evaluation_fn,
    )


async def _run_retriever_all_runs(
    *,
    conversation_index: int,
    settings: RetrieverSweepSettings,
    config: dict[str, Any],
    base_eval_params: dict[str, Any],
    questions: list[dict[str, Any]],
    run_evaluation_fn: Callable[[dict[str, Any]], Awaitable[None]] = run_evaluation,
) -> list[dict[str, Any]]:
    run_jobs = [
        _run_retriever_run(
            conversation_index=conversation_index,
            settings=settings,
            config=config,
            base_eval_params=base_eval_params,
            questions=questions,
            run_idx=run_idx,
            run_evaluation_fn=run_evaluation_fn,
        )
        for run_idx in range(settings.num_runs)
    ]
    if settings.parallel_runs and settings.num_runs > 1:
        return list(await asyncio.gather(*run_jobs))

    batch_results = []
    for run_job in run_jobs:
        batch_results.append(await run_job)
    return batch_results


async def run_retriever_sweep_for_questions(
    conversation_index: int,
    settings: RetrieverSweepSettings,
    retriever_configs: list[dict[str, Any]],
    base_eval_params: dict[str, Any],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    batch_results = []
    for config in retriever_configs:
        logger.info(
            "[conv %s] Answering and evaluating %s (%s run(s))...",
            conversation_index,
            config["name"],
            settings.num_runs,
        )
        batch_results.extend(
            await _run_retriever_all_runs(
                conversation_index=conversation_index,
                settings=settings,
                config=config,
                base_eval_params=base_eval_params,
                questions=questions,
            )
        )
    return batch_results


__all__ = [
    "RetrieverSweepSettings",
    "answer_with_config",
    "evaluate_batch",
    "get_batch_paths",
    "run_retriever_sweep_for_questions",
    "validate_retriever_configs",
]
