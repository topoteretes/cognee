from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cognee.eval_framework.beam.eval.registry import ANSWERING_STRATEGIES
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.sweeps.retriever_sweep_runner import validate_retriever_configs

REPO_ROOT = Path(__file__).resolve().parents[4]
TEMP_ROOT = REPO_ROOT / "temp"
ANSWER_FIELD_NAMES = ("answer", "ideal_response", "ideal_answer", "ideal_summary")


def make_timestamped_output_dir(prefix: str = "beam", root_dir: Path = TEMP_ROOT) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root_dir / f"{prefix}_{timestamp}"


def build_beam_eval_params(
    conversation_index: int,
    output_dir: Path,
    *,
    answering_questions: bool,
    qa_engine: str = "existing_ingestion_sweep",
    benchmark_name: str = "BEAM",
) -> dict[str, Any]:
    return EvalConfig(
        benchmark=benchmark_name,
        building_corpus_from_scratch=True,
        number_of_samples_in_corpus=20,
        qa_engine=qa_engine,
        answering_questions=answering_questions,
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="BeamEval",
        evaluation_metrics=["beam_rubric", "kendall_tau"],
        task_getter_type="Default",
        calculate_metrics=True,
        dashboard=False,
        questions_path=str(output_dir / f"beam_questions_conv{conversation_index}.json"),
        answers_path=str(output_dir / f"beam_answers_conv{conversation_index}.json"),
        metrics_path=str(output_dir / f"beam_metrics_conv{conversation_index}.json"),
        aggregate_metrics_path=str(
            output_dir / f"beam_aggregate_metrics_conv{conversation_index}.json"
        ),
        dashboard_path=str(output_dir / f"beam_dashboard_conv{conversation_index}.html"),
    ).to_dict()


def load_beam_sweep_payload_from_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("BEAM sweep config must be a JSON object")

    return payload


def build_registry_base_configs() -> list[dict[str, Any]]:
    configs = []
    for name, spec in sorted(ANSWERING_STRATEGIES.items()):
        if spec.mode != "fixed_retriever":
            continue
        configs.append(
            {
                "name": name,
                "mode": spec.mode,
                "retriever_cls": spec.cls,
                "retriever_kwargs": dict(spec.default_kwargs),
            }
        )
    return configs


def resolve_beam_sweep_config(
    payload: dict[str, Any],
    base_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("BEAM sweep config payload must be a dictionary")

    variants = payload.get("retrievers")
    if not isinstance(variants, list) or not variants:
        raise ValueError("BEAM sweep config requires a non-empty 'retrievers' list")

    lookup = {config["name"]: config for config in base_configs}
    configs = []
    names = set()

    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"Retriever variant at index {index} must be a JSON object")

        config = _build_variant_config(variant, lookup, names, index)
        configs.append(config)

    validate_retriever_configs(configs)
    return configs


def _build_variant_config(
    variant: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    names: set[str],
    index: int,
) -> dict[str, Any]:
    name = _get_required_string(variant, "name", f"retriever variant at index {index}")
    if name in names:
        raise ValueError(f"Duplicate retriever variant name: {name}")
    names.add(name)

    base_name = _get_required_string(variant, "base", f"retriever variant '{name}'")
    try:
        config = deepcopy(lookup[base_name])
    except KeyError as exc:
        available = ", ".join(sorted(lookup))
        raise ValueError(
            f"Unsupported base retriever '{base_name}' for variant '{name}'. Available: {available}"
        ) from exc

    config["name"] = name
    config["retriever_kwargs"] = {
        **dict(config.get("retriever_kwargs", {})),
        **_optional_dict(variant, "kwargs", name),
    }

    if "qa_prompt_paths" in variant:
        config["qa_prompt_paths"] = _optional_dict(variant, "qa_prompt_paths", name)
    if "agentic_prompt_paths" in variant:
        config["agentic_prompt_paths"] = _optional_dict(variant, "agentic_prompt_paths", name)
    if "system_prompt" in variant:
        config["system_prompt"] = _get_required_string(variant, "system_prompt", name)

    return config


def _optional_dict(payload: dict[str, Any], key: str, variant_name: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Retriever variant '{variant_name}' has non-object '{key}'")
    return dict(value)


def _get_required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing non-empty '{key}'")
    return value.strip()


def load_and_annotate_questions(questions_path: str) -> list[dict[str, Any]]:
    with open(questions_path, "r", encoding="utf-8") as handle:
        raw_questions = json.load(handle)

    raw_questions = _normalize_questions_payload(raw_questions)

    annotated_questions: list[dict[str, Any]] = []
    for index, question in enumerate(raw_questions):
        for required_field in ("question", "answer", "question_type"):
            if required_field not in question:
                raise ValueError(
                    f"Question at index {index} is missing required field '{required_field}'"
                )

        annotated = dict(question)
        annotated["question_idx"] = index
        annotated.setdefault("conversation_id", "unknown")
        annotated_questions.append(annotated)

    return annotated_questions


def _normalize_questions_payload(raw_questions: Any) -> list[dict[str, Any]]:
    if isinstance(raw_questions, list):
        return raw_questions
    if isinstance(raw_questions, dict):
        return _flatten_grouped_beam_questions(raw_questions)
    raise ValueError("Questions file must contain a list or grouped BEAM questions object")


def _flatten_grouped_beam_questions(grouped_questions: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for question_type, entries in grouped_questions.items():
        if not isinstance(entries, list):
            continue
        for raw_question in entries:
            if not isinstance(raw_question, dict) or "question" not in raw_question:
                continue
            questions.append(
                {
                    "question": raw_question["question"],
                    "answer": _extract_answer(raw_question),
                    "question_type": question_type,
                    "rubric": _normalize_rubric(raw_question.get("rubric", [])),
                    "difficulty": raw_question.get("difficulty", "unknown"),
                }
            )
    return questions


def _extract_answer(raw_question: dict[str, Any]) -> str:
    for field_name in ANSWER_FIELD_NAMES:
        value = raw_question.get(field_name)
        if value:
            return value if isinstance(value, str) else str(value)
    return ""


def _normalize_rubric(rubric: Any) -> list[str]:
    if isinstance(rubric, str):
        return [rubric]
    if isinstance(rubric, list):
        return [item if isinstance(item, str) else str(item) for item in rubric]
    return []


def filter_questions_by_type(
    questions: list[dict[str, Any]], question_types: Optional[list[str]]
) -> list[dict[str, Any]]:
    if not question_types:
        return questions

    target_types = set(question_types)
    return [question for question in questions if question.get("question_type") in target_types]
