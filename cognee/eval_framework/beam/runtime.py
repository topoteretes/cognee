import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cognee.eval_framework.benchmark_adapters.beam_adapter import BEAMAdapter
from cognee.eval_framework.benchmark_adapters.beam_json_file_adapter import BEAMJsonFileAdapter
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.eval_config import EvalConfig
from cognee.modules.chunking.ConversationChunker import ConversationChunker
from cognee.shared.logging_utils import get_logger

logger = get_logger()

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "temp"


def make_timestamped_output_dir(prefix: str = "beam", root_dir: Path = TEMP_ROOT) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root_dir / f"{prefix}_{timestamp}"


def build_beam_eval_params(
    conversation_index: int,
    output_dir: Path,
    *,
    answering_questions: bool,
    qa_engine: str = "beam_router",
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
        evaluation_engine="DeepEval",
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


def build_beam_benchmark(
    *,
    split: str,
    conversation_index: int,
    max_batches: Optional[int] = None,
    conversation_json: Optional[str] = None,
) -> Any:
    if conversation_json:
        return BEAMJsonFileAdapter(conversation_json)

    return BEAMAdapter(
        split=split,
        max_batches=max_batches,
        conversation_index=conversation_index,
    )


async def build_beam_conversation_corpus(
    *,
    conversation_index: int,
    output_dir: Path,
    split: str,
    max_batches: Optional[int] = None,
    conversation_json: Optional[str] = None,
    answering_questions: bool,
    qa_engine: str = "beam_router",
) -> dict[str, Any]:
    params = build_beam_eval_params(
        conversation_index=conversation_index,
        output_dir=output_dir,
        answering_questions=answering_questions,
        qa_engine=qa_engine,
    )
    params["benchmark"] = build_beam_benchmark(
        split=split,
        conversation_index=conversation_index,
        max_batches=max_batches,
        conversation_json=conversation_json,
    )
    params["chunker"] = ConversationChunker

    if conversation_json:
        logger.info(
            "[conv %s] Building corpus from local JSON: %s",
            conversation_index,
            conversation_json,
        )
    else:
        logger.info("[conv %s] Building corpus...", conversation_index)

    await run_corpus_builder(params)
    return params


def load_and_annotate_questions(questions_path: str) -> list[dict[str, Any]]:
    with open(questions_path, "r", encoding="utf-8") as handle:
        raw_questions = json.load(handle)

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


def filter_questions_by_type(
    questions: list[dict[str, Any]], question_types: Optional[list[str]]
) -> list[dict[str, Any]]:
    if not question_types:
        return questions

    target_types = set(question_types)
    return [question for question in questions if question.get("question_type") in target_types]


async def prepare_beam_questions(
    *,
    conversation_index: int,
    output_dir: Path,
    split: str,
    max_batches: Optional[int] = None,
    question_types: Optional[list[str]] = None,
    conversation_json: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params = await build_beam_conversation_corpus(
        conversation_index=conversation_index,
        output_dir=output_dir,
        split=split,
        max_batches=max_batches,
        conversation_json=conversation_json,
        answering_questions=False,
    )
    questions = load_and_annotate_questions(params["questions_path"])
    questions = filter_questions_by_type(questions, question_types)
    logger.info("[conv %s] Loaded %s questions", conversation_index, len(questions))
    return params, questions
