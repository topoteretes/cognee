from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    create_and_insert_answers_table,
)
from cognee.eval_framework.beam.preprocessed_runtime import (
    DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    ingest_preprocessed_corpus,
    prune_preprocessed_ingestion_state,
)
from cognee.eval_framework.beam.runtime import (
    filter_questions_by_type,
    load_and_annotate_questions,
)
from cognee.eval_framework.benchmark_adapters.beam_10m_preprocessed_adapter import (
    BEAM10MPreprocessedAdapter,
)
from cognee.eval_framework.corpus_builder.run_corpus_builder import (
    create_and_insert_questions_table,
)
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_10M_TOP_K = {
    "summarization": 150,
    "DEFAULT": 50,
}
_10M_CONTEXT_EXTENSION_ROUNDS = 8
_10M_WIDE_SEARCH_TOP_K = 300
_10M_TRIPLET_DISTANCE_PENALTY = 4.0
DEFAULT_PREPROCESSED_10M_CHUNKS_PER_BATCH = 40


def build_beam_10m_eval_params(
    *,
    conversation_index: int,
    output_dir: Path,
    answering_questions: bool,
) -> dict[str, Any]:
    return EvalConfig(
        benchmark="BEAM-10M",
        building_corpus_from_scratch=True,
        number_of_samples_in_corpus=20,
        qa_engine="beam_router",
        answering_questions=answering_questions,
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="BeamEval",
        evaluation_metrics=["beam_rubric", "kendall_tau"],
        task_getter_type="Default",
        calculate_metrics=True,
        dashboard=False,
        questions_path=str(output_dir / f"beam10m_questions_conv{conversation_index}.json"),
        answers_path=str(output_dir / f"beam10m_answers_conv{conversation_index}.json"),
        metrics_path=str(output_dir / f"beam10m_metrics_conv{conversation_index}.json"),
        aggregate_metrics_path=str(
            output_dir / f"beam10m_aggregate_metrics_conv{conversation_index}.json"
        ),
        dashboard_path=str(output_dir / f"beam10m_dashboard_conv{conversation_index}.html"),
    ).to_dict()


async def build_beam_10m_preprocessed_conversation_corpus(
    *,
    conversation_index: int,
    output_dir: Path,
    plans: Optional[list[str]] = None,
    max_batches_per_plan: Optional[int] = None,
    docs_per_add_batch: int = DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    preprocessed_max_chunk_size: int = DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    cognify_chunk_size: int = DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    chunks_per_batch: int = DEFAULT_PREPROCESSED_10M_CHUNKS_PER_BATCH,
    custom_prompt: Optional[str] = None,
) -> dict[str, Any]:
    params = build_beam_10m_eval_params(
        conversation_index=conversation_index,
        output_dir=output_dir,
        answering_questions=False,
    )
    dataset_name = f"beam10m_preprocessed_{output_dir.name}_conv{conversation_index}"
    params.update(
        {
            "dataset_name": dataset_name,
            "chunker": TextChunker,
            "chunk_size": cognify_chunk_size,
            "docs_per_add_batch": docs_per_add_batch,
            "preprocessed_max_chunk_size": preprocessed_max_chunk_size,
            "ingestion_mode": "batched_preprocessed_10m",
            "chunks_per_batch": chunks_per_batch,
        }
    )

    adapter = BEAM10MPreprocessedAdapter(
        conversation_index=conversation_index,
        plans=plans,
        max_batches_per_plan=max_batches_per_plan,
        preprocessed_max_chunk_size=preprocessed_max_chunk_size,
    )
    plan_documents, questions = adapter.load_plan_corpus(
        limit=params.get("number_of_samples_in_corpus"),
        load_golden_context=params.get("evaluating_contexts", False),
    )

    logger.info(
        "[conv %s] Building BEAM-10M preprocessed corpus with %s plans",
        conversation_index,
        len(plan_documents),
    )
    await prune_preprocessed_ingestion_state()

    total_plans = len(plan_documents)
    for plan_index, (plan_name, documents) in enumerate(plan_documents, start=1):
        logger.info(
            "[conv %s] Ingesting %s (%s/%s) with %s preprocessed docs",
            conversation_index,
            plan_name,
            plan_index,
            total_plans,
            len(documents),
        )
        await ingest_preprocessed_corpus(
            documents,
            dataset_name=dataset_name,
            docs_per_add_batch=docs_per_add_batch,
            chunk_size=cognify_chunk_size,
            chunks_per_batch=chunks_per_batch,
            custom_prompt=custom_prompt,
            skip_prune=True,
            batch_label=f"{plan_name} preprocessed",
        )

    with open(params["questions_path"], "w", encoding="utf-8") as handle:
        json.dump(questions, handle, ensure_ascii=False, indent=4)

    await create_and_insert_questions_table(questions_payload=questions)
    return params


async def answer_beam_10m_questions(
    params: dict[str, Any],
    *,
    question_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    questions = load_and_annotate_questions(params["questions_path"])
    questions = filter_questions_by_type(questions, question_types)
    logger.info("Loaded %s BEAM-10M questions for answering", len(questions))

    if not questions:
        with open(params["answers_path"], "w", encoding="utf-8") as handle:
            json.dump([], handle, ensure_ascii=False, indent=4)
        await create_and_insert_answers_table([])
        return []

    router = BEAMRouter(
        top_k_overrides=_10M_TOP_K,
        context_extension_rounds=_10M_CONTEXT_EXTENSION_ROUNDS,
        wide_search_top_k=_10M_WIDE_SEARCH_TOP_K,
        triplet_distance_penalty=_10M_TRIPLET_DISTANCE_PENALTY,
    )
    answers = await router.answer_questions(questions)

    with open(params["answers_path"], "w", encoding="utf-8") as handle:
        json.dump(answers, handle, ensure_ascii=False, indent=4)

    await create_and_insert_answers_table(answers)
    return answers


async def run_beam_10m_preprocessed_conversation(
    *,
    conversation_index: int,
    output_dir: Path,
    plans: Optional[list[str]] = None,
    max_batches_per_plan: Optional[int] = None,
    question_types: Optional[list[str]] = None,
    docs_per_add_batch: int = DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    preprocessed_max_chunk_size: int = DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    cognify_chunk_size: int = DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    chunks_per_batch: int = DEFAULT_PREPROCESSED_10M_CHUNKS_PER_BATCH,
) -> dict[str, Any]:
    params = await build_beam_10m_preprocessed_conversation_corpus(
        conversation_index=conversation_index,
        output_dir=output_dir,
        plans=plans,
        max_batches_per_plan=max_batches_per_plan,
        docs_per_add_batch=docs_per_add_batch,
        preprocessed_max_chunk_size=preprocessed_max_chunk_size,
        cognify_chunk_size=cognify_chunk_size,
        chunks_per_batch=chunks_per_batch,
    )

    answers = await answer_beam_10m_questions(params, question_types=question_types)
    if not answers:
        logger.warning(
            "[conv %s] No answers generated after question filtering", conversation_index
        )
        return {}

    await run_evaluation(params)

    with open(params["aggregate_metrics_path"], "r", encoding="utf-8") as handle:
        return json.load(handle)
