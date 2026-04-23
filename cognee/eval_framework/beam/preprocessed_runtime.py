from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cognee

from cognee.eval_framework.beam.runtime import (
    build_beam_eval_params,
    filter_questions_by_type,
    load_and_annotate_questions,
)
from cognee.eval_framework.benchmark_adapters.beam_preprocessed_adapter import (
    BEAMPreprocessedAdapter,
)
from cognee.eval_framework.corpus_builder.run_corpus_builder import (
    create_and_insert_questions_table,
)
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.shared.logging_utils import get_logger

logger = get_logger()

DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH = 10
DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE = 800
DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE = 1200


def _chunk_documents(documents: list[str], docs_per_add_batch: int) -> list[list[str]]:
    if docs_per_add_batch < 1:
        raise ValueError("docs_per_add_batch must be at least 1")

    return [
        documents[index : index + docs_per_add_batch]
        for index in range(0, len(documents), docs_per_add_batch)
    ]


async def prune_preprocessed_ingestion_state() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def ingest_preprocessed_corpus(
    corpus_list: list[str],
    *,
    dataset_name: str,
    docs_per_add_batch: int = DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    chunk_size: int = DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    chunks_per_batch: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    skip_prune: bool = False,
    batch_label: str = "preprocessed",
) -> None:
    """Sequentially ingest preprocessed documents using add + cognify batches."""
    if not skip_prune:
        await prune_preprocessed_ingestion_state()

    batches = _chunk_documents(corpus_list, docs_per_add_batch)
    for batch_index, batch_docs in enumerate(batches, start=1):
        logger.info(
            "Ingesting %s batch %s/%s (%s docs) into dataset '%s'",
            batch_label,
            batch_index,
            len(batches),
            len(batch_docs),
            dataset_name,
        )
        await cognee.add(batch_docs, dataset_name=dataset_name)
        await cognee.cognify(
            datasets=[dataset_name],
            chunker=TextChunker,
            chunk_size=chunk_size,
            chunks_per_batch=chunks_per_batch,
            custom_prompt=custom_prompt,
        )
        logger.info(
            "Completed %s batch %s/%s for dataset '%s'",
            batch_label,
            batch_index,
            len(batches),
            dataset_name,
        )


async def build_beam_preprocessed_conversation_corpus(
    *,
    conversation_index: int,
    output_dir: Path,
    split: str,
    max_batches: Optional[int] = None,
    answering_questions: bool,
    qa_engine: str = "beam_router",
    docs_per_add_batch: int = DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    preprocessed_max_chunk_size: int = DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    cognify_chunk_size: int = DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    chunks_per_batch: Optional[int] = None,
    custom_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Build one BEAM conversation corpus through preprocessed add + cognify batches."""
    params = build_beam_eval_params(
        conversation_index=conversation_index,
        output_dir=output_dir,
        answering_questions=answering_questions,
        qa_engine=qa_engine,
    )
    dataset_name = f"beam_preprocessed_{output_dir.name}_conv{conversation_index}"
    params.update(
        {
            "benchmark": "BEAM",
            "dataset_name": dataset_name,
            "chunker": TextChunker,
            "chunk_size": cognify_chunk_size,
            "docs_per_add_batch": docs_per_add_batch,
            "preprocessed_max_chunk_size": preprocessed_max_chunk_size,
            "ingestion_mode": "batched_preprocessed",
        }
    )
    if chunks_per_batch is not None:
        params["chunks_per_batch"] = chunks_per_batch

    adapter = BEAMPreprocessedAdapter(
        split=split,
        conversation_index=conversation_index,
        max_batches=max_batches,
        preprocessed_max_chunk_size=preprocessed_max_chunk_size,
    )
    corpus_list, questions = adapter.load_corpus(
        limit=params.get("number_of_samples_in_corpus"),
        load_golden_context=params.get("evaluating_contexts", False),
    )

    logger.info(
        "[conv %s] Building preprocessed corpus with %s docs (batch size %s)",
        conversation_index,
        len(corpus_list),
        docs_per_add_batch,
    )
    await ingest_preprocessed_corpus(
        corpus_list,
        dataset_name=dataset_name,
        docs_per_add_batch=docs_per_add_batch,
        chunk_size=cognify_chunk_size,
        chunks_per_batch=chunks_per_batch,
        custom_prompt=custom_prompt,
    )

    with open(params["questions_path"], "w", encoding="utf-8") as handle:
        json.dump(questions, handle, ensure_ascii=False, indent=4)

    await create_and_insert_questions_table(questions_payload=questions)
    return params


async def prepare_beam_preprocessed_questions(
    *,
    conversation_index: int,
    output_dir: Path,
    split: str,
    max_batches: Optional[int] = None,
    question_types: Optional[list[str]] = None,
    docs_per_add_batch: int = DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    preprocessed_max_chunk_size: int = DEFAULT_PREPROCESSED_MAX_CHUNK_SIZE,
    cognify_chunk_size: int = DEFAULT_PREPROCESSED_COGNIFY_CHUNK_SIZE,
    chunks_per_batch: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params = await build_beam_preprocessed_conversation_corpus(
        conversation_index=conversation_index,
        output_dir=output_dir,
        split=split,
        max_batches=max_batches,
        answering_questions=False,
        docs_per_add_batch=docs_per_add_batch,
        preprocessed_max_chunk_size=preprocessed_max_chunk_size,
        cognify_chunk_size=cognify_chunk_size,
        chunks_per_batch=chunks_per_batch,
    )
    questions = load_and_annotate_questions(params["questions_path"])
    questions = filter_questions_by_type(questions, question_types)
    logger.info("[conv %s] Loaded %s questions", conversation_index, len(questions))
    return params, questions
