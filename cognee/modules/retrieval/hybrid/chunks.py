import asyncio
from typing import Any, Optional

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.hybrid.pairs import (
    attach_source_chunks,
    chunk_summary_pairs,
    source_chunk_ids_to_load,
    summary_id_for_chunk,
    summary_text_by_chunk_id,
)
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs
from cognee.modules.retrieval.hybrid.results import (
    display_value,
    payload,
    payload_matches_node_filter,
    result_id,
    scored_payload,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


async def retrieve_hybrid_chunks(
    vector_engine: Any,
    query: str,
    chunks_top_k: int,
    text_summaries_top_k: Optional[int],
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
    use_importance_weight: bool,
    query_vector: Optional[list[float]] = None,
) -> dict[str, Any]:
    candidate_limit = max(0, chunks_top_k * 2)
    summary_limit = summary_candidate_limit(chunks_top_k, text_summaries_top_k)
    bm25_chunks, vector_chunks, summary_hits = await asyncio.gather(
        search_bm25_chunks(query, candidate_limit, node_name, node_name_filter_operator),
        search_collection(
            vector_engine,
            "DocumentChunk_text",
            query,
            candidate_limit,
            node_name,
            node_name_filter_operator,
            required=True,
            query_vector=query_vector,
        ),
        search_collection(
            vector_engine,
            "TextSummary_text",
            query,
            summary_limit,
            node_name,
            node_name_filter_operator,
            query_vector=query_vector,
        ),
    )

    pairs = chunk_summary_pairs(
        bm25_chunks,
        vector_chunks,
        summary_hits,
        node_name,
        node_name_filter_operator,
    )
    missing_source_chunk_ids = source_chunk_ids_to_load(pairs)
    if missing_source_chunk_ids:
        source_chunks = await load_source_chunks_for_summaries(
            vector_engine,
            missing_source_chunk_ids,
            node_name,
            node_name_filter_operator,
        )
        attach_source_chunks(pairs, source_chunks)

    ranked_pairs = rank_chunk_summary_pairs(pairs, chunks_top_k, use_importance_weight)
    if summary_limit > 0:
        await load_summary_text_for_ranked_pairs(
            vector_engine,
            ranked_pairs,
            node_name,
            node_name_filter_operator,
        )

    return {
        "chunks": [pair["chunk"] for pair in ranked_pairs if pair["chunk"] is not None],
        "chunk_summaries": summary_text_by_chunk_id(ranked_pairs),
    }


def summary_candidate_limit(chunks_top_k: int, text_summaries_top_k: Optional[int]) -> int:
    if text_summaries_top_k is None:
        return max(0, chunks_top_k)
    return text_summaries_top_k


async def search_bm25_chunks(
    query: str,
    limit: int,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> list[dict]:
    if limit <= 0:
        return []

    try:
        retriever = BM25ChunksRetriever(top_k=limit, with_scores=True)
        scored_chunks = await retriever.get_retrieved_objects(query)
    except NoDataError:
        return []
    except Exception as error:
        logger.warning("BM25 chunk retrieval failed; using vector chunks only: %s", error)
        return []

    chunks = []
    for item in scored_chunks:
        chunk, score = scored_payload(item)
        if score <= 0:
            continue
        if not isinstance(chunk, dict):
            continue
        if not payload_matches_node_filter(chunk, node_name, node_name_filter_operator):
            continue
        chunks.append(chunk)
    return chunks


async def search_collection(
    vector_engine: Any,
    collection_name: str,
    query: str,
    limit: int,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
    *,
    required: bool = False,
    apply_node_filter: bool = True,
    query_vector: Optional[list[float]] = None,
) -> list[Any]:
    if limit <= 0:
        return []

    search_node_name = node_name if apply_node_filter else None
    search_operator = node_name_filter_operator if apply_node_filter else "OR"
    try:
        return await vector_engine.search(
            collection_name,
            None if query_vector is not None else query,
            query_vector=query_vector,
            limit=limit,
            include_payload=True,
            node_name=search_node_name,
            node_name_filter_operator=search_operator,
        )
    except CollectionNotFoundError as error:
        if required:
            logger.error("%s collection not found", collection_name)
            raise NoDataError("No data found in the system, please add data first.") from error
        logger.debug("%s collection not found; using empty channel", collection_name)
        return []


async def load_source_chunks_for_summaries(
    vector_engine: Any,
    chunk_ids: list[str],
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> list[Any]:
    chunks = await vector_engine.retrieve("DocumentChunk_text", chunk_ids)
    found_ids = {result_id(chunk) for chunk in chunks}
    missing_ids = sorted(set(chunk_ids) - {chunk_id for chunk_id in found_ids if chunk_id})
    if missing_ids:
        logger.warning(
            "TextSummary_text hit referenced missing DocumentChunk_text row(s): %s",
            missing_ids,
        )

    source_chunks = []
    filtered_ids = []
    for chunk in chunks:
        if payload_matches_node_filter(payload(chunk), node_name, node_name_filter_operator):
            source_chunks.append(chunk)
            continue

        chunk_id = result_id(chunk)
        if chunk_id:
            filtered_ids.append(chunk_id)

    if filtered_ids:
        logger.warning(
            "TextSummary_text source chunk failed node filter: %s",
            sorted(filtered_ids),
        )
    return source_chunks


async def load_summary_text_for_ranked_pairs(
    vector_engine: Any,
    ranked_pairs: list[dict],
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> None:
    summary_ids_by_chunk_id = {}
    for pair in ranked_pairs:
        if pair["summary_text"]:
            continue

        chunk_id = pair["chunk_id"]
        if not chunk_id:
            continue

        summary_id = pair["summary_id"] or summary_id_for_chunk(chunk_id)
        if summary_id is None:
            logger.debug("Cannot fetch paired TextSummary for non-UUID chunk id %s", chunk_id)
            continue

        pair["summary_id"] = summary_id
        summary_ids_by_chunk_id[chunk_id] = summary_id

    if not summary_ids_by_chunk_id:
        return

    try:
        summaries = await vector_engine.retrieve(
            "TextSummary_text",
            list(summary_ids_by_chunk_id.values()),
        )
    except CollectionNotFoundError:
        logger.warning("TextSummary_text collection missing while loading chunk summaries")
        return

    summaries_by_id = {result_id(summary): summary for summary in summaries}
    for pair in ranked_pairs:
        chunk_id = pair["chunk_id"]
        summary_id = pair["summary_id"]
        if not chunk_id or not summary_id:
            continue

        summary = summaries_by_id.get(summary_id)
        if summary is None:
            logger.warning(
                "DocumentChunk_text row has no paired TextSummary_text row: chunk_id=%s",
                chunk_id,
            )
            continue

        summary_payload = payload(summary)
        summary_text = display_value(summary_payload.get("text"))
        if not summary_text:
            logger.warning(
                "Paired TextSummary_text row has no text: chunk_id=%s summary_id=%s",
                chunk_id,
                summary_id,
            )
            continue

        if not payload_matches_node_filter(summary_payload, node_name, node_name_filter_operator):
            logger.warning(
                "Paired TextSummary_text row failed node filter: chunk_id=%s summary_id=%s",
                chunk_id,
                summary_id,
            )
            continue

        pair["summary_text"] = summary_text
