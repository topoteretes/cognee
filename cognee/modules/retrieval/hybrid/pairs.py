import math
from numbers import Real
from typing import Any, Optional
from uuid import UUID, uuid5

from cognee.modules.retrieval.hybrid.results import (
    display_value,
    payload,
    payload_matches_node_filter,
    result_id,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


def chunk_summary_pairs(
    bm25_chunks: list[Any],
    vector_chunks: list[Any],
    summary_hits: list[Any],
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
) -> list[dict]:
    pairs = []

    for rank_field, chunks in (("bm25_rank", bm25_chunks), ("vector_rank", vector_chunks)):
        score_field = rank_field.replace("_rank", "_score")
        for rank, scored_chunk in enumerate(chunks or []):
            chunk, native_score = _result_and_native_score(scored_chunk)
            chunk_id = result_id(chunk)
            chunk_text = display_value(payload(chunk).get("text"))
            if not chunk_id and not chunk_text:
                continue

            pair = _find_chunk_summary_pair(pairs, chunk_id, chunk_text)
            if pair is None:
                pair = _new_chunk_summary_pair(chunk_id=chunk_id, chunk_text=chunk_text)
                pairs.append(pair)
            if pair["chunk"] is None:
                set_pair_chunk(pair, chunk)
            elif pair["chunk_id"] is None and chunk_id:
                # Text-merged onto an idless chunk (e.g. BM25 payload without id):
                # adopt the id so summary hits can pair by source_chunk_id.
                pair["chunk_id"] = chunk_id
            if pair[rank_field] is None:
                pair[rank_field] = rank
                pair[score_field] = native_score

    for rank, scored_summary in enumerate(summary_hits or []):
        summary, native_score = _result_and_native_score(scored_summary)
        summary_payload = payload(summary)
        if not payload_matches_node_filter(summary_payload, node_name, node_name_filter_operator):
            continue

        chunk_id = display_value(summary_payload.get("source_chunk_id"))
        if not chunk_id:
            logger.warning(
                "TextSummary_text hit has no source_chunk_id: summary_id=%s", result_id(summary)
            )
            continue

        pair = _find_chunk_summary_pair(pairs, chunk_id, None)
        if pair is None:
            pair = _new_chunk_summary_pair(chunk_id=chunk_id)
            pairs.append(pair)
        if pair["summary_rank"] is None:
            pair["summary_rank"] = rank
            pair["summary_id"] = result_id(summary)
            pair["summary_text"] = display_value(summary_payload.get("text"))
            pair["summary_score"] = native_score

    return pairs


def source_chunk_ids_to_load(pairs: list[dict]) -> list[str]:
    return [
        pair["chunk_id"]
        for pair in pairs
        if pair["summary_rank"] is not None and pair["chunk"] is None and pair["chunk_id"]
    ]


def attach_source_chunks(pairs: list[dict], chunks: list[Any]) -> None:
    for chunk in chunks:
        pair = _find_chunk_summary_pair(pairs, result_id(chunk), None)
        if pair:
            set_pair_chunk(pair, chunk)


def summary_text_by_chunk_id(pairs: list[dict]) -> dict[str, str]:
    summaries = {}
    for pair in pairs:
        if pair["chunk_id"] and pair["summary_text"]:
            summaries[pair["chunk_id"]] = pair["summary_text"]
    return summaries


def summary_id_for_chunk(chunk_id: str) -> Optional[str]:
    try:
        chunk_uuid = UUID(chunk_id)
    except (TypeError, ValueError):
        return None
    return str(uuid5(chunk_uuid, "TextSummary"))


def set_pair_chunk(pair: dict, chunk: Any) -> None:
    pair["chunk"] = chunk
    pair["chunk_id"] = result_id(chunk) or pair["chunk_id"]
    pair["chunk_text"] = display_value(payload(chunk).get("text")) or pair["chunk_text"]


def _find_chunk_summary_pair(
    pairs: list[dict],
    chunk_id: Optional[str],
    chunk_text: Optional[str],
) -> Optional[dict]:
    for pair in pairs:
        if chunk_id and pair["chunk_id"] == chunk_id:
            return pair
        if chunk_text and pair["chunk_id"] is None and pair["chunk_text"] == chunk_text:
            return pair
    return None


def _new_chunk_summary_pair(
    chunk_id: Optional[str] = None,
    chunk_text: Optional[str] = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "summary_id": None,
        "summary_text": None,
        "chunk": None,
        "bm25_rank": None,
        "bm25_score": None,
        "vector_rank": None,
        "vector_score": None,
        "summary_rank": None,
        "summary_score": None,
    }


def _result_and_native_score(result: Any) -> tuple[Any, Optional[float]]:
    """Return a channel result and its uncalibrated backend score, when available."""
    if isinstance(result, (list, tuple)) and len(result) == 2:
        candidate, raw_score = result
    else:
        candidate = result
        raw_score = getattr(result, "score", None)

    if isinstance(raw_score, Real) and not isinstance(raw_score, bool):
        numeric_score = float(raw_score)
        if math.isfinite(numeric_score):
            return candidate, numeric_score
    return candidate, None
