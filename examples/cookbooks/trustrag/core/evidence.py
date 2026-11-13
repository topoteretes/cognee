"""Evidence Adapter for Cognee Search.

Bridges Cognee's search results and EvidenceReference objects into
TrustRAG's VerifiableEvidence model required for downstream verification,
fact checking, and trust score computation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VerifiableEvidence(BaseModel):
    """Normalized evidence item extracted from Cognee's storage layer."""

    chunk_id: str
    text: str = Field(description="Source text content for this evidence chunk")
    doc_name: Optional[str] = Field(default=None, description="Document or source filename")
    data_id: Optional[str] = Field(default=None, description="Cognee data identifier")
    chunk_index: Optional[int] = Field(default=None, description="Position within source document")
    score: float = Field(default=0.5, description="Retrieval relevance score")
    kind: str = Field(default="segment", description="Evidence kind (segment, graph_node, graph_edge)")
    dataset_name: Optional[str] = Field(default=None, description="Originating Cognee dataset")
    section: Optional[str] = Field(default=None, description="Section or heading in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary")


def _extract_val(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely extract an attribute or dictionary key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default
    for k in keys:
        val = getattr(obj, k, None)
        if val is not None:
            return val
    return default


def _extract_context_text(payload: Any) -> str:
    """Extract context or completion text from a search result payload or dictionary."""
    context = _extract_val(payload, "context", "context_result")
    if context is not None:
        if isinstance(context, str):
            return context
        if isinstance(context, (list, tuple)):
            return "\n\n".join(str(c) for c in context if c)

    completion = _extract_val(payload, "completion", "text_result")
    if completion is not None:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, (list, tuple)):
            return "\n\n".join(str(c) for c in completion if c)

    result_obj = _extract_val(payload, "result", "search_result", "result_object")
    if result_obj is not None:
        if isinstance(result_obj, str):
            return result_obj
        if isinstance(result_obj, (list, tuple)):
            texts = []
            for item in result_obj:
                t = _extract_val(item, "text", "content", default=str(item))
                if t:
                    texts.append(str(t))
            return "\n\n".join(texts)

    return ""


def payload_to_evidence(
    search_results: Any,
    top_k: int = 10,
    default_dataset: Optional[str] = None,
) -> List[VerifiableEvidence]:
    """Convert Cognee search output into a normalized list of VerifiableEvidence.

    Handles SearchResult wrapper objects, SearchResultPayload objects,
    raw chunk lists, and dictionaries.
    """
    if search_results is None:
        return []

    if not isinstance(search_results, (list, tuple)):
        items = [search_results]
    else:
        items = list(search_results)

    evidence_list: List[VerifiableEvidence] = []
    seen_chunk_ids: set[str] = set()

    for item in items:
        # Check if item is wrapped in Cognee's SearchResult(search_result=..., dataset_name=...)
        payload = _extract_val(item, "search_result", default=item)
        dataset_name = _extract_val(item, "dataset_name", default=None) or default_dataset

        error = _extract_val(payload, "error", default=None)
        if error:
            logger.warning("Cognee search item reported error: %s", error)
            continue

        context_text = _extract_context_text(payload)
        evidence_refs = _extract_val(payload, "evidence", default=[])
        result_object = _extract_val(payload, "result_object", default=None)

        if evidence_refs:
            for idx, ref in enumerate(evidence_refs):
                chunk_id = str(
                    _extract_val(ref, "chunk_id", "artifact_id", default=f"cognee_chunk_{len(evidence_list)}")
                )
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                doc_name = _extract_val(ref, "document_name", default=None)
                data_id = _extract_val(ref, "data_id", default=None)
                chunk_idx = _extract_val(ref, "chunk_index", default=idx)
                kind = _extract_val(ref, "kind", default="segment")
                source_text = _extract_val(ref, "source_text", default=None) or context_text

                score = _extract_val(ref, "score", default=None)
                if score is None:
                    rank = _extract_val(ref, "rank", default=None)
                    score = max(0.1, 1.0 - (rank * 0.1)) if rank is not None else 0.5

                evidence_list.append(
                    VerifiableEvidence(
                        chunk_id=chunk_id,
                        text=source_text,
                        doc_name=doc_name,
                        data_id=data_id,
                        chunk_index=chunk_idx,
                        score=float(score),
                        kind=str(kind),
                        dataset_name=dataset_name,
                    )
                )

        elif isinstance(result_object, (list, tuple)) and result_object:
            # Direct chunk search returns list of chunk items
            for idx, chunk in enumerate(result_object):
                chunk_id = str(
                    _extract_val(chunk, "id", "chunk_id", "artifact_id", default=f"chunk_{len(evidence_list)}")
                )
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                text = _extract_val(chunk, "text", "content", default="")
                doc_name = _extract_val(chunk, "document_name", "file_name", default=None)
                score = _extract_val(chunk, "score", default=1.0)

                evidence_list.append(
                    VerifiableEvidence(
                        chunk_id=chunk_id,
                        text=text,
                        doc_name=doc_name,
                        data_id=str(_extract_val(chunk, "document_id", default="")),
                        chunk_index=idx,
                        score=float(score) if score is not None else 1.0,
                        kind="segment",
                        dataset_name=dataset_name,
                    )
                )

        elif context_text:
            # Fallback when single context is returned without structured evidence refs
            chunk_id = f"context_{len(evidence_list)}"
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                evidence_list.append(
                    VerifiableEvidence(
                        chunk_id=chunk_id,
                        text=context_text,
                        doc_name=dataset_name or "knowledge_context",
                        score=1.0,
                        kind="context",
                        dataset_name=dataset_name,
                    )
                )

    evidence_list.sort(key=lambda e: -e.score)
    return evidence_list[:top_k]


def build_chunk_text_map(evidence_items: List[VerifiableEvidence]) -> Dict[str, str]:
    """Map chunk IDs to their corresponding text string for verification."""
    return {item.chunk_id: item.text for item in evidence_items if item.chunk_id and item.text}


async def search_cognee(
    query_text: str,
    dataset_name: Optional[str] = None,
    search_type: str = "GRAPH_COMPLETION",
    top_k: int = 10,
    include_references: bool = True,
) -> List[VerifiableEvidence]:
    """Execute Cognee search and return structured VerifiableEvidence."""
    import cognee
    from cognee import SearchType

    try:
        st = SearchType(search_type.upper().strip())
    except (ValueError, KeyError):
        st = SearchType.GRAPH_COMPLETION

    datasets = [dataset_name] if dataset_name else None

    raw_results = await cognee.search(
        query_text=query_text,
        query_type=st,
        datasets=datasets,
        top_k=top_k,
        include_references=include_references,
        verbose=True,
    )

    return payload_to_evidence(raw_results, top_k=top_k, default_dataset=dataset_name)


def search_cognee_sync(
    query_text: str,
    dataset_name: Optional[str] = None,
    search_type: str = "GRAPH_COMPLETION",
    top_k: int = 10,
    include_references: bool = True,
) -> List[VerifiableEvidence]:
    """Synchronous entry point for search_cognee."""
    return asyncio.run(
        search_cognee(
            query_text=query_text,
            dataset_name=dataset_name,
            search_type=search_type,
            top_k=top_k,
            include_references=include_references,
        )
    )
