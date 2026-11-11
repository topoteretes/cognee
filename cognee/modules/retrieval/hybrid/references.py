from collections.abc import Mapping
from typing import Any

from cognee.modules.retrieval.utils.document_sources import DocumentSource
from cognee.modules.retrieval.utils.references import append_chunk_evidence


def hybrid_chunks(retrieved_objects: Any) -> list:
    """Every chunk of the chunk lane, across the queries of a batch."""
    if isinstance(retrieved_objects, list):
        return [chunk for objects in retrieved_objects for chunk in hybrid_chunks(objects)]
    if isinstance(retrieved_objects, dict):
        return list(retrieved_objects.get("chunks", []))
    return []


def cite_hybrid_completions(
    completions: list,
    retrieved_objects,
    enabled: bool,
    sources: Mapping[str, DocumentSource] | None = None,
) -> list:
    """Cite each answer from the chunk lane the LLM actually read."""
    if not enabled:
        return completions
    if isinstance(retrieved_objects, list):
        cited = []
        for index, completion in enumerate(completions):
            if index >= len(retrieved_objects):
                cited.append(completion)
                continue
            cited.extend(
                cite_hybrid_completions(
                    [completion], retrieved_objects[index], enabled=True, sources=sources
                )
            )
        return cited

    chunks = retrieved_objects.get("chunks", []) if isinstance(retrieved_objects, dict) else []
    return append_chunk_evidence(completions, chunks, enabled=True, sources=sources)
