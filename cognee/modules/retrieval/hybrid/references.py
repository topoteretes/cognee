from typing import Any

from cognee.modules.retrieval.utils.references import (
    answer_grounded_chunks,
    append_chunk_evidence,
)


def _chunk_lanes(completions: list, retrieved_objects) -> list[tuple[Any, list]]:
    """Pair each completion with the chunk lane the LLM read for it."""
    if isinstance(retrieved_objects, list):
        return [
            (completion, _chunks(retrieved_objects[index]))
            for index, completion in enumerate(completions)
            if index < len(retrieved_objects)
        ]
    return [(completion, _chunks(retrieved_objects)) for completion in completions]


def _chunks(retrieved_objects) -> list:
    return retrieved_objects.get("chunks", []) if isinstance(retrieved_objects, dict) else []


def cite_hybrid_completions(completions: list, retrieved_objects, enabled: bool) -> list:
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
                cite_hybrid_completions([completion], retrieved_objects[index], enabled=True)
            )
        return cited

    return append_chunk_evidence(completions, _chunks(retrieved_objects), enabled=True)


def hybrid_cited_chunks(completions: list, retrieved_objects) -> list:
    """The chunks :func:`cite_hybrid_completions` cites, as retrieved objects.

    Same selection as the Evidence block, so structured evidence and the cited
    text always name the same chunks. Non-string completions cite nothing.
    """
    return [
        chunk
        for completion, chunks in _chunk_lanes(completions, retrieved_objects)
        if isinstance(completion, str)
        for chunk in answer_grounded_chunks(chunks, answer=completion)
    ]
