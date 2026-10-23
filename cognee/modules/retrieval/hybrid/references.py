from typing import Any

from cognee.modules.retrieval.utils.references import render_citations, select_citations


def _chunk_lanes(completions: list, retrieved_objects) -> list[tuple[Any, list | None]]:
    """Pair each completion with the chunk lane the LLM read for it; None when it has none."""
    if isinstance(retrieved_objects, list):
        return [
            (
                completion,
                _chunks(retrieved_objects[index]) if index < len(retrieved_objects) else None,
            )
            for index, completion in enumerate(completions)
        ]
    return [(completion, _chunks(retrieved_objects)) for completion in completions]


def _chunks(retrieved_objects) -> list:
    return retrieved_objects.get("chunks", []) if isinstance(retrieved_objects, dict) else []


def cite_hybrid_completions(completions: list, retrieved_objects) -> tuple[list, list]:
    """Cite each answer from the chunk lane the LLM actually read.

    Returns the answers, each with its Evidence block appended, and the chunks
    those blocks cite, in citation order. One selection feeds both, so the
    structured evidence built from the chunks and the cited text always name
    the same chunks. Non-string answers and answers without a lane are returned
    unchanged and cite nothing.
    """
    cited_answers: list = []
    cited_chunks: list = []
    for completion, chunks in _chunk_lanes(completions, retrieved_objects):
        citations = (
            select_citations(chunks, answer=completion)
            if chunks is not None and isinstance(completion, str)
            else []
        )
        evidence = render_citations(citations)
        cited_answers.append(f"{completion}\n\n{evidence}" if evidence else completion)
        cited_chunks.extend(citation[0] for citation in citations)
    return cited_answers, cited_chunks
