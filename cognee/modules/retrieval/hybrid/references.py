from cognee.modules.retrieval.utils.references import append_chunk_evidence


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

    chunks = retrieved_objects.get("chunks", []) if isinstance(retrieved_objects, dict) else []
    return append_chunk_evidence(completions, chunks, enabled=True)
