"""Fail-open embedding helpers for the session layer.

Session entries (QA history and active context guidance) store an optional embedding
vector inline in the cache, computed at write time. Selection at read time is a
brute-force cosine pass in Python: a session holds tens of entries, so no vector
collection (and no collection lifecycle) is needed. Every helper degrades to the
pre-embedding behavior when a vector is missing or the embedding call fails.
"""

import math
from typing import List, Optional, Sequence

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_embeddings")

# Number of semantically recalled QA turns added on top of the recency window.
SEMANTIC_TOP_K = 3
# Minimum cosine similarity for a QA turn to be semantically recalled. QA entries embed
# question + answer jointly, so on-topic turns score noticeably lower than identical text
# would; measured across demo runs: 0.36-0.41 (on-topic) vs 0.18-0.19 (off-topic).
MIN_QA_SIMILARITY = 0.30
# Cosine similarity above which a candidate context entry counts as a duplicate.
NEAR_DUP_SIMILARITY = 0.9


async def embed_text_safe(text: str) -> Optional[List[float]]:
    """Embed one text with the configured embedding engine; return None on any failure."""
    if not text or not text.strip():
        return None
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        vectors = await vector_engine.embedding_engine.embed_text([text])
        if vectors and vectors[0]:
            return list(vectors[0])
        return None
    except Exception as error:
        logger.warning("Session embedding failed open: %s", error)
        return None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 for missing, mismatched, or zero vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def select_hybrid_qa_entries(
    entries: list,
    query_embedding: Optional[List[float]],
    *,
    last_n: int,
    semantic_top_k: int = SEMANTIC_TOP_K,
    min_similarity: float = MIN_QA_SIMILARITY,
) -> list:
    """Select QA history as the union of the last N turns and the semantic top K.

    ``entries`` must be in chronological order (oldest first), as returned by the cache
    adapters. The recency window keeps the conversation coherent; the semantic picks
    recall relevant turns that fell outside it. The union is deduplicated and returned
    in chronological order so it still reads as a conversation.

    With no query embedding this is exactly the old behavior: the last N entries.
    Entries without an embedding can never be semantically recalled but still appear
    via the recency window.
    """
    if last_n <= 0:
        return []

    recent = entries[-last_n:]
    if query_embedding is None or semantic_top_k <= 0:
        return list(recent)

    older = entries[:-last_n] if len(entries) > last_n else []
    scored = []
    for index, entry in enumerate(older):
        embedding = getattr(entry, "embedding", None)
        if embedding is None and isinstance(entry, dict):
            embedding = entry.get("embedding")
        if not embedding:
            continue
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity >= min_similarity:
            scored.append((similarity, index, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    recalled_indices = sorted(index for _, index, _ in scored[:semantic_top_k])
    recalled = [older[index] for index in recalled_indices]
    return recalled + list(recent)
