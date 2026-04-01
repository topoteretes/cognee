"""Pure RAG benchmark implementation.

Uses Cognee's embedding engine + in-memory cosine similarity search.
No cognify, no graph, no entity extraction — just embed, store, retrieve.
Represents a standard RAG setup without Cognee's knowledge graph.
"""

from __future__ import annotations

import numpy as np

from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from .agents import (
    format_trace_summary,
    run_conversation,
    sales_agent_turn,
    setup_runtime,
)
from .leads import LEADS, BuyingProfile
from .metrics import MetricsCollector
from .models import ConversationResult, SalesResponse

TOP_K = 3  # Match graph mode's top_k for fair comparison


class InMemoryVectorStore:
    """Minimal in-memory vector store with cosine similarity search."""

    def __init__(self):
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, text: str, vector: list[float]) -> None:
        self._texts.append(text)
        self._vectors.append(vector)

    def search(self, query_vector: list[float], top_k: int = TOP_K) -> list[str]:
        if not self._vectors:
            return []
        vecs = np.array(self._vectors)
        q = np.array(query_vector)
        # Cosine similarity
        norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1, norms)
        similarities = vecs @ q / norms
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self._texts[i] for i in top_indices if similarities[i] > 0.0]


# Per-persona vector stores
_stores: dict[str, InMemoryVectorStore] = {}
_embedding_engine = None


async def _get_embedding_engine():
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = get_embedding_engine()
    return _embedding_engine


async def _embed(text: str) -> list[float]:
    engine = await _get_embedding_engine()
    vectors = await engine.embed_text([text])
    return vectors[0]


async def _query_memory(profile: BuyingProfile) -> str:
    """Retrieve past traces via vector similarity search + LLM synthesis."""
    store = _stores.get(profile.persona_tag)
    if store is None or not store._texts:
        return ""

    query = (
        f"What sales approaches worked or failed for {profile.persona_tag} customers? "
        f"Which Cognee feature and pitch angle closed the deal or lost it?"
    )
    query_vec = await _embed(query)
    chunks = store.search(query_vec, top_k=TOP_K)
    if not chunks:
        return ""

    # Return raw chunks — the sales agent will interpret them directly
    return "\n---\n".join(chunks)


async def _save_trace(profile: BuyingProfile, result: ConversationResult) -> None:
    """Embed and store the trace summary in the per-persona vector store."""
    summary = format_trace_summary(profile, result)
    vector = await _embed(summary)
    store = _stores.setdefault(profile.persona_tag, InMemoryVectorStore())
    store.add(summary, vector)


async def sales_turn_rag(
    conversation_history: list, lead_intro: str, round_num: int, memory_context: str
) -> SalesResponse:
    return await sales_agent_turn(conversation_history, lead_intro, round_num, memory_context)


async def setup_rag() -> None:
    global _stores, _embedding_engine
    _stores = {}
    _embedding_engine = None
    await setup_runtime()


async def run_all_leads(collector: MetricsCollector) -> list:
    results = []
    for lead in LEADS:
        print(f"\n--- Lead {lead.lead_id}: {lead.persona_tag} ---")

        memory_context = await _query_memory(lead)
        if memory_context:
            print(f"  [rag] Retrieved context: {memory_context[:120]}...")

        collector.start_lead(lead.lead_id, lead.persona_tag, "rag")

        result = await run_conversation(
            sales_turn_rag,
            lead,
            memory_context=memory_context,
        )

        collector.end_lead(result)
        results.append(result)
        print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")

        await _save_trace(lead, result)
    return results
