"""Temporal rerank on top of HybridRetriever: the TEMPORAL search type (SDK-828).

The candidate fetch and the query-interval extraction run concurrently. The
graph adapter is then asked which of the candidate chunks and entities are
attached to a Timestamp inside the window (``get_temporal_anchors``, a native
query on Ladybug, Neo4j and the Postgres demo, a neighbourhood walk elsewhere),
and the oversized candidate set is reordered so the anchored candidates come
first before the final limit. Facts are selected against the entities that
survive that cut. Context formatting and completion are inherited unchanged.

get_retrieved_objects returns the plain hybrid result shape — the reranked
view, or the baseline slice on fallback. Diagnostics for the last query live
on the instance: last_interval, last_reason, last_anchors, last_baseline.
"""

import asyncio

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.hybrid.facts import FactCandidates, select_facts_from_candidates
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result, result_id
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.temporal_hybrid.matching import (
    empty_anchors,
    extract_query_interval,
    rerank_hybrid,
    slice_hybrid,
    to_epoch_ms,
)


class TemporalHybridRetriever(HybridRetriever):
    """Hybrid retrieval reranked by temporal overlap, falling back to plain hybrid."""

    def __init__(self, candidate_top_k: int | None = None, top_k: int | None = 5, **kwargs):
        # The REST request models accept ``top_k: null`` and the registry passes it
        # through unchanged; resolve it here, before the arithmetic, the way
        # HybridRetriever's lanes fall back to their own defaults.
        if top_k is None:
            top_k = 5
        if candidate_top_k is None:
            candidate_top_k = top_k * 4
        if top_k <= 0 or candidate_top_k < top_k:
            raise ValueError("limits must be positive with top_k <= candidate_top_k")
        super().__init__(
            chunks_top_k=candidate_top_k,
            entities_top_k=candidate_top_k,
            facts_top_k=top_k,
            **kwargs,
        )
        self.top_k = top_k
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.last_interval = (None, None)
        self.last_reason = None
        self.last_anchors = empty_anchors()
        self.last_baseline = empty_hybrid_result()
        self._fact_candidates: FactCandidates | None = None

    def _remember_fact_candidates(self, fact_candidates: FactCandidates) -> None:
        # One query per call (query_batch is refused below), so keeping the
        # candidates on the instance is safe here where it is not for Hybrid.
        self._fact_candidates = fact_candidates

    def _with_facts(self, result: dict) -> dict:
        """Facts selected against the entities ``result`` actually shows.

        Hybrid deduplicated its facts against the whole candidate entity set;
        after the cut to ``top_k`` a fact shown only under a discarded entity
        would be gone from both sections. Without the candidates (a caller that
        bypassed Hybrid's fetch) the inherited facts stand.
        """
        if self._fact_candidates is None:
            return result
        facts = select_facts_from_candidates(self._fact_candidates, result.get("entities") or [])
        return {**result, "facts": facts}

    async def _anchors(self, start, end, candidates: dict) -> dict:
        """Which candidate chunks and entities are attached to a time in the window."""
        graph = await get_graph_engine()
        chunk_ids = {
            chunk_id
            for chunk_id in (result_id(chunk) for chunk in candidates.get("chunks") or [])
            if chunk_id
        }
        entity_ids = {
            str(entity["id"])
            for entity in candidates.get("entities") or []
            if isinstance(entity, dict) and entity.get("id")
        }
        anchors = await graph.get_temporal_anchors(
            chunk_ids, entity_ids, to_epoch_ms(start), to_epoch_ms(end)
        )
        return {
            "timestamp_ids": set(anchors.get("timestamp_ids") or ()),
            "chunk_ids": set(anchors.get("chunk_ids") or ()),
            "entity_ids": set(anchors.get("entity_ids") or ()),
        }

    async def get_retrieved_objects(self, query=None, query_batch=None) -> dict:
        if query_batch:
            raise NotImplementedError("TemporalHybridRetriever answers one query at a time")
        if not str(query or "").strip():
            raise ValueError("query must not be blank")
        self._reset_diagnostics()

        # Duplicates super()'s emptiness check on purpose: returning here keeps
        # the empty-graph path free of LLM and embedding calls.
        self._unified_engine = await get_unified_engine()
        if await self._unified_engine.graph.is_empty():
            self.last_reason = "empty_graph"
            return empty_hybrid_result()

        candidates, (start, end, reason) = await asyncio.gather(
            super().get_retrieved_objects(query=query),
            extract_query_interval(query),
        )
        self.last_interval = (start, end)
        self.last_baseline = self._with_facts(slice_hybrid(candidates, self.top_k))
        if reason is not None:
            self.last_reason = reason
            return self.last_baseline

        self.last_anchors = await self._anchors(start, end, candidates)
        if not self.last_anchors["timestamp_ids"]:
            # Nothing in the candidate set is dated inside the window. Tell the
            # two cases apart for the diagnostics: a window the graph has no
            # time in at all, or one whose matches lie outside the candidates.
            graph = await get_graph_engine()
            in_window = await graph.get_timestamps_in_range(to_epoch_ms(start), to_epoch_ms(end))
            self.last_reason = "no_candidate_overlap" if in_window else "no_temporal_match"
            return self.last_baseline

        reranked = self._with_facts(rerank_hybrid(candidates, self.last_anchors, self.top_k))
        if reranked["chunks"] == self.last_baseline["chunks"] and (
            reranked["entities"] == self.last_baseline["entities"]
        ):
            # Informational: the window matched, but nothing in the candidate set
            # moved. The result is the baseline either way.
            self.last_reason = "no_candidate_overlap"
        return reranked
