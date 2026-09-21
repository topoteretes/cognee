"""Temporal filter on top of cognee's HybridRetriever.

The candidate fetch and the query-interval extraction run concurrently; the
temporal index (built once per instance from one graph snapshot) then selects
the eligible chunks, which filter the oversized candidate set before the final
limit. Context formatting and completion are inherited unchanged.

get_retrieved_objects returns the plain hybrid result shape — the filtered
view, or the baseline slice on fallback. Diagnostics for the last query live
on the instance: last_interval, last_reason, last_matches, last_baseline.
"""

import asyncio

from temporal_matching import (
    build_temporal_index,
    empty_matches,
    extract_query_interval,
    filter_hybrid,
    match_temporal_neighborhood,
    slice_hybrid,
)

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


class TemporalHybridRetriever(HybridRetriever):
    """Hybrid retrieval filtered by temporal overlap, falling back to plain hybrid.

    The temporal index is one graph snapshot; build one retriever per graph state.
    """

    def __init__(self, candidate_top_k: int = 20, top_k: int = 5, **kwargs):
        if top_k <= 0 or candidate_top_k < top_k:
            raise ValueError("limits must be positive with top_k <= candidate_top_k")
        super().__init__(
            chunks_top_k=candidate_top_k,
            entities_top_k=candidate_top_k,
            facts_top_k=top_k,
            **kwargs,
        )
        self.top_k = top_k
        self._temporal_index = None
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.last_interval = (None, None)
        self.last_reason = None
        self.last_matches = empty_matches()
        self.last_baseline = empty_hybrid_result()

    async def _index(self) -> dict:
        if self._temporal_index is None:
            graph = await get_graph_engine()
            self._temporal_index = build_temporal_index(*await graph.get_graph_data())
        return self._temporal_index

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
        self.last_baseline = slice_hybrid(candidates, self.top_k)
        if reason is not None:
            self.last_reason = reason
            return self.last_baseline

        self.last_matches = match_temporal_neighborhood(await self._index(), start, end)
        if not self.last_matches["eligible_chunk_ids"]:
            self.last_reason = "no_temporal_match"
            return self.last_baseline

        filtered = filter_hybrid(candidates, self.last_matches, self.top_k)
        if filtered is None:
            self.last_reason = "no_candidate_overlap"
            return self.last_baseline
        return filtered
