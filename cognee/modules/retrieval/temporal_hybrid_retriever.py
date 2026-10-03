"""Temporal rerank on top of HybridRetriever: the TEMPORAL search type (SDK-828).

The candidate fetch and the query-interval extraction run concurrently. The
graph adapter then returns the Timestamp nodes inside the window
(``get_timestamps_in_range``, a native query on Ladybug, Neo4j and the Postgres
demo, a scan elsewhere), their one-hop neighbourhood names the chunks and
entities anchored to them, and the oversized candidate set is reordered so the
anchored candidates come first before the final limit. Context formatting and
completion are inherited unchanged.

get_retrieved_objects returns the plain hybrid result shape — the reranked
view, or the baseline slice on fallback. Diagnostics for the last query live
on the instance: last_interval, last_reason, last_anchors, last_baseline.
"""

import asyncio

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.temporal_hybrid.matching import (
    anchors_from_neighborhood,
    chunks_containing,
    empty_anchors,
    extract_query_interval,
    rerank_hybrid,
    slice_hybrid,
    to_epoch_ms,
)

# Bounds on the neighbourhood reads: one question rarely names a window with
# more matching timestamps than this, and past it the anchored set already
# covers far more than the candidate set can hold.
MAX_MATCHED_TIMESTAMPS = 500


class TemporalHybridRetriever(HybridRetriever):
    """Hybrid retrieval reranked by temporal overlap, falling back to plain hybrid."""

    def __init__(self, candidate_top_k: int | None = None, top_k: int = 5, **kwargs):
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

    async def _anchors(self, start, end) -> dict:
        """Timestamps in the window and the chunks and entities attached to them."""
        graph = await get_graph_engine()
        timestamps = await graph.get_timestamps_in_range(to_epoch_ms(start), to_epoch_ms(end))
        timestamp_ids = {str(node["id"]) for node in timestamps[:MAX_MATCHED_TIMESTAMPS]}
        if not timestamp_ids:
            return empty_anchors()
        nodes, edges = await graph.get_neighborhood(sorted(timestamp_ids), depth=1)
        anchors = anchors_from_neighborhood(timestamp_ids, nodes, edges)
        # An entity anchored to a time carries that time into every chunk that
        # mentions it — the reception held in July 1805 dates the chunks about
        # the reception, not just the one that names the month.
        if anchors["entity_ids"]:
            entity_nodes, entity_edges = await graph.get_neighborhood(
                sorted(anchors["entity_ids"]), depth=1, edge_types=["contains"]
            )
            anchors["chunk_ids"] |= chunks_containing(
                anchors["entity_ids"], entity_nodes, entity_edges
            )
        return {**anchors, "timestamp_ids": timestamp_ids}

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

        self.last_anchors = await self._anchors(start, end)
        if not self.last_anchors["timestamp_ids"]:
            self.last_reason = "no_temporal_match"
            return self.last_baseline

        reranked = rerank_hybrid(candidates, self.last_anchors, self.top_k)
        if reranked["chunks"] == self.last_baseline["chunks"] and (
            reranked["entities"] == self.last_baseline["entities"]
        ):
            # Informational: the window matched, but nothing in the candidate set
            # moved. The result is the baseline either way.
            self.last_reason = "no_candidate_overlap"
        return reranked
