"""The uncut result of a hybrid fetch, and the one step that cuts it.

A hybrid fetch produces more than it shows: ranked chunks, ranked entities with
their edge bullets, and the raw fact hits. ``HybridCandidates`` carries that as
a value, so a retriever can reorder the lists before showing them (the temporal
rerank) without the fetch having decided anything.

``finalize`` is the only place a candidate list is cut. It cuts chunks and
entities to the limits it is given and selects the standalone facts against
the entities it keeps, so a fact is dropped as "already shown under entity X"
only when X is shown.
"""

from dataclasses import dataclass, field, replace

from cognee.modules.retrieval.hybrid.facts import FactCandidates, select_facts_from_candidates
from cognee.modules.retrieval.hybrid.results import result_id


@dataclass(frozen=True)
class HybridCandidates:
    """Everything one hybrid fetch found, in rank order, before any cut."""

    chunks: list = field(default_factory=list)
    chunk_summaries: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    fact_candidates: FactCandidates = field(default_factory=FactCandidates)

    def prioritize(self, chunk_ids: set[str], entity_ids: set[str]) -> "HybridCandidates":
        """The same candidates with the named chunks and entities moved to the front.

        A stable partition: the fetch order is kept within each half, nothing is
        removed or altered.
        """
        return replace(
            self,
            chunks=self._first(self.chunks, chunk_ids),
            entities=self._first(self.entities, entity_ids),
        )

    def finalize(
        self, *, chunks_limit: int, entities_limit: int, entity_edge_budget: int | None = None
    ) -> dict:
        """Cut to the limits and select facts against the entities kept.

        ``entity_edge_budget`` is the fact budget spent when no entity is kept
        (``resolve_facts_top_k``). The fetch sizes it for its own entity limit;
        a caller that cuts to a smaller limit passes the budget that matches
        what it shows, so facts cannot outgrow the entity lane they replace.
        """
        chunks = list(self.chunks)[:chunks_limit]
        entities = list(self.entities)[:entities_limit]
        fact_candidates = self.fact_candidates
        if entity_edge_budget is not None:
            fact_candidates = replace(fact_candidates, entity_edge_budget=entity_edge_budget)
        return {
            "chunks": chunks,
            "chunk_summaries": self._summaries_for(self.chunk_summaries, chunks),
            "entities": entities,
            "facts": select_facts_from_candidates(fact_candidates, entities),
        }

    @staticmethod
    def _first(items: list, ids: set[str]) -> list:
        first = [item for item in items if result_id(item) in ids]
        rest = [item for item in items if result_id(item) not in ids]
        return first + rest

    @staticmethod
    def _summaries_for(summaries: dict, chunks: list) -> dict:
        chunk_ids = {result_id(chunk) for chunk in chunks}
        return {key: value for key, value in (summaries or {}).items() if str(key) in chunk_ids}
