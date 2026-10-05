"""The uncut result of a hybrid fetch, and the one step that cuts it.

A hybrid fetch produces more than it shows: ranked chunks, ranked entities with
their edge bullets, and the raw fact hits. ``HybridCandidates`` carries that as
a value, so a retriever that reorders the lists before showing them (the
temporal rerank) can do so without the fetch having decided anything.

``finalize`` is the only place a candidate list is cut. It cuts chunks and
entities to the limits it is given and selects the standalone facts against
the entities it keeps, so a fact is dropped as "already shown under entity X"
only when X is shown.
"""

from dataclasses import dataclass, field

from cognee.modules.retrieval.hybrid.facts import FactCandidates, select_facts_from_candidates
from cognee.modules.retrieval.hybrid.results import result_id


@dataclass(frozen=True)
class HybridCandidates:
    """Everything one hybrid fetch found, in rank order, before any cut."""

    chunks: list = field(default_factory=list)
    chunk_summaries: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    fact_candidates: FactCandidates = field(default_factory=FactCandidates)


def finalize(candidates: HybridCandidates, *, chunks_limit: int, entities_limit: int) -> dict:
    """Cut the candidates to the limits and select facts against the entities kept."""
    chunks = list(candidates.chunks)[:chunks_limit]
    entities = list(candidates.entities)[:entities_limit]
    return {
        "chunks": chunks,
        "chunk_summaries": summaries_for(candidates.chunk_summaries, chunks),
        "entities": entities,
        "facts": select_facts_from_candidates(candidates.fact_candidates, entities),
    }


def summaries_for(summaries: dict, chunks: list) -> dict:
    """The summaries that belong to ``chunks``, keyed as they were."""
    chunk_ids = {result_id(chunk) for chunk in chunks}
    return {key: value for key, value in (summaries or {}).items() if str(key) in chunk_ids}
