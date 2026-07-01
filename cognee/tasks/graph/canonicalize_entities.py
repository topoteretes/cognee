"""LLM-judge entity canonicalization task (issue #3629).

Commit 2 scope: entity gathering + blocking (candidate generation) only. The
judge, union-find, mirror-loser-onto-winner merge primitive, and audit are added
in a later commit; ``canonicalize_entities`` is a gather-only stub for now and is
not yet wired into the cognify pipeline.

Design notes (grounded in the extraction path):
- After ``extract_graph_and_summarize`` the objects flowing toward storage are
  ``TextSummary`` instances; entities live nested under
  ``TextSummary.made_from.contains`` as ``(Edge, Entity)`` tuples (or bare
  ``Entity``), never as top-level items.
- Blocking embeds entity names transiently (no vector-index write) and keeps only
  pairs whose cosine similarity clears a threshold, so the LLM judge added later
  only ever sees genuine near-duplicates.
"""

from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.models import Entity
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import TextSummary

logger = get_logger("canonicalize_entities")


def _gather_entities(summaries: List) -> List[Entity]:
    """Collect the unique ``Entity`` objects reachable from a list of TextSummary.

    Walks ``TextSummary.made_from.contains``, handling both the ``(Edge, Entity)``
    tuple shape produced by extraction and bare ``Entity`` items. ``Event`` items,
    ``(Edge, non-Entity)`` tuples, and non-TextSummary rows (e.g. raw DLT rows that
    ``summarize_text`` passes through) are skipped. Dedup is by ``str(id)`` and
    order-stable (first occurrence wins).
    """
    seen = {}
    for summary in summaries:
        if not isinstance(summary, TextSummary):
            # Non-TextSummary rows (e.g. DLT rows appended by summarize_text) carry
            # no made_from.contains entity graph to canonicalize.
            continue

        chunk = getattr(summary, "made_from", None)
        for item in getattr(chunk, "contains", None) or []:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Entity):
                entity = item[1]
            elif isinstance(item, Entity):
                entity = item
            else:
                # Event, or (Edge, non-Entity) — not a canonicalization candidate.
                continue

            seen.setdefault(str(entity.id), entity)

    return list(seen.values())


def _dedup_relations(relations: List) -> List:
    """Deduplicate outbound relation tuples by (relationship_type, target id).

    Relations are ``(Edge, target)`` tuples (see expand_with_nodes_and_edges); the
    first occurrence of a given (relationship_type, target-id) pair is kept. Items
    that are not well-formed relation tuples are passed through unchanged.
    """
    seen = set()
    result = []
    for relation in relations:
        if isinstance(relation, tuple) and len(relation) == 2:
            edge, target = relation
            relationship_type = getattr(edge, "relationship_type", None)
            key = (relationship_type, str(getattr(target, "id", target)))
            if key in seen:
                continue
            seen.add(key)
        result.append(relation)
    return result


async def _block_candidate_pairs(entities: List[Entity], cfg) -> List[Tuple[Entity, Entity]]:
    """Produce candidate near-duplicate entity pairs via embedding cosine similarity.

    Entity names are embedded transiently (no vector-index write). Pairs whose
    cosine similarity is at or above ``cfg.canonicalization_similarity_threshold``
    are returned, highest-similarity first, capped at
    ``cfg.canonicalization_max_pairs`` (the drop is logged — never silently
    truncated). Identical-id pairs are skipped (already merged by construction).
    """
    names = [entity.name for entity in entities]

    vector_engine = get_vector_engine()
    vectors = await vector_engine.embedding_engine.embed_text(names)

    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard zero-norm rows so they yield 0 similarity instead of NaN.
    norms[norms == 0] = 1.0
    unit = matrix / norms
    similarities = unit @ unit.T

    scored_pairs = []
    for i in range(len(entities)):
        for k in range(i + 1, len(entities)):
            if str(entities[i].id) == str(entities[k].id):
                continue
            similarity = float(similarities[i][k])
            if similarity >= cfg.canonicalization_similarity_threshold:
                scored_pairs.append((similarity, entities[i], entities[k]))

    scored_pairs.sort(key=lambda scored: scored[0], reverse=True)

    max_pairs = cfg.canonicalization_max_pairs
    if len(scored_pairs) > max_pairs:
        logger.info(
            "canonicalization blocking capping candidate pairs",
            extra={"found": len(scored_pairs), "kept": max_pairs},
        )
        scored_pairs = scored_pairs[:max_pairs]

    return [(a, b) for _similarity, a, b in scored_pairs]


@task_summary("Canonicalized entities in {n} summary(ies)")
async def canonicalize_entities(
    summaries: List, ctx: Optional[SimpleNamespace] = None, **kwargs
) -> List:
    """Reconcile duplicate entities across a batch of TextSummary objects.

    Commit 2 stub: gathers candidate entities and returns the summaries unchanged.
    Blocking (``_block_candidate_pairs``) and the relation-dedup helper
    (``_dedup_relations``) are implemented and unit-tested but not yet driven from
    here — that happens once the LLM judge + merge primitive land.
    """
    entities = _gather_entities(summaries)
    if len(entities) < 2:
        return summaries

    # commit 4: judge + merge — block candidate pairs (_block_candidate_pairs),
    # judge them via LLMGateway, collapse transitive chains with union-find, apply
    # the mirror-loser-onto-winner primitive (with _dedup_relations), and emit the
    # structured merge audit. Until then this task is a no-op pass-through and is
    # not wired into the cognify pipeline.
    return summaries
