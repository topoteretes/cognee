"""LLM-judge entity canonicalization task (issue #3629).

Pipeline seam: runs after ``extract_graph_and_summarize`` and before
``add_data_points``. It gathers the entities produced by extraction, blocks them
into genuine near-duplicate candidate pairs (embedding cosine), asks an LLM judge
whether each pair is the same real-world entity, and — for confident matches —
merges duplicates by renaming the loser onto the winner's canonical identity so
the existing UUID5 dedup in ``deduplicate_nodes_and_edges`` collapses them at
storage.

Design notes (grounded in the extraction path):
- After ``extract_graph_and_summarize`` the objects flowing toward storage are
  ``TextSummary`` instances; entities live nested under
  ``TextSummary.made_from.contains`` as ``(Edge, Entity)`` tuples (or bare
  ``Entity``), never as top-level items.
- Blocking embeds entity names transiently (no vector-index write) and keeps only
  pairs whose cosine similarity clears a threshold, so the LLM judge only ever
  sees genuine near-duplicates.
- The merge collapses each confirmed-duplicate component onto a deterministic
  winner (see ``_merge_component``): every member is made IDENTICAL to the winner
  — same id, unioned relations, reconciled description, provenance — so
  ``get_graph_from_model`` yields the same canonical node no matter which member it
  reaches first (traversal order is extraction-driven, not tied to winner
  selection), and the existing UUID5 dedup drops the rest. No active
  edge-reassignment: edges follow automatically because they are live object
  references whose endpoint ids are read at ``get_graph_from_model`` time.
"""

import asyncio
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.models import Entity
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.models import CanonicalizationJudgment
from cognee.tasks.summarization.models import TextSummary

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

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


def _render_pairs(indexed_pairs: List[Tuple[int, Tuple[Entity, Entity]]]) -> str:
    """Render a batch of (pair_index, (a, b)) into judge-readable text."""
    blocks = []
    for pair_index, (a, b) in indexed_pairs:
        blocks.append(
            f"Pair {pair_index}:\n"
            f"  Entity A: name={a.name!r} description={a.description!r}\n"
            f"  Entity B: name={b.name!r} description={b.description!r}"
        )
    return "\n\n".join(blocks)


async def _judge_pairs(
    pairs: List[Tuple[Entity, Entity]], cfg
) -> List[Tuple[object, Entity, Entity]]:
    """Ask the LLM judge to rule on each candidate pair, batched and concurrent.

    Returns a list of ``(PairJudgment, entity_a, entity_b)`` with each judgment
    mapped back to its pair via ``PairJudgment.pair_index`` (out-of-range indices
    are dropped defensively).
    """
    batch_size = max(1, cfg.canonicalization_judge_batch_size)
    indexed = list(enumerate(pairs))
    batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]

    system_prompt = render_prompt("canonicalize_entities.txt", {})

    async def judge_batch(batch):
        result = await LLMGateway.acreate_structured_output(
            text_input=_render_pairs(batch),
            system_prompt=system_prompt,
            response_model=CanonicalizationJudgment,
        )
        return result.judgments

    batch_results = await asyncio.gather(*[judge_batch(batch) for batch in batches])

    mapped = []
    for judgments in batch_results:
        for judgment in judgments:
            if 0 <= judgment.pair_index < len(pairs):
                a, b = pairs[judgment.pair_index]
                mapped.append((judgment, a, b))
    return mapped


def _select_winner(members: List[Entity], canonical_name: str) -> Entity:
    """Deterministically pick the surviving entity for a merge component.

    A member whose normalized name equals the normalized canonical name wins
    (ties broken by smallest ``str(id)``); otherwise the member with the smallest
    normalized name wins. Order-independent.
    """
    target = Entity._normalize_identity_value(canonical_name)
    exact = [m for m in members if Entity._normalize_identity_value(m.name) == target]
    if exact:
        return min(exact, key=lambda m: str(m.id))
    return min(members, key=lambda m: Entity._normalize_identity_value(m.name))


def _log_merge(winner: Entity, loser: Entity, judgment, ctx) -> None:
    """Emit one structured audit record for a single loser→winner merge.

    Reads the loser's original name/description, so it must be called BEFORE the
    winner is mirrored onto the loser.
    """
    logger.info(
        "entity_canonicalization_merge",
        extra={
            "canonical_id": str(winner.id),
            "canonical_name": winner.name,
            "merged_id": str(loser.id),
            "merged_name": loser.name,
            "confidence": judgment.confidence,
            "rationale": judgment.rationale,
            "property_diff": {"description": [loser.description, winner.description]},
            "pipeline_run_id": str(getattr(ctx, "pipeline_run_id", None)),
            "dataset_id": str(getattr(getattr(ctx, "dataset", None), "id", None)),
        },
    )


def _merge_component(winner: Entity, losers: List[Entity], judgment, ctx) -> None:
    """Collapse a confirmed-duplicate component onto ``winner``, order-independently.

    ``get_graph_from_model`` builds the single surviving node from whichever member
    it reaches first (extraction/traversal order, unrelated to winner selection),
    then skips the rest by shared id. So every member must end up IDENTICAL for the
    collapse to be lossless: this unions all members' outbound relations, reconciles
    the description, stamps the merge provenance on the winner, then mirrors the
    finalized winner onto every loser field-for-field. Mirroring only some fields
    (as a pairwise merge would) silently drops the others' edges / provenance / type
    whenever a loser is the first member traversed.

    The canonical id is the winner's own id (derived from its name), never a raw LLM
    ``canonical_name`` — so the merged node cannot collide with an unrelated entity
    that happens to share the judge's chosen surface form.
    """
    reconciled = (judgment.reconciled_description or "").strip()
    if reconciled:
        winner.description = reconciled

    relations = list(winner.relations)
    for loser in losers:
        relations.extend(loser.relations)
    winner.relations = _dedup_relations(relations)

    aliases = list(winner.merged_aliases or [])
    for loser in losers:
        _log_merge(winner, loser, judgment, ctx)
        if loser.name not in aliases:
            aliases.append(loser.name)
    winner.merged_aliases = aliases
    winner.merge_confidence = judgment.confidence
    winner.source_task = "canonicalize_entities"
    winner.update_version()

    # Mirror the finalized winner onto every loser so the collapsed node is the
    # same regardless of which member get_graph_from_model reaches first.
    for loser in losers:
        for field_name in type(winner).model_fields:
            setattr(loser, field_name, getattr(winner, field_name))


def _apply_merges(
    entities: List[Entity], judgments: List[Tuple[object, Entity, Entity]], cfg, ctx
) -> None:
    """Collapse confirmed duplicate entities using union-find + the merge primitive.

    Only judgments with ``is_same_entity`` and confidence at or above
    ``cfg.canonicalization_confidence_threshold`` are accepted. Accepted pairs are
    unioned into components (handles transitive chains A~B, B~C); each component
    uses its highest-confidence judgment (tie-break: smallest normalized
    canonical_name) to pick the reconciled description and prefer a winner, selects
    a deterministic winner, and collapses the whole component onto it (see
    ``_merge_component``).
    """
    parent = {str(e.id): str(e.id) for e in entities}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    accepted = []
    for judgment, a, b in judgments:
        if not judgment.is_same_entity:
            continue
        if judgment.confidence < cfg.canonicalization_confidence_threshold:
            continue
        id_a, id_b = str(a.id), str(b.id)
        if id_a not in parent or id_b not in parent:
            continue
        union(id_a, id_b)
        accepted.append((judgment, a, b))

    if not accepted:
        return

    # Per-component canonical decision: highest confidence wins; tie-break on the
    # smallest normalized canonical_name for determinism.
    def _is_better(candidate, current) -> bool:
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return Entity._normalize_identity_value(
            candidate.canonical_name
        ) < Entity._normalize_identity_value(current.canonical_name)

    best = {}
    for judgment, a, _b in accepted:
        root = find(str(a.id))
        current = best.get(root)
        if current is None or _is_better(judgment, current):
            best[root] = judgment

    members_by_root = {}
    for entity in entities:
        members_by_root.setdefault(find(str(entity.id)), []).append(entity)

    for root, judgment in best.items():
        members = members_by_root.get(root, [])
        if len(members) < 2:
            continue
        winner = _select_winner(members, judgment.canonical_name)
        losers = [member for member in members if member is not winner]
        _merge_component(winner, losers, judgment, ctx)


@task_summary("Canonicalized entities in {n} summary(ies)")
async def canonicalize_entities(
    summaries: List, ctx: Optional["PipelineContext"] = None, **kwargs
) -> List:
    """Reconcile duplicate entities across a batch of TextSummary objects.

    Gathers candidate entities, blocks them into near-duplicate pairs, judges each
    pair with an LLM, and collapses confident duplicates onto a canonical winner
    in place. Returns the same ``summaries`` list; entities are mutated in place so
    ``add_data_points`` stores the reconciled graph. Spliced into the cognify
    pipeline by ``get_default_tasks`` when the ``entity_canonicalization`` config
    flag is on (default off).
    """
    entities = _gather_entities(summaries)
    if len(entities) < 2:
        return summaries

    cfg = get_cognify_config()
    # Canonicalization is best-effort enrichment sitting between extraction and
    # storage: if the embedder or judge fails (rate limit, timeout, malformed
    # structured output), skip the merge rather than let the exception abort the
    # pipeline run and discard the already-extracted graph for this batch.
    try:
        pairs = await _block_candidate_pairs(entities, cfg)
        judgments = await _judge_pairs(pairs, cfg) if pairs else []
    except Exception as error:
        logger.warning(
            "entity canonicalization skipped (blocking/judge failed)",
            extra={"error": str(error)},
        )
        return summaries

    if judgments:
        _apply_merges(entities, judgments, cfg, ctx)

    return summaries
