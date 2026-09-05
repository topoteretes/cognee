# COG-6129: Entity-type and relationship-pattern similarity for graph bucketing

Branch: `feature/cog-6129-graph-similarity-metric` · Status: implemented, not yet merged

## Context

The Global Context Index groups a dataset's `TextSummary` nodes into buckets, which are
then used to build a hierarchical index for retrieval. With
`bucketing_strategy="graph"`, level 0 decides whether two summaries belong in the same
bucket by comparing **only the entities they share** (`weighted_jaccard`, IDF-weighted —
rare entities count more, ubiquitous ones count zero).

This signal has a conceptual limitation: two summaries can talk about completely
different entities and still be "close" in the same sense (e.g. two sentences that both
describe an employment relationship, even if the people/companies named don't overlap),
or share an entity purely by coincidence without the two summaries being meaningfully
related at all. The one signal available today can't tell these cases apart.

## Proposal

Add two **optional** similarity signals that stack on top of the existing one instead
of replacing it:

### 1. Entity-type similarity

Beyond "do they share the same entity?", also ask "do they share the same entity
*type*?" (e.g. both mention a company, even if different companies). Same exact formula
as the existing signal (`weighted_jaccard`, with IDF computed over the population of
*types* instead of entities), applied to a different set:

```python
type_ids_for_entities(entity_ids, entity_type_by_entity_id) -> set[str]
type_similarity(left_type_ids, right_type_ids, type_idf_weights)  # = weighted_jaccard
```

### 2. Relationship-pattern similarity

Beyond "do they share entities/types?", also ask "do they share the same kind of
*relationship* between similar entities?" — e.g. "X works for Y" is a recognizable
pattern even as X and Y change from one summary to another. Concretely, pairs of
`(source_entity, target_entity, relationship_name)` triples are compared:

```python
pattern_similarity(left_edge, right_edge, ...) -> float
```

The relationship name is a **hard gate applied first**: if two relationships don't
match (either exactly, or by embedding distance under a threshold), the pair scores 0
without even comparing source/target. If the relationship does match, the final score
is the average of the (entity+type) similarity of source-to-source and
target-to-target — so the comparison is **positional**, not symmetric: "X works for Y"
and "Y works for X" only get a high score if `type_weight > 0` and the two roles still
share a compatible entity type.

### 3. Weighted combination, behind a single switch

The three signals (entity, type, pattern) combine into a single weighted average:

```python
combined_similarity(entity_score, type_score, pattern_score,
                     entity_weight=1.0, type_weight=0.0, pattern_weight=0.0)
```

Rather than exposing these three floats directly to callers, they sit behind one
public switch:

```python
GraphSimilarityMode = Literal["entity", "combined"]

resolve_graph_similarity_weights("entity")   # -> (1.0, 0.0, 0.0) -- today's behavior
resolve_graph_similarity_weights("combined") # -> (1/3, 1/3, 1/3) -- equal blend
```

`graph_similarity_mode="entity"` (default) reproduces today's behavior **exactly** — no
observable difference until a caller opts into `"combined"`. The 1/3-each split for
`"combined"` is fixed, not independently tunable per weight; picking a different split
is future, out-of-scope work pending empirical comparison (see "Open question" below).

### 4. Scoring as a uniform, swappable abstraction

The four signals (entity, type, pattern, combined) are implemented in a new module,
`bucketing/graph/scorers.py`, as functions sharing one signature —
`(summary_profile, bucket_profile) -> float` — built via factories that close over the
shared dataset-wide inputs (IDF weights, entity types, the relation index, the three
resolved weights):

```python
scorers = build_scorers(idf_weights, entity_type_by_entity_id, type_idf_weights,
                         edge_type_embeddings, pattern_distance_threshold,
                         entity_weight, type_weight, pattern_weight)
# scorers == {"entity": ..., "type": ..., "pattern": ..., "combined": ...}
score = scorers["combined"](summary_profile, bucket_profile)
```

`graph/placement.py`'s candidate-selection code (`_choose_best_candidate`,
`_choose_existing_graph_bucket`) builds an `EntityGroupProfile` once per summary/bucket
(caching its entity ids, type ids, and relation triples) and calls `scorers["combined"]`
instead of computing `entity_score`/`type_score`/`pattern_score` inline — replacing what
was previously scattered, repeated logic with one small, directly testable module.

## Where it fits

All three signals apply **only at level 0** of the `"graph"` strategy (exactly where
the existing single signal already applied) — no change to `bucketing_strategy="vector"`,
nor to levels ≥ 1 (which today already always fall back to vector-based placement,
independent of this proposal).

`graph_similarity_mode` is threaded end-to-end through `update_global_context_index()` →
`build_and_persist_context_index()` → `build_context_index()`. Getting this right took a
second pass: the first version only added `entity_weight`/`type_weight`/`pattern_weight`
to `build_context_index()` itself, but `update_global_context_index()` and
`build_and_persist_context_index()` never forwarded them — meaning the new signal was
unreachable from any real call path (not `update_global_context_index()`, not
`global_context_index_pipeline()`, not `cognee.improve()`). This is now fixed; a
dedicated regression test confirms the value actually reaches `build_context_index()`.
The most-used public entry point (`cognee.improve()`, which hardcodes
`bucketing_strategy="graph"`) and `global_context_index_pipeline()` still don't expose
`graph_similarity_mode` — out of scope for this change, same as `bucketing_strategy`'s
own precedent.

## A new, non-free cost: embedding relationship names

To compare two relationship names as "close enough even if not identical" (e.g. "works
for" vs "is employed by"), every distinct relationship name in the dataset needs an
embedding. This embedding is **not reused** from anywhere in the existing system:

- Not every vector adapter returns the stored vector back when you retrieve it
  (PGVector, the most widely used one, drops the vector in `retrieve()`; only LanceDB
  does not). This isn't a database limitation — the SQL query already selects the
  `vector` column — it's just data the current code doesn't return.
- Even if it always did, the only vector-search primitive available is `search()`
  (approximate nearest-neighbor, top-K), which doesn't give a pointwise distance
  between two arbitrary names — exactly what pairwise comparison needs.

The solution adopted: **a fresh embedding, computed once per build**, of every distinct
relationship name (`_embed_relationship_names` in `bucketing/graph/inputs.py`) — not per
pairwise comparison. With the number of distinct relationships typically small relative
to the number of summaries, the added cost is marginal, but it is still an **extra call
to the embedding engine that doesn't exist today**, and it fires every time
`global_context_index=True`, regardless of whether `pattern_weight` actually ends up
`> 0` (the computation happens unconditionally in `load_graph_bucketing_inputs`,
independent of the weights it will later be used with).

## What does NOT change

- No new `bucketing_strategy` — it stays `"graph"` / `"vector"`; the new signals are an
  internal refinement of the existing graph strategy.
- No change to incremental-update logic — the same extension points
  (`place_graph_summaries_incrementally`, `_choose_existing_graph_bucket`) build the
  same `EntityGroupProfile`s and call the same `scorers["combined"]`, resolved from the
  same `graph_similarity_mode` passed to the initial build.
- No default changed: without explicit opt-in (`graph_similarity_mode="entity"`, the
  default), the system behaves exactly as it did before this proposal.

## Verification

`cognee/tests/unit/tasks/memify/` + `cognee/tests/unit/modules/graph/` — 253 passed,
zero pre-existing tests modified beyond the new additions below:

- `scoring.py`: formula correctness, pattern-matching symmetry/asymmetry, default
  behavior equivalent to entity-only (`test_global_context_index_idf.py`).
- `scorers.py`: profile building, each of the four scorers individually, `combined`'s
  blending and its pattern-computation short-circuit when `pattern_weight == 0`
  (`test_global_context_index_graph_scorers.py`, new).
- `resolve_graph_similarity_weights`, and a dedicated reachability test proving
  `graph_similarity_mode` actually threads from `update_global_context_index()` down to
  `build_context_index()` — the exact gap described above, now covered so it can't
  silently regress (`test_global_context_index.py`).

## Open question

With the default mode, this proposal changes nothing observable. The natural next
step — explicitly **out of scope for this issue** — is deciding if and when to make
`"combined"` the default (or expose a non-equal weight split), which requires an
empirical comparison first (not just a theoretical one), along the same lines as the
manual analysis already done during this design phase.
