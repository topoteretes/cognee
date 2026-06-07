# Temporal Freshness Policies for Agent Memory

Related issue: https://github.com/topoteretes/cognee/issues/3004

## Summary

Cognee already has temporal ingestion and `SearchType.TEMPORAL` for event-time
queries. This proposal adds a complementary memory freshness layer: preserve
historical facts, but make it possible to identify which facts are current,
which facts have been superseded, and which facts should be excluded from
"what is true now" retrieval.

The motivating use case is agent memory over changing real-world preferences
and rules. A user, customer, project, workflow, API, or business rule may change
over time. Old memories should not be deleted, because they are useful for
audit and timeline questions, but they also should not be returned to an agent
as current truth after a newer approved memory supersedes them.

## Prior Art: Graphiti

Graphiti's temporal context graph is a useful reference point:
https://github.com/getzep/graphiti

- Raw source records are stored as episodes.
- Derived facts are represented as relationships with validity windows.
- Facts have `valid_at` and `invalid_at` timestamps.
- Facts also carry ingestion/lifecycle metadata such as `created_at` and
  `expired_at`.
- When new information contradicts older information, the older fact is
  invalidated rather than deleted.
- Search can filter by date fields, allowing "current" retrieval and historical
  retrieval to use the same graph.

The main design lesson is that recency scoring alone is not enough. A robust
memory system needs explicit validity state:

- event/reference time: when the source episode happened
- valid time: when the fact became true
- invalid time: when the fact stopped being true
- transaction/lifecycle time: when the memory system stored or invalidated the
  fact
- provenance: which source episode(s) support the fact

Cognee has the temporal foundation already. The proposed feature adds a
lightweight validity and supersession model that fits Cognee's existing
remember/recall API, datasets, NodeSets, and retrievers.

## Goals

- Preserve temporal history instead of deleting old facts.
- Let callers ask for current memory without receiving superseded facts.
- Keep the feature opt-in and backward-compatible.
- Work first as metadata and retrieval policy before requiring a new graph
  engine abstraction.
- Support both deterministic policies and LLM-assisted contradiction detection.
- Expose freshness metadata in search results for explainability.

## Non-Goals

- Replacing `SearchType.TEMPORAL`.
- Forcing all remembered data into a new schema.
- Solving application-specific truth adjudication without caller-provided
  keys or policy.
- Requiring Neo4j or Graphiti mode for the basic feature.
- Deleting superseded facts by default.

## Proposed Concepts

### Memory Validity Metadata

Add optional metadata accepted by `remember()` and persisted with data/chunks,
graph nodes, graph edges, or DataPoints where appropriate:

```python
await cognee.remember(
    "Customer 55442 prefers delivery address B for standard orders.",
    dataset_name="customer_memory",
    temporal_cognify=True,
    memory_key={
        "entity": "customer:55442",
        "predicate": "prefers_delivery_address",
        "object_type": "delivery_address",
    },
    valid_at="2026-06-07T00:00:00Z",
    freshness_policy="newer_conflicting_fact",
    provenance={
        "source": "order_review",
        "source_id": "review-123",
    },
)
```

Suggested fields:

- `memory_key`: stable identity for facts that can supersede one another.
- `valid_at`: when the fact became true.
- `invalid_at`: when the fact stopped being true, if known.
- `expired_at`: when Cognee invalidated the fact.
- `freshness_state`: `current`, `superseded`, `stale`, `historical`, or
  `unknown`.
- `superseded_by`: id of the fact that superseded this one.
- `provenance`: source metadata or source episode/data ids.

Naming can follow Cognee conventions; the field names above intentionally align
with Graphiti's practical model.

### Freshness Policies

Add policy values that can be passed through `remember()` or
`retriever_specific_config`:

```python
FreshnessPolicy = Literal[
    "none",
    "latest_per_key",
    "newer_conflicting_fact",
    "llm_contradiction",
]
```

Policy semantics:

- `none`: preserve current behavior.
- `latest_per_key`: for the same `memory_key`, mark older current facts as
  superseded when a newer fact is remembered.
- `newer_conflicting_fact`: only supersede older facts when the fact key matches
  and the new fact is not a duplicate.
- `llm_contradiction`: evaluate candidate facts with a small structured-output
  prompt, similar to Graphiti's duplicate/contradiction split.

### Retrieval Policies

Add retrieval-time policy separate from write-time freshness:

```python
await cognee.recall(
    query_type=SearchType.TEMPORAL,
    query_text="Current delivery preference for customer 55442",
    datasets=["customer_memory"],
    retriever_specific_config={
        "freshness": {
            "mode": "current",
            "as_of": "2026-06-07T00:00:00Z",
        }
    },
)
```

Modes:

- `all`: current behavior; return matching facts regardless of validity.
- `current`: include facts where `valid_at <= as_of` and
  `invalid_at is null or invalid_at > as_of`, excluding superseded facts.
- `prefer_current`: include historical facts if useful, but rank current facts
  higher and include freshness metadata.
- `historical`: return facts valid during a requested interval.

Default `as_of` should be current UTC time.

## Candidate Architecture

### Phase 1: Metadata and Filter Helpers

1. Add a small Pydantic model for memory validity metadata.
2. Accept optional validity fields in `remember()` kwargs.
3. Store these fields as structured metadata on the underlying data item and, if
   available, graph/vector objects derived from it.
4. Add helper functions that build current/historical date filters from
   `valid_at`, `invalid_at`, and `freshness_state`.
5. Add unit tests around metadata validation and filter construction.

This phase should not change default retrieval behavior.

### Phase 2: Retrieval-Time Filtering

1. Thread `freshness` config through `retriever_specific_config`.
2. Apply freshness filtering in `TemporalRetriever` first.
3. Consider supporting the same filter in chunk and graph-completion retrievers
   when result metadata has the required fields.
4. Return freshness metadata with search results.

This gives applications current-only retrieval without automatic write-time
invalidation yet.

### Phase 3: Supersession on Write

1. When a new fact is remembered with a `memory_key`, search for candidate
   current facts with the same key.
2. Detect exact duplicates deterministically.
3. Detect contradictions either by policy-specific deterministic comparison or
   by a structured LLM prompt.
4. For superseded facts, set `invalid_at` to the new fact's `valid_at`, set
   `expired_at` to current UTC time, set `freshness_state` to `superseded`, and
   link `superseded_by`.
5. Preserve provenance for both old and new facts.

This mirrors Graphiti's strongest idea: invalidate contradicted facts, do not
delete them.

### Phase 4: Public API Refinement

Once the lower-level path is proven, consider lifting the most stable pieces to
top-level parameters:

```python
await cognee.remember(..., memory_key=..., valid_at=..., freshness_policy=...)
await cognee.recall(..., freshness_policy="current", as_of=...)
```

Until then, `retriever_specific_config` is a good low-risk integration point.

## Result Shape

Search results should expose freshness fields when available:

```json
{
  "search_result": "Customer 55442 prefers delivery address B.",
  "metadata": {
    "freshness_state": "current",
    "valid_at": "2026-06-07T00:00:00Z",
    "invalid_at": null,
    "supersedes": ["fact-a"],
    "provenance": {"source": "order_review", "source_id": "review-123"}
  }
}
```

This allows the calling agent or application to explain why a fact is trusted.

## Test Plan

- Remembering facts without freshness metadata preserves current behavior.
- `current` retrieval returns an active fact and excludes a superseded fact.
- `historical` retrieval can still return the superseded fact for the interval
  where it was valid.
- `latest_per_key` marks older same-key facts as superseded.
- `newer_conflicting_fact` keeps non-conflicting same-entity facts active.
- LLM contradiction detection returns separate duplicate and contradicted ids.
- Superseded facts keep provenance and remain auditable.
- Date boundaries use timezone-aware UTC datetimes.

## Open Questions

- Which internal object should be the canonical owner of validity metadata:
  DataPoint, graph edge, vector result metadata, or a new memory-fact object?
- Should `memory_key` be a dict, a string, or both?
- Should freshness filtering be available across all retrievers or only temporal
  retrieval initially?
- Should write-time supersession be synchronous, backgrounded, or part of
  `improve()`?
- How much contradiction detection should be deterministic before calling an
  LLM?
- Should Cognee expose a first-class "episode" abstraction in the core API, or
  is existing data provenance enough?

## Compatibility

This proposal is additive. Existing calls to `remember()`, `recall()`, and
`search()` should behave unchanged unless the caller provides freshness metadata
or a freshness retrieval policy.
