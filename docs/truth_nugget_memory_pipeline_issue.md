# Build truth-nugget memory improvement and context indexes

## Context

`remember()` already runs `add()`, `cognify()`, and optionally `improve()` in order. The current
cognify pipeline extracts entity relationships and creates a `TextSummary` for every chunk. The
current global context index uses those summaries as its leaves. Entity consolidation exists as a
separate memify pipeline, but it is not part of the default improve flow.

Introduce an alternative remember workflow built around first-order truth nuggets. Cognify should
extract entities and raw truth nuggets without a separate summary call. Improve should then run an
entity-learning branch and a source-context branch independently.

Recommended Linear project: **Q3.R&D - Graph as world model**. Its stated outcome includes phased
plans for new memory-as-world-model directions, which matches this parent issue better than the
broader self-improvement operations project.

## Scope

Included:

- Model an extracted entity with `name` and `type`.
- Model a `TruthNugget` with `text`, referenced entities, and an optional temporal scope containing
  `start`, `end`, and `description`.
- Treat entity descriptions as truth nuggets instead of privileged entity properties.
- Persist the first-order structure as `Document -> DocumentChunk -> TruthNugget -> Entity`.
- Require every raw truth nugget to belong to exactly one chunk and preserve its document and chunk
  provenance.
- Index raw truth-nugget text for factual retrieval.
- Deliver the work through four phased mini-PRs:
  1. **Truth-nugget cognify:** extract and persist entities and raw nuggets in one structured LLM
     call per chunk. Do not create relationship facts or call `summarize_text` in this mode.
  2. **Entity improvement branch:** consolidate duplicate entities first, reconnect their raw
     nuggets, and then create derived learning nuggets for canonical entities and entity types.
  3. **Document context indexes:** build a `DocumentContextIndex` from a document's raw nuggets.
     Chunks provide ordering and provenance rather than duplicate semantic leaves. Each index may
     expose one or more top-level context nodes based on topic, temporal boundaries, document size,
     and context limits.
  4. **Global context and orchestration:** use exposed document-context nodes as the only leaves of
     the global context index. Update `improve()` to run the entity branch and context branch in
     parallel. Within the context branch, finish affected document indexes before updating the
     global index.
- Keep the entity branch sequential: entity consolidation must finish before entity and entity-type
  learning generation.
- Keep the context branch independent from entity consolidation by reading only immutable raw
  nugget text, temporal scope, document ID, chunk ID, and chunk order.
- Mark extracted nuggets, entity learnings, entity-type learnings, document context nodes, and
  global context nodes as distinct types.
- Restrict entity learning to raw truth nuggets. Restrict document context construction to raw truth
  nuggets. Generated learnings and context nodes must never become inputs to either branch.
- Preserve temporal boundaries and contradictions during context aggregation. Each generated
  context node must retain descendant nugget IDs, covered chunk IDs/indexes, document ID, and
  temporal coverage.
- Rebuild only affected document indexes and invalidate the global ancestors of changed exposed
  roots. Repeated improve runs must be incremental and idempotent.
- Preserve the public lifecycle `remember -> add -> cognify -> improve` for the alternative mode.

Excluded:

- Replacing the existing cognify behavior as the default in the first rollout.
- Adding entity roles, relationship types, extraction confidence, or mandatory temporal values.
- Feeding entity or entity-type learnings into document or global context indexes.
- Feeding document or global context nodes into entity learning.
- Changing session-only `remember(session_id=...)` behavior.
- Migrating existing relationship graphs or `TextSummary` indexes to the new representation.
- Applying the model to code graphs, deterministic DLT ingestion, or custom graph models in the
  first rollout.

## Acceptance Criteria

- [ ] The alternative cognify mode stores entities and raw truth nuggets with chunk/document
      provenance and creates no `TextSummary` nodes.
- [ ] A temporal truth nugget can store optional `start`, `end`, and `description` values without
      inventing missing precision.
- [ ] Improve consolidates entities before producing entity and entity-type learning nuggets, and
      repeated runs do not multiply equivalent derived learnings.
- [ ] A `DocumentContextIndex` consumes raw nuggets only, preserves temporal and source coverage,
      and can expose one or multiple top-level nodes.
- [ ] The global context index consumes exposed document-context nodes only and refreshes affected
      ancestors when a document index changes.
- [ ] Improve runs the entity and context branches concurrently while preserving the required
      ordering inside each branch.
- [ ] End-to-end tests show that `remember()` produces raw nuggets, canonical entities, derived
      entity/type learnings, document context roots, and global context without recursive use of
      generated content.

## Metrics / Instrumentation

* Expected impact: remove the per-chunk summary call from cognify and separate factual evidence,
  entity learning, and source context into observable stages.
* How tracked: retain stage-level LLM call/token telemetry for cognify and improve. Update the
  cognify estimator for the removed summary call; improve cost estimation remains separate unless
  explicitly added in a phase issue.

## Dependencies

* Related entity-dedup work: [SDK-201](https://linear.app/cognee/issue/SDK-201/embedding-similarity-fuzzy-entity-dedup-rework-of-pr-3929-gh-3628).
  Align canonicalization and audit behavior before selecting destructive merge semantics.
* Coordinate incremental invalidation with [COG-6060](https://linear.app/cognee/issue/COG-6060/incremental-loading-cognify).
* Public documentation should align with [SDK-373](https://linear.app/cognee/issue/SDK-373/make-rememberrecall-the-documented-and-agent-discoverable-memory-path).
* The existing global context documentation in [COG-5113](https://linear.app/cognee/issue/COG-5113/documentation-document-global-context-index)
  will need revision after the leaf contract changes.

## Rollout / Release

* docs updated: document the alternative remember mode, truth-nugget model, temporal behavior, and
  strict separation between raw evidence, learnings, and context indexes.
* release notes: required when the alternative mode becomes part of the public SDK/API surface.
