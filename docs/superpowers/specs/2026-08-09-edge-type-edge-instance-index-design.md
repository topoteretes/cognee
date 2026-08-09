# Edge Type and Edge Instance Index Separation

Status: Approved design

Issue: [topoteretes/cognee#4361](https://github.com/topoteretes/cognee/issues/4361)

## Summary

Cognee currently uses per-edge `edge_text` as the identity and embedded content of
`EdgeType_relationship_name`. Because `edge_text` is instance-specific prose, the collection grows
with the number of edges instead of the number of relationship types. This makes type counts
meaningless and causes excessive vector storage and search cost.

The fix separates relationship-type identity from edge-instance semantic content:

- `EdgeType_relationship_name` contains one point per canonical `relationship_name`.
- `EdgeInstance_text` contains one point per directed graph edge and embeds that edge's prose.
- `Triplet_text` remains unchanged and optional.

The central invariant is:

> Edge identity is structural; edge text is mutable content.

## Goals

1. Make `EdgeType_relationship_name` strictly type-level with O(distinct relationship types)
   cardinality.
2. Preserve semantic search over full per-edge prose in `EdgeInstance_text` with O(edges)
   cardinality.
3. Give every indexing, retrieval, deletion, and migration path the same identity semantics.
4. Rebuild existing derived vector data from the graph, removing legacy and orphaned points.
5. Preserve existing `Triplet_text` behavior and configuration without using it as a compatibility
   mechanism.

## Non-goals

- Adding parallel graph edges with the same `(source, relationship_name, target)` tuple.
- Changing `Triplet_text`, triplet IDs, or the optional triplet-embedding pipeline.
- Redesigning `generate_edge_object_id` normalization.
- Solving the separate PGVector ANN-index issue.
- Inferring trustworthy type information from legacy `EdgeType_relationship_name` rows.

## Existing Failure

All current edge-indexing paths prefer a nonblank `edge_text` and fall back to
`relationship_name`. They then construct `EdgeType` points and deduplicate with a `Counter`. When
`edge_text` contains unique prose, the counter creates one nominal edge-type point per edge.

The behavior was introduced with synthesized edge retrieval text and remains present in current
`dev` across:

- standard `index_graph_edges` indexing;
- PostgreSQL hybrid writes;
- Neptune Analytics hybrid writes;
- retrieval IDs and fact selection;
- deletion and orphan cleanup.

## Approved Approach

Perform an end-to-end cutover with an automatic data migration. Dual-writing legacy rows and
new-writes-only repair were rejected because they either preserve the broken collection semantics
or leave existing deployments corrupted until manual intervention.

## Data Model

### `EdgeType_relationship_name`

- Model: `EdgeType`
- Point ID: `EdgeType.id_for(relationship_name)`
- Indexed content: canonical, nonblank `relationship_name`
- Deduplication key: `relationship_name`
- Cardinality: O(distinct relationship types)
- Purpose: relationship-type matching and graph-wide edge counts
- Payload count invariant:

  ```text
  EdgeType.number_of_edges ==
      count(current directed graph edges where edge.relationship_name == relationship_name)
  ```

No `edge_text` value may participate in this collection's ID, deduplication, or embedded content.

### `EdgeInstance_text`

- Model: a new `EdgeInstance` datapoint with indexed field `text`
- Point ID: the exact existing `edge_object_id`
- Indexed content: nonblank `edge_text`, otherwise a readable `relationship_name` fallback
- Cardinality: O(current directed graph edges)
- Purpose: semantic retrieval of edge-specific facts
- Useful payload fields: `relationship_name`, `source_node_id`, and `target_node_id`

The indexed text is mutable content and never participates in the point ID. Updating only
`edge_text` replaces the existing point's payload and vector. Every adapter's upsert contract must
replace both values on an ID conflict; updating only the payload is insufficient.

### `Triplet_text`

`Triplet_text` remains completely unchanged and optional. `EdgeInstance_text` is neither an alias
nor a replacement for it.

## Edge Instance Identity

`EdgeInstance_text` must reuse the exact output of `generate_edge_object_id`; it must not reproduce
that helper's normalization logic independently.

The existing identity contract is:

- identity tuple: `(source, relationship_name, target)`;
- directional: `(A, depends_on, B)` differs from `(B, depends_on, A)`;
- canonicalization is exactly whatever `generate_edge_object_id` applies, currently stringifying
  values, lowercasing, replacing spaces with underscores, removing apostrophes, and generating a
  UUID5;
- `edge_text` is excluded from identity;
- changing source, target, or relationship name produces a new ID;
- duplicate directed tuples use upsert/last-write semantics.

All supported graph adapters already model a directed tuple as one edge: PostgreSQL and Turso use
the tuple as a composite primary key, while Ladybug, Neo4j, and Neptune use merge/upsert behavior.
Parallel edges with an identical directed tuple are outside the current Cognee graph model and are
not introduced by this change.

If an incoming edge lacks `edge_object_id`, preprocessing calls `generate_edge_object_id` once and
attaches the result to the shared edge representation before any standard, PostgreSQL, or Neptune
indexing path consumes it.

## Shared Point Construction

A shared edge-index builder accepts prepared graph edges and returns:

1. `EdgeType` points grouped strictly by `relationship_name`;
2. `EdgeInstance` points keyed by `edge_object_id`.

The builder owns classification and fallback rules so adapters do not independently decide whether
a value is a relationship type or edge prose. Adapter-specific code remains responsible for its
storage transaction and batching mechanics. Adapter payload schemas must preserve
`EdgeType.number_of_edges` and the EdgeInstance join fields instead of reducing every point to only
`id` and `text`.

## Write Semantics

### Standard graph/vector path

`index_graph_edges` indexes both shared point sets through the standard vector interface. Rewriting
an existing directed edge upserts its `EdgeInstance_text` point. Type points use graph-wide counts,
not counts from the incoming batch.

### PostgreSQL hybrid path

The hybrid adapter continues to write graph and vector rows transactionally. It embeds unique
relationship names for `EdgeType_relationship_name` and edge-instance text for
`EdgeInstance_text`, then upserts both collections with deterministic IDs. Counts for affected
relationship names are queried from graph state after the graph mutation inside the transaction.

### Neptune Analytics hybrid path

Neptune uses the same shared classification and IDs. Its existing graph-first, retryable write
sequence creates or updates both vector collections after the graph upsert. A retry converges via
deterministic IDs and upserts.

### Count updates

After a graph mutation, affected relationship names are recalculated from current graph state:

- count greater than zero: upsert/update the corresponding `EdgeType` point and payload;
- count equal to zero: delete the corresponding `EdgeType` point.

The incoming batch size is never used as the persisted graph-wide count.

## Retrieval Semantics

Retrieval searches the two collections for different purposes:

- `EdgeType_relationship_name` supplies relationship-type matches.
- `EdgeInstance_text` supplies full-prose matches and semantic facts.

Graph edge results expose both joins:

- `edge_type_id = EdgeType.id_for(relationship_name)`;
- `edge_instance_id = edge_object_id`.

Edge display order remains deterministic:

1. pinned structural edges;
2. instance-ranked edges;
3. type-ranked edges;
4. unranked graph order.

Facts are selected only from `EdgeInstance_text`. `EdgeType_relationship_name` rows are never
treated as prose facts. A missing collection returns an empty lane during migration, on an empty
store, or when an older database has not completed migration.

## Update and Deletion Semantics

- Updating only `edge_text` upserts the same instance ID with replacement content and vector.
- Changing a structural tuple field creates a new edge identity; normal graph mutation semantics
  are responsible for deleting the old edge when appropriate.
- Hard-deleting a graph edge deletes its `EdgeInstance_text` point by `edge_object_id`.
- EdgeInstance deletion is driven by graph-edge deletion, not by individual provenance removal.
- Removing one provenance reference from a shared edge leaves the graph edge and its instance point
  intact.
- After a hard deletion, affected relationship counts are recalculated from current graph state.
- A zero count removes the stale `EdgeType` point; a positive count updates it.

Delete planning and cleanup use structural IDs for edge instances and relationship-name IDs for
types. They never derive either identity from `edge_text`.

## Automatic Migration

The graph is the sole source of truth. Legacy `EdgeType_relationship_name` rows are not trusted
because they may represent per-edge prose.

The migration performs the following bounded-batch rebuild:

1. Acquire the existing migration lock without stamping the new revision.
2. Read current graph edges in bounded batches where the adapter permits streaming/pagination.
3. For historical edges missing `edge_object_id`, call `generate_edge_object_id`, attach the exact
   result to the edge properties, and persist that repaired graph edge before indexing it. This
   makes the graph and all later retrieval/deletion paths use the same ID.
4. Build the complete desired type counts from actual `relationship_name` values.
5. Build desired `EdgeInstance` points from current graph edges.
6. Replace the contents of `EdgeType_relationship_name`, removing every stale legacy prose row.
7. Replace the contents of `EdgeInstance_text`, removing points for graph edges that no longer
   exist.
8. Stamp the migration revision only after both collection rebuilds complete successfully.

PostgreSQL uses transactional replacement where supported. Other vector stores run collection
replacement under the migration lock. If a process fails after clearing or partially rebuilding a
derived collection, the revision remains unset or stale. The next run repeats the graph-derived
replacement and converges through deterministic IDs and upserts.

The migration is idempotent:

- rerunning produces the same point IDs, content, and counts;
- it creates no duplicates;
- it removes stale points on every complete run;
- it does not read, write, or rename `Triplet_text`.

## Failure Handling

- A missing or blank `relationship_name` is not indexed as an `EdgeType`; the graph write retains
  its existing validation behavior.
- A blank `edge_text` uses a readable relationship-name fallback for `EdgeInstance.text` only.
- Missing `edge_object_id` is repaired once during shared preprocessing and propagated downstream.
- Standard and Neptune writes are retryable after partial vector failure because the graph is the
  source of truth and point IDs are deterministic.
- PostgreSQL hybrid maintains its graph/vector transactional guarantee.
- Migration errors prevent revision stamping and are surfaced with collection and batch context.

## Cross-adapter Consistency

The following paths must share the same point-construction contract:

- standard edge indexing;
- PostgreSQL hybrid writes;
- Neptune Analytics hybrid writes;
- retrieval ranking and fact selection;
- direct deletion and provenance-aware deletion;
- automatic migration.

Tests must prevent an adapter from reintroducing `get_edge_retrieval_text(edge_text,
relationship_name)` as the source of `EdgeType` identity.

## Testing Strategy

Implementation follows test-driven development. Regression tests are written and observed failing
before production changes.

### Shared unit tests

- Two edges with the same relationship and distinct prose create one `EdgeType` with count two and
  two `EdgeInstance` points.
- `EdgeType.id_for` receives only relationship names.
- `EdgeInstance` IDs exactly equal the existing `edge_object_id` output.
- Changing only `edge_text` preserves the instance ID and changes indexed content.
- Reversing endpoints or changing relationship name changes the instance ID.
- Blank edge prose affects only the indexed-text fallback.
- Duplicate directed tuples use last-write content.

### Adapter contract tests

- Standard, PostgreSQL hybrid, and Neptune hybrid produce equivalent IDs and texts.
- PostgreSQL writes both vector collections atomically with graph rows.
- Repeated writes upsert rather than duplicate instance points.
- Type counts reflect current graph state rather than batch size.

### Retrieval tests

- Type hits rank by `edge_type_id` and never become facts.
- Instance hits rank by `edge_instance_id` and supply facts.
- Pinned, instance-ranked, type-ranked, and unranked ordering is stable.
- Missing either collection degrades to an empty retrieval lane.
- `Triplet_text` calls and behavior remain unchanged.

### Deletion and provenance tests

- Hard edge deletion removes its instance point.
- Removing one source reference from a shared edge preserves the instance point.
- Deleting the last edge of a type removes the type point.
- Deleting one of several edges updates the graph-wide count without removing the type.

### Migration tests

- Legacy per-edge prose rows are removed from the type collection.
- Type rows and counts are reconstructed only from graph relationship names.
- Instance rows are reconstructed only from current graph edges.
- Stale type and instance points are removed.
- Failure before completion leaves the revision unstamped.
- Retry after interruption converges without duplicates.
- A second successful run is a no-op in resulting state.
- PostgreSQL hybrid and Neptune hybrid migration behavior is covered explicitly.

### Verification commands

At minimum, run targeted unit and migration suites, Ruff on touched files, and relevant PostgreSQL
integration tests. The final pull request follows `CONTRIBUTING.md`: it targets `dev`, documents the
commands run, includes required test evidence, uses a semantic title, and uses DCO sign-off for
commits.

## Documentation and Compatibility

- Add a changelog entry under `Unreleased / Fixed` if requested by maintainers.
- Document that `EdgeType_relationship_name` is type-level and `EdgeInstance_text` is
  instance-level.
- Existing APIs remain unchanged; only derived collection semantics and internal retrieval joins
  change.
- Operators should expect the migration to rebuild edge embeddings and temporarily consume
  embedding-provider capacity proportional to the number of current graph edges.
