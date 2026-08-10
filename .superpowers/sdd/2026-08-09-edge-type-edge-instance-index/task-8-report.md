# Task 8 Report: Automatic graph-derived migration

## Delivered

- Added the append-only `separate_edge_instance_index` migration after
  `postgres_graph_provenance_columns` for Cognee `1.4.1.dev1`.
- The upgrade reads graph edges in 1,000-edge batches, repairs missing
  `edge_object_id` properties with the shared structural-ID helper, rebuilds
  exact type counts, and replaces both edge vector collections without touching
  `Triplet_text`. Unsupported paginated adapters fall back to one full graph
  read that is replayed in bounded slices.
- The downgrade removes `EdgeInstance_text` and restores the legacy prose-keyed
  EdgeType collection from current graph edge text. Upgrade, retry, and
  downgrade all converge through collection replacement.
- Regression coverage pins the structural UUID, stale cleanup, exact IDs and
  counts, failure/retry behavior, runner stamping order, pagination fallback,
  downgrade behavior, and compatibility checks for current edge indexes.

## Validation

- Focused migration/runner pytest: `35 passed`.
- `ruff check` passed for all six changed source and test files.
- `ruff format --check` passed for all six changed source and test files.
- `python -m py_compile` passed for all six changed source and test files.
- `git diff --check` passed.
- Provider-backed migration and backwards-compatibility scripts were not run;
  they require their seeded external-store workflow.
