# Phase 1: Storage Capabilities (COG-5522 Part 1)

Implements the adapter-level primitives the Part 0 contract declared, on the
default stack only: **Ladybug/Kuzu graph** + **LanceDB vector**. No delete/rollback
workflow, no routing, no stamping, no migration — those are Parts 2 and 3. This
layer just makes the default backends *satisfy the contract* so Part 2's
graph-native delete/rollback can run against real storage.

## What shipped

### Ladybug graph (`infrastructure/databases/graph/ladybug/adapter.py`)

**Declared provenance columns.** Four `STRING[]` columns on both `Node` and
`EDGE` — `source_ref_keys`, `source_dataset_ids`, `source_run_ids`,
`source_run_refs` — added at both schema-bootstrap sites (`_ensure_schema` and
`_initialize_connection`). Provenance is queryable by a column scan
(`list_contains`), never parsed out of the opaque `properties` JSON blob. A
module constant `PROVENANCE_LIST_COLUMNS` documents the set; node write paths
pop these keys so they can never leak into the blob.

**Graph-native marker store.** A dedicated `GraphMetadata(key, value)` node
table holds the marker (`provenance_version`, `delete_mode`). A separate table —
rather than a sentinel `Node` row — keeps marker rows out of every `:Node` data
query. `is_empty()` is scoped to `MATCH (n:Node)` so a *marked but data-empty*
graph still reads as empty (which `ensure_graph_native_for_new_graph` relies on).

**Implemented contract methods** (previously raised `UnsupportedProvenanceCapability`):
- `attach_node_source_refs` / `attach_edge_source_refs`
- `remove_node_source_refs` / `remove_edge_source_refs`
- `delete_edge_triples`
- `get_node_delete_data` / `get_edge_delete_data`
- `find_nodes_by_source_ref` / `find_edges_by_source_ref`
- `find_node_source_refs_by_dataset` / `find_edge_source_refs_by_dataset`
- `find_node_source_refs_by_pipeline_run` / `find_edge_source_refs_by_pipeline_run`
- `set_graph_metadata` / `get_graph_metadata`
- `remove_belongs_to_set_tags` (was an inherited no-op; now detags the blob's
  `belongs_to_set`, scoped or unscoped)

### LanceDB vector (`infrastructure/databases/vector/lancedb/LanceDBAdapter.py`)

`remove_belongs_to_set_tags` was already complete (scoped / unscoped / orphan /
empty / missing-collection branches; covered by `test_belongs_to_set_merge.py`).
Part 1's only change: `delete_data_points` now guards empty input and
single-quote-escapes the id predicate, so it accepts both UUIDs and
graph-computed deterministic string ids idempotently (missing collection /
missing id are no-ops). The stored `id` column is already a `str`, so no type
change was needed.

## Design decisions

- **Columns, not blob.** Kuzu supports native `STRING[]`; the adapter just never
  used it. Provenance must be column-queryable per the Part 0 storage contract.
- **NULL → []** on every read. Unset (and `ALTER`-added) array columns read back
  as `NULL` on Kuzu; `_as_str_list` normalizes so "missing field" is always `[]`.
  `list_contains(NULL, x)` is verified safe (no match, no error).
- **`source_dataset_ids` / `source_run_ids` are materialized**, re-derived from
  `source_ref_keys` / `source_run_refs` on every attach/remove, so dataset- and
  run-scoped lookups filter by column without parsing ref strings. Removing the
  last ref for a dataset/run drops that id automatically.
- **Run refs are recorded per `(run, key)` pair**, independent of whether the key
  was new — so a later run re-touching an existing key stays rollbackable. A
  write with no `pipeline_run_id` records no run ref (non-rollbackable by run id).
- **`edge_text`** stays in the edge `properties` blob (key `"edge_text"`, written
  by `ensure_default_edge_properties`); `get_edge_delete_data` falls back to
  `relationship_name` via `get_edge_retrieval_text` (lazy-imported to avoid the
  infra→modules import cycle).
- **`add_nodes`/`add_edges` never touch provenance columns on `ON MATCH`**, so
  re-cognify preserves refs while still updating content.

## Acceptance criteria → evidence

All Part 1 acceptance criteria are covered by:
- `tests/unit/infrastructure/databases/graph/test_ladybug_provenance_capabilities.py`
  — attach/remove invariants, six lookups, snapshots (indexed_fields,
  missing-artifact omission, edge-text fallback), `delete_edge_triples` endpoint
  preservation, metadata round-trip, scoped/unscoped detag, re-write preservation.
- `tests/unit/infrastructure/databases/vector/test_lancedb_provenance_capabilities.py`
  — `delete_data_points` for UUID + string ids, idempotency, missing collection,
  empty input, single-quote escaping.
- `tests/unit/infrastructure/databases/provenance/test_provenance_contract.py`
  (Part 0) and `test_unified_graph_native_delete.py` (Part 2 fakes) stay green.

## Out of scope (unchanged from the plan)

- Delete/rollback workflow, routing, SDK/API, graph write-site stamping — Part 2.
- Old-graph migration and ledger retirement — Part 3.
- Secondary/shadow/FTS indexes — full column scans only for this first cut.
- Follow-up backends (Neo4j, Postgres graph, PGVector, Neptune) keep raising
  `UnsupportedProvenanceCapability` until they pass the same contract tests.
