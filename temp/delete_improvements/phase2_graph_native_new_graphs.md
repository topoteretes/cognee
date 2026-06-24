# Part 2 — graph-native delete/rollback for new graphs (Lazar's Part 0 contract)

This is the Part 2 plan as re-ported onto **Lazar's official Part 0 contract**
(`cognee/infrastructure/databases/provenance/` + the provenance methods on
`GraphDBInterface` and `GraphVectorStoreInterface`). It supersedes the original
Part 2 design that was built on a different contract (no opaque `uuid5` refs, no
`ProvenanceDeleteResult`, no `dataset_ids` snapshot field, no marker node, no
`supports_graph_native_provenance` flag).

## Goal

When a graph is **graph-native** (its graph-level metadata advertises the
graph-native delete mode), delete and rollback are performed directly against the
graph's own provenance — `source_ref_keys` carried on nodes/edges — instead of
the relational ledger. Old/unmarked graphs keep the existing relational-ledger
path completely unchanged. The new path is **inert until Part 1**, because
`set_graph_metadata` / `attach_*` raise `UnsupportedProvenanceCapability` on
pre-Part-1 backends, so no graph is ever marked.

## Lazar's contract (the target API we build on)

**Parseable refs** (`provenance/source_refs.py`):
- `make_source_ref_key(dataset_id, data_id) -> "source_ref:v1:{dataset}:{data}"`
- `make_source_run_ref(pipeline_run_id, source_ref_key) -> "source_run_ref:v1:{run}:{source_ref_key}"`
- getters: `get_dataset_id_from_source_ref_key`, `get_data_id_from_source_ref_key`,
  `get_pipeline_run_id_from_source_run_ref`, `get_source_ref_key_from_source_run_ref`

**Constants** (`provenance/constants.py`): `GRAPH_PROVENANCE_VERSION = "1"`,
`GRAPH_DELETE_MODE_GRAPH_NATIVE = "graph_native"`,
`GRAPH_PROVENANCE_VERSION_KEY = "provenance_version"`,
`GRAPH_DELETE_MODE_KEY = "delete_mode"`.

**Dataclasses** (`provenance/delete_data.py`):
- `EdgeIdentity(source_id, target_id, relationship_name)` — field order is
  source, target, relationship.
- `NodeDeleteData(node_id, node_type, indexed_fields, node_properties,
  source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs)`
- `EdgeDeleteData(edge, edge_text, edge_properties, source_ref_keys,
  source_dataset_ids, source_run_ids, source_run_refs)`

**`GraphDBInterface` provenance methods** (all raise
`UnsupportedProvenanceCapability` by default until Part 1):
- write: `attach_node_source_refs`, `attach_edge_source_refs`,
  `remove_node_source_refs`, `remove_edge_source_refs`, `delete_edge_triples`,
  `delete_nodes`
- read: `get_node_delete_data`, `get_edge_delete_data`,
  `find_nodes_by_source_ref`, `find_edges_by_source_ref`,
  `find_node_source_refs_by_dataset`, `find_edge_source_refs_by_dataset`,
  `find_node_source_refs_by_pipeline_run`, `find_edge_source_refs_by_pipeline_run`
- marker: `get_graph_metadata() -> dict[str,str]`, `set_graph_metadata(dict)`

**`GraphVectorStoreInterface`**: `delete_by_source_ref(str)`,
`delete_by_dataset_id(str)`, `rollback_by_pipeline_run_id(str)` — all take **str**
args, return **None**, single-arg rollback.

## Semantics

- **Ownership** of a node/edge = its `source_ref_keys`. Removing a set of refs
  leaves the artifact **unowned** iff none remain → hard delete. Otherwise it
  **survives** → only the targeted refs are detached.
- **Data delete** removes one `source_ref:v1:{dataset}:{data}` key.
- **Dataset delete** removes all of a dataset's refs (from
  `find_*_source_refs_by_dataset`).
- **Rollback** removes the refs a run attached (from
  `find_*_source_refs_by_pipeline_run` — derived from the run refs).
- **Vectors are deleted before any graph mutation.** Remove-refs and deletes are
  all idempotent, so a failure leaves graph provenance intact and a retry
  converges.

## Committed spine (already on this branch)

1. **Marker** — `cognee/infrastructure/databases/provenance/markers.py`
   - `is_graph_native_graph(graph_engine)`: reads `get_graph_metadata`; any
     exception (incl. `UnsupportedProvenanceCapability`) → `False` (fail-safe to
     the ledger).
   - `ensure_graph_native_for_new_graph(graph_engine)`: already-marked → `True`;
     non-empty graph → `False` (stays on ledger); empty graph → try
     `set_graph_metadata({version, graph_native})`, `UnsupportedProvenanceCapability`
     → `False`. Only a successful mark returns `True`. This is the inertness gate.

2. **Planner** —
   `cognee/infrastructure/databases/unified/provenance_delete_planner.py`
   - `execute_source_ref_removal(graph_engine, vector_engine, *, node_data,
     edge_data, refs_by_node, refs_by_edge) -> None`.
   - Unowned vs surviving decided from the `source_ref_keys` snapshots:
     `set(node_data[nid].source_ref_keys) - set(refs_by_node[nid])` empty → unowned.
   - Vector ids from snapshots only (mirrors `delete_from_graph_and_vector`):
     node → `f"{node_type}_{field}"` per `indexed_fields`, id = node_id;
     edge → `EdgeType.id_for(edge_text)` in `EdgeType_relationship_name`, and
     `generate_node_id(source_id + relationship_name + target_id)` in
     `Triplet_text` (wrapped in try/except — collection may not exist).
   - Retry-safe order: (1) delete vectors for unowned; (2) `remove_*_source_refs`
     on all matched artifacts; (3) `delete_nodes` / `delete_edge_triples` for the
     unowned ones.
   - Best-effort cleanup parity: prune orphaned `EdgeType` nodes (recompute
     remaining edge texts via `get_graph_data`) and strip orphaned NodeSet tags
     (`node_type == "NodeSet"`, label from `node_properties["name"]`) via
     `graph_engine.remove_belongs_to_set_tags` + `vector_engine.remove_belongs_to_set_tags`.

3. **`UnifiedStoreEngine`** —
   `cognee/infrastructure/databases/unified/unified_store_engine.py`
   - `supports_graph_native_delete()`: GRAPH + VECTOR caps and both engines
     present (no separate provenance-capability flag — unsupported reads raise).
   - `delete_by_source_ref(key)`: find nodes/edges by ref, fetch delete-data,
     build `refs_by_node = {nid: [key]}` / `refs_by_edge = {e: [key]}`, call planner.
   - `delete_by_dataset_id(dataset_id)`: `refs_by_* = find_*_source_refs_by_dataset`,
     fetch delete-data for those ids, call planner.
   - `rollback_by_pipeline_run_id(pipeline_run_id)`:
     `refs_by_* = find_*_source_refs_by_pipeline_run`, fetch delete-data, call planner.

## Remaining Part 2 work (not in the spine)

- **Write stamping** (`cognee/tasks/storage/add_data_points.py`): for marked
  graphs, skip `upsert_nodes`/`upsert_edges`; after graph writes,
  `attach_node_source_refs` / `attach_edge_source_refs` with the
  `make_source_ref_key(dataset.id, data_item.id)` and the run id. One call each;
  the adapter derives dataset/run ids and run refs.
- **Routing**: `delete_data_nodes_and_edges.py`,
  `delete_dataset_nodes_and_edges.py`, `cognify/rollback.py`, and
  `datasets.py delete_data` route to the unified graph-native methods when
  `is_graph_native_graph` and `supports_graph_native_delete`; otherwise keep the
  ledger path. `forget(memory_only=True)` and deprecated `delete()` route
  transitively.

## Tests

- **Unit** —
  `cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_native_delete.py`:
  a `FakeProvenanceGraphEngine` implementing Lazar's provenance methods over
  in-memory nodes/edges keyed by `source_ref_keys` (deriving dataset ids + run
  refs on attach), plus a recording `FakeVectorEngine`. Proves unowned
  hard-delete + shared detach, cross-dataset preservation, rollback removes only
  run-introduced artifacts, vectors-first + retry convergence after an injected
  vector failure, unsupported-capability propagation, and the no-candidate no-op.
  `9 passed`.
- **Integration** (gated) —
  `cognee/tests/integration/tasks/test_graph_native_delete_part2.py`: realistic
  default-stack (Ladybug + LanceDB + SQLite) end-to-end delete/rollback, but
  module-level `pytest.mark.skip` until Part 1 real adapters land (COG-5522 Part
  1). Collects + skips (`3 skipped`).
