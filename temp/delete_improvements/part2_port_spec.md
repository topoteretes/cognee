# Part 2 port spec — rebase onto Lazar's Part 0 (COG-5522)

Part 2 (graph-native delete/rollback for new graphs) was originally built on a
*different* Part 0 contract. This spec re-ports it onto **Lazar's official Part 0**
(`cognee/infrastructure/databases/provenance/` + the provenance methods on
`GraphDBInterface` + `GraphVectorStoreInterface`). Follow this exactly; it is the
authority. Read the live contract files first:

- `cognee/infrastructure/databases/provenance/__init__.py`, `constants.py`,
  `source_refs.py`, `delete_data.py`
- `cognee/infrastructure/databases/exceptions/exceptions.py`
  (`UnsupportedProvenanceCapability`, a `CogneeApiError`, HTTP 501, `log=False`)
- `cognee/infrastructure/databases/graph/graph_db_interface.py` (provenance section)
- `cognee/infrastructure/databases/unified/unified_store_engine.py`,
  `graph_vector_store_interface.py`
- `cognee/tests/unit/infrastructure/databases/provenance/test_provenance_contract.py`,
  `fakes.py`

## Lazar's contract (the target API)

**Refs are parseable strings** (NOT opaque hashes):
- `make_source_ref_key(dataset_id: UUID, data_id: UUID) -> "source_ref:v1:{dataset}:{data}"`
- `make_source_run_ref(pipeline_run_id: UUID, source_ref_key: str) -> "source_run_ref:v1:{run}:{source_ref_key}"`
- getters: `get_dataset_id_from_source_ref_key`, `get_data_id_from_source_ref_key`,
  `get_pipeline_run_id_from_source_run_ref`, `get_source_ref_key_from_source_run_ref`

**Constants:** `GRAPH_PROVENANCE_VERSION = "1"` (str), `GRAPH_DELETE_MODE_GRAPH_NATIVE = "graph_native"`,
`GRAPH_PROVENANCE_VERSION_KEY = "provenance_version"`, `GRAPH_DELETE_MODE_KEY = "delete_mode"`.

**Dataclasses** (`delete_data.py`):
- `EdgeIdentity(source_id, target_id, relationship_name)` — NOTE field order: source, target, relationship.
- `NodeDeleteData(node_id, node_type, indexed_fields, node_properties, source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs)`
- `EdgeDeleteData(edge, edge_text, edge_properties, source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs)`

**`GraphDBInterface` provenance methods** (all raise `UnsupportedProvenanceCapability` by default):
- Write: `attach_node_source_refs(node_ids, source_ref_keys, pipeline_run_id=None)`,
  `attach_edge_source_refs(edges, source_ref_keys, pipeline_run_id=None)`,
  `remove_node_source_refs(node_ids, source_ref_keys)`,
  `remove_edge_source_refs(edges, source_ref_keys)`,
  `delete_edge_triples(edges)`
- Read: `get_node_delete_data(node_ids) -> dict[str, NodeDeleteData]`,
  `get_edge_delete_data(edges) -> dict[EdgeIdentity, EdgeDeleteData]`,
  `find_nodes_by_source_ref(source_ref_key) -> list[str]`,
  `find_edges_by_source_ref(source_ref_key) -> list[EdgeIdentity]`,
  `find_node_source_refs_by_dataset(dataset_id) -> dict[str, list[str]]`,
  `find_edge_source_refs_by_dataset(dataset_id) -> dict[EdgeIdentity, list[str]]`,
  `find_node_source_refs_by_pipeline_run(pipeline_run_id) -> dict[str, list[str]]`,
  `find_edge_source_refs_by_pipeline_run(pipeline_run_id) -> dict[EdgeIdentity, list[str]]`
- Marker: `set_graph_metadata(metadata: dict[str,str])`, `get_graph_metadata() -> dict[str,str]`

**`GraphVectorStoreInterface`** (delete_by_source_ref / delete_by_dataset_id /
rollback_by_pipeline_run_id) take **str** args, return **None**, raise by default.
`rollback_by_pipeline_run_id(pipeline_run_id: str)` is **single-arg**.

## What to build (all NEW Part 2 code; keep old graphs on the ledger)

### 1. Marker module — `cognee/infrastructure/databases/provenance/markers.py`
- `async def is_graph_native_graph(graph_engine) -> bool`: read
  `await graph_engine.get_graph_metadata()`; return
  `meta.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_NATIVE`. Catch ANY
  exception (incl. `UnsupportedProvenanceCapability`) → return `False` (fail safe
  to the ledger path).
- `async def ensure_graph_native_for_new_graph(graph_engine) -> bool`: if already
  graph-native → `True`. Else if `await graph_engine.is_empty()` is False → `False`
  (old graph stays on ledger). Else try
  `await graph_engine.set_graph_metadata({GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION, GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_NATIVE})`;
  on `UnsupportedProvenanceCapability` (backend has no provenance, i.e. pre-Part-1)
  return `False`. Return `True` only after a successful mark.
  This is the inertness gate: on the default stack `set_graph_metadata` raises, so
  nothing is ever marked → Part 2 is inert until Part 1.

### 2. Planner — `cognee/infrastructure/databases/unified/provenance_delete_planner.py`
A retry-safe `execute_source_ref_removal(graph_engine, vector_engine, *, node_data: dict[str, NodeDeleteData], edge_data: dict[EdgeIdentity, EdgeDeleteData], refs_by_node: dict[str, list[str]], refs_by_edge: dict[EdgeIdentity, list[str]]) -> None`:
- Determine unowned vs surviving from the snapshots: a node is **unowned** iff
  `set(node_data[nid].source_ref_keys) - set(refs_by_node[nid])` is empty (removing
  these refs leaves no owning ref); else it **survives** (detach). Same for edges
  via `source_ref_keys`.
- Vector ids **from snapshots only** (mirror `delete_from_graph_and_vector`):
  node → `f"{node_type}_{field}"` for field in `indexed_fields`, id = node_id;
  edge → `EdgeType.id_for(edge_text)` in `EdgeType_relationship_name`, and
  `generate_node_id(source_id + relationship_name + target_id)` in `Triplet_text`
  (wrap Triplet delete in try/except — collection may not exist).
- **Ordering (retry-safe):** (1) delete vectors for unowned; (2) `remove_node_source_refs`
  / `remove_edge_source_refs` for the targeted refs on ALL matched artifacts
  (idempotent); (3) `delete_nodes(unowned node ids)` and `delete_edge_triples(unowned edges)`.
  Vectors first so a failure leaves graph provenance intact and retry converges.
- Post-delete cleanup parity with `delete_from_graph_and_vector` (best-effort, non-fatal,
  try/except): strip orphaned NodeSet tags via `graph_engine.remove_belongs_to_set_tags`
  + `vector_engine.remove_belongs_to_set_tags` using the NodeSet label read from the
  unowned nodes' `node_properties` (NodeSet name; `node_type == "NodeSet"`); and prune
  orphaned `EdgeType` nodes by recomputing remaining edge texts from
  `graph_engine.get_graph_data()`.

### 3. `UnifiedStoreEngine` delete/rollback (in `unified_store_engine.py`)
Add the three methods (return **None**, str args) + `supports_graph_native_delete()`.
- `supports_graph_native_delete()`: GRAPH+VECTOR caps and graph present. (Lazar has no
  `supports_graph_native_provenance` flag; routing relies on `is_graph_native_graph`,
  and unsupported reads raise.)
- `delete_by_source_ref(source_ref_key)`: `node_ids = await graph.find_nodes_by_source_ref(key)`;
  `edges = await graph.find_edges_by_source_ref(key)`; `node_data = await graph.get_node_delete_data(node_ids)`;
  `edge_data = await graph.get_edge_delete_data(edges)`; build `refs_by_node = {nid: [key]}`,
  `refs_by_edge = {e: [key]}`; call the planner.
- `delete_by_dataset_id(dataset_id)`: `refs_by_node = await graph.find_node_source_refs_by_dataset(dataset_id)`
  (dict node_id→source_ref_keys owned by the dataset); same for edges; fetch delete-data for
  those ids; call the planner (removing those dataset-owned refs; unowned if no refs remain).
- `rollback_by_pipeline_run_id(pipeline_run_id)`: `refs_by_node = await graph.find_node_source_refs_by_pipeline_run(pipeline_run_id)`
  (source_ref_keys this run attached, per node); same for edges; fetch delete-data; call the
  planner (removing the run's attached refs; unowned if no refs remain — i.e. the run solely
  introduced the artifact).

### 4. Write stamping — `cognee/tasks/storage/add_data_points.py`
- Before the first node write, `is_graph_native = await ensure_graph_native_for_new_graph(graph_engine)`
  (only inside the `if user and dataset and data_item:` block).
- For marked graphs: SKIP `upsert_nodes`/`upsert_edges`. After the graph writes succeed,
  `source_ref_key = make_source_ref_key(dataset.id, data_item.id)`; build node id list and
  `EdgeIdentity(str(e[0]), str(e[1]), e[2])` list (source, target, relationship) for `edges`
  (incl. custom edges, already extended into `edges`). Call
  `await graph_engine.attach_node_source_refs(node_ids, [source_ref_key], str(pipeline_run_id) if pipeline_run_id else None)`
  and `attach_edge_source_refs(edge_ids, [source_ref_key], <same run arg>)`. (ONE call each —
  the adapter derives dataset_ids/run_ids/run_refs.)
- Non-graph-native graphs keep the existing ledger upserts unchanged.

### 5. Routing (old/unmarked graphs keep the ledger path; auth unchanged)
- `cognee/modules/graph/methods/delete_data_nodes_and_edges.py`: after auth, if
  `is_graph_native_graph(graph_engine)` and `unified.supports_graph_native_delete()` →
  `await unified.delete_by_source_ref(make_source_ref_key(dataset_id, data_id)); return`.
- `delete_dataset_nodes_and_edges.py`: → `await unified.delete_by_dataset_id(str(dataset_id)); return`.
- `cognee/modules/cognify/rollback.py`: after the missing-id guard, if graph-native →
  `await unified.rollback_by_pipeline_run_id(str(pipeline_run_id))`, reset pipeline_status for
  the run's data ids (reuse the existing reset logic / factor a helper), `return`.
- `cognee/api/v1/datasets/datasets.py` `delete_data`: graph-native graphs have no ledger rows,
  so the `has_data_related_nodes` gate would wrongly route to `legacy_delete`. Add: if
  `is_graph_native_graph(graph_engine)` → `delete_data_nodes_and_edges(...)`; else the existing gate.
- `forget(memory_only=True)` and deprecated `delete()` route transitively (no edit).

### 6. Tests + doc
- Unit tests `cognee/tests/unit/.../provenance/test_unified_graph_native_delete.py`:
  a `FakeProvenanceGraphEngine` implementing Lazar's provenance methods (find_*/get_*_delete_data/
  attach_*/remove_*/delete_edge_triples/get_graph_metadata/set_graph_metadata/is_empty/get_graph_data/
  delete_nodes/remove_belongs_to_set_tags) over in-memory nodes/edges carrying source_ref_keys
  (+ derived dataset_ids/run refs), plus a recording `FakeVectorEngine`. Prove: unowned delete +
  shared detach; cross-dataset preservation; rollback removes only run-introduced; vectors-first /
  retry convergence on injected vector failure; unsupported-capability propagation; no-candidate
  no-op. Match Lazar's `EdgeIdentity(source, target, relationship)` order and parseable refs.
- Integration test (skipped, module `pytest.mark.skip` until Part 1):
  `cognee/tests/integration/tasks/test_graph_native_delete_part2.py`.
- Doc: `cognee/temp/delete_improvements/phase2_graph_native_new_graphs.md` updated for Lazar's contract.

## Semantics summary
- Ownership = `source_ref_keys` on the artifact. Removing a ref set → unowned iff none remain.
- Dataset delete removes the dataset's refs (from `find_*_source_refs_by_dataset`).
- Rollback removes the refs a run attached (from `find_*_source_refs_by_pipeline_run`).
- Vectors deleted before any graph mutation; remove-refs + deletes are idempotent → retry-safe.
- Marker = graph metadata; inert until Part 1 because `set_graph_metadata`/`attach_*` raise on
  pre-Part-1 backends.
