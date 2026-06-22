# Part 2 — graph-native delete & rollback for new graphs

This is the implementation plan for Part 2 of the graph-native delete/rollback
work (see `phases_overview.md`). It builds directly on the locked Part 0 contract
(`part0_contract.md`) and runs in parallel with Part 1 (storage primitives).

**Goal.** Re-point `delete`, dataset-delete, `forget`, and `rollback` at the
graph itself — reading provenance off nodes/edges instead of the relational
`nodes`/`edges` ledger — **but only for graphs created under
`delete_mode="graph_native"`**. Every pre-existing (unmarked) graph keeps the
ledger path untouched until Part 3 migrates it. No data is dropped, no API
surface changes, and the new path is exercised end-to-end against the Part 0
fake stores before any real backend is wired (that gate is owned by Part 1).

## Where this sits relative to Part 0 and Part 1

Part 0 already shipped the *spine* of Part 2 — committed, with no live caller:

| Spine file | Role |
| --- | --- |
| `cognee/modules/graph/provenance/markers.py` | `is_graph_native_graph` / `ensure_graph_native_for_new_graph` — **the routing authority**. |
| `cognee/modules/graph/provenance/constants.py` | marker values + property keys (`SOURCE_REFS_KEY`, `SOURCE_RUN_REFS_KEY`, `DATASET_IDS_KEY`, `DELETE_MODE_KEY`, `PROVENANCE_VERSION_KEY`). |
| `cognee/modules/graph/provenance/refs.py` | `make_source_ref(dataset_id, data_id)` / `make_source_run_ref(dataset_id, pipeline_run_id)`. |
| `cognee/modules/graph/provenance/snapshots.py` | `EdgeIdentity` / `NodeDeleteData` / `EdgeDeleteData` (now incl. `dataset_ids`). |
| `cognee/modules/graph/provenance/results.py` | `ProvenanceDeleteResult`. |
| `cognee/infrastructure/databases/unified/provenance_delete_planner.py` | retry-safe ref-removal planner (`execute_ref_removal`). |
| `cognee/infrastructure/databases/unified/unified_store_engine.py` | `UnifiedStoreEngine.delete_by_source_ref` / `delete_by_dataset_id` / `rollback_by_pipeline_run_id`. |
| `cognee/infrastructure/databases/graph/graph_db_interface.py` | provenance read + write primitives (raising defaults). |
| `cognee/tests/unit/modules/graph/provenance/test_unified_graph_native_delete.py` | the reference test harness (fake provenance graph + recording vector engine). |
| `cognee/tests/unit/modules/graph/provenance/test_fake_graph_vector_store.py` | Part 0 reference semantics. |

What is **not yet** done — and is the subject of this plan — is the *wiring*:
write-site stamping, the routing branches in the live delete/rollback entry
points, the ledger-consumer guards, and the integration gate. The planner and
`UnifiedStoreEngine` already pass against the Part 0 fakes; Part 2 makes the
live code paths reach them for graph-native graphs, while leaving old graphs on
the ledger.

Part 2 depends on Part 1 only for the **real adapter** implementations of the
`GraphDBInterface` read/write primitives and the LanceDB vector reads. Against
the fakes, Part 2 is self-contained and complete today.

## The six phases

```
Part 0 (spine, committed)
   │
   ├── Phase 1  delete orchestration   ─┐
   ├── Phase 2  rollback orchestration ─┤ (validated vs Part 0 fakes)
   ├── Phase 3  write stamping         ─┤
   ├── Phase 4  delete/rollback routing ┤
   ├── Phase 5  ledger consumers        ┘
   └── Phase 6  integration gate  ── gated on Part 1 (kept skipped; PR stays draft)
```

### Phase 1 — delete orchestration

Drive single-data-item and whole-dataset deletes through
`UnifiedStoreEngine.delete_by_source_ref` / `delete_by_dataset_id` instead of
the ledger discovery + `delete_from_graph_and_vector` chain.

- A per-data delete computes `make_source_ref(dataset_id, data_id)` and calls
  `delete_by_source_ref`. The engine reads `NodeDeleteData`/`EdgeDeleteData` off
  the graph (`get_nodes_delete_data_by_source_ref` /
  `get_edges_delete_data_by_source_ref`), and the planner removes that one ref.
- A whole-dataset delete calls `delete_by_dataset_id(dataset_id)`, which reads by
  `DATASET_IDS_KEY` and removes the dataset id ref.
- Orchestration **never reads relational `Node`/`Edge` rows** — vector ids are
  derived only from the snapshots, via the Part 0 helpers (see
  `provenance_delete_planner.py`):
  - node vectors: collection `f"{node_type}_{field}"`, id `= node_id`;
  - edge type vectors: `EdgeType.id_for(edge_retrieval_text)` in
    `EdgeType_relationship_name`;
  - triplet vectors: `generate_node_id(source + relationship + target)` in
    `Triplet_text` (absent-collection tolerated).
- Ownership rule (mirrors the ledger's "unique by slug"): an artifact is
  hard-deleted only when the removed ref was its *last* owning ref; otherwise it
  is **detached** (ref stripped, artifact kept). Custom edges are deleted by the
  same path because they carry the same source refs (see Phase 3).

Validated by `test_unified_graph_native_delete.py`:
`test_delete_by_source_ref_deletes_unowned_detaches_shared`,
`test_delete_by_dataset_id_preserves_cross_dataset`,
`test_edge_vector_ids_use_part0_helpers`, `test_no_candidates_is_noop`.

### Phase 2 — rollback orchestration

Drive `cognify_rollback_handler` through
`UnifiedStoreEngine.rollback_by_pipeline_run_id(pipeline_run_id, dataset_id)`.

- The engine computes `make_source_run_ref(dataset_id, pipeline_run_id)`, reads
  by `SOURCE_RUN_REFS_KEY`, and removes that run ref.
- Rollback hard-deletes only what the run **solely** introduced: an artifact
  survives if it still carries any `source_ref` **or** any *other*
  `source_run_ref`. This is the graph-native equivalent of the ledger's
  "exclude still-shared slugs" aliased query.
- Rollback is dataset-scoped: the run ref is a hash of
  `(dataset_id, pipeline_run_id)`, so a rollback for a run in a different dataset
  reads nothing and no-ops.
- After the store op, the handler still resets `pipeline_status` exactly as
  today — only the artifact discovery/removal moves to the graph-native path.

Validated by `test_rollback_removes_only_run_introduced`,
`test_rollback_is_dataset_scoped`.

### Phase 3 — write stamping

At the single central write site
(`cognee/tasks/storage/add_data_points.py`), for graph-native graphs, stamp
provenance **onto the graph artifacts** and **skip the relational ledger
upserts** entirely. The stamping rules:

1. **Refs are attached after a successful graph write**, never before. The graph
   `add_nodes`/`add_edges` (and `add_nodes_with_vectors`/`add_edges_with_vectors`
   on hybrid backends) must succeed first; only then are
   `source_refs` / `source_run_refs` / `dataset_ids` / `delete_mode` /
   `provenance_version` stamped. A write that fails leaves no orphaned
   provenance, mirroring the planner's "mutate provenance last" discipline on
   the delete side.
2. **Marked graphs skip the ledger.** When `is_graph_native_graph(graph_engine)`
   is true, `add_data_points` does **not** call `upsert_nodes`/`upsert_edges`
   (rows 1–3 of `write_site_inventory.md`); the on-graph stamp is the single
   source of truth. Unmarked (old) graphs keep writing ledger rows.
3. **Custom edges are stamped too.** The `custom_edges` branch
   (`add_data_points.py:147`) attaches the same source refs / run refs /
   dataset ids as data-point edges, so a later `delete_by_source_ref` /
   `rollback_by_pipeline_run_id` finds and removes them. (Today they reuse the
   same ledger session; under graph-native they reuse the same stamp.)
4. **No dataset / no data-item ⇒ non-provenanced.** The existing guard
   (`add_data_points.py:94`, requires `user and dataset and data_item`) is
   preserved: when any is missing, the graph write proceeds with **no source
   refs and no dataset ids** — exactly as today it writes no ledger rows. Such
   artifacts are simply not reachable by graph-native delete (same coverage gap
   the ledger has, documented in `write_site_inventory.md`).
5. **No `pipeline_run_id` ⇒ refs but no run ownership.** When `dataset` and
   `data_item` are present but `pipeline_run_id` is `None` (e.g. the
   coding-rules edge persistence and the migration loader's `_provenance_ctx`,
   per `write_site_inventory.md`), the artifact still gets `source_refs` and
   `dataset_ids` (so per-data / dataset delete works) but **no
   `source_run_refs`** (so rollback cannot scope it to a run — matching the
   ledger's `pipeline_run_id=None` behaviour exactly).

Acceptance for this phase is structural: a graph-native `add_data_points` writes
**zero** `upsert_nodes`/`upsert_edges` calls and attaches node + edge source
refs (incl. custom edges); writes without dataset/data-item stay
non-provenanced; writes without `pipeline_run_id` attach refs but create no run
ownership.

> Marking happens via `ensure_graph_native_for_new_graph(graph_engine)` called
> **before the first nodes are written** on a brand-new graph (so an empty graph
> is distinguishable from a populated pre-Part-2 one). The natural call site is
> the pipeline's first graph-writing step / engine bootstrap, before
> `add_data_points` runs.

### Phase 4 — delete / rollback routing

Each live entry point gains a single front-door branch keyed on the marker, then
either calls the unified boundary (graph-native) or the existing ledger flow
(old graphs). **The marker is the routing authority** — see "How the marker
routes" below.

Entry points that route through the unified boundary for graph-native graphs:

| Entry point (file) | Today (ledger path, kept for old graphs) | Graph-native route |
| --- | --- | --- |
| `cognee/api/v1/datasets/datasets.py` — `Datasets.delete_data` | `has_data_related_nodes` → `delete_data_nodes_and_edges` (or `legacy_delete`) | `delete_by_source_ref` |
| `cognee/api/v1/datasets/datasets.py` — `Datasets.delete` (whole dataset) | `delete_dataset_nodes_and_edges` | `delete_by_dataset_id` |
| `cognee/api/v1/forget/forget.py` — forget handlers (incl. `memory_only=True`) | `delete_data_nodes_and_edges` / `delete_dataset_nodes_and_edges` | `delete_by_source_ref` / `delete_by_dataset_id` |
| `cognee/api/v1/delete/routers/get_delete_router.py` — deprecated `delete()` | funnels into `Datasets.delete_data` | same branch as `delete_data` |
| `cognee/modules/cognify/rollback.py` — `cognify_rollback_handler` | `select(Node)`/`select(Edge)` by `pipeline_run_id` → `delete_from_graph_and_vector` | `rollback_by_pipeline_run_id` |

`forget(memory_only=True)` is explicitly in scope: it removes graph/vector
artifacts while leaving the relational data records, which is exactly what the
graph-native delete does (it never touches relational rows). The deprecated
`delete()` router and the dlt orphan-cleanup caller
(`tasks/ingestion/resolve_dlt_sources.py`) inherit the branch from the
orchestrators they call, so they need no separate change beyond the shared
front door.

**Old graphs keep the ledger path.** The branch is `if await
is_graph_native_graph(graph_engine): <unified boundary> else: <existing ledger
flow>`. Because `is_graph_native_graph` returns `False` for any graph without
the marker (and fails safe to `False` on any read error), every pre-Part-2 graph
takes the unchanged ledger path. The ledger consumers (groups (a) in
`ledger_consumer_inventory.md`) are not removed — that is Part 3.

### Phase 5 — ledger consumers

Ledger consumers read graph provenance **only inside their graph-native
branch**; their ledger reads are unchanged on the old-graph branch.

- The selection/discovery consumers (`get_data_related_nodes/edges`,
  `get_dataset_related_nodes/edges`, `has_data_related_nodes`,
  `get_shared_slugs_losing_dataset_anchor`,
  `get_orphaned_nodeset_labels_for_dataset`) and the central executor
  `delete_from_graph_and_vector` are **bypassed** on the graph-native branch
  (their work is done by the planner + `UnifiedStoreEngine`), and **kept intact**
  on the old-graph branch.
- The legacy-flag checks (`has_nodes_in_legacy_ledger` /
  `has_edges_in_legacy_ledger`) are not consulted on the graph-native branch —
  the `delete_mode` marker already answered "is this graph-native?" at the
  front door.
- The non-delete analytics consumer
  (`get_global_context_graph_inputs`, group (b) in the inventory) is **out of
  Part 2 scope**: it still reads the ledger. Part 3 re-sources it. Because Part 2
  only *adds* on-graph stamps and never drops ledger rows for old graphs, this
  consumer keeps working throughout.

Acceptance: ledger consumers read graph provenance only in their graph-native
branch; old unmarked graphs keep the ledger path bit-for-bit.

### Phase 6 — integration gate (gated on Part 1)

A default-stack integration suite (Ladybug graph + LanceDB vector + SQLite)
that runs a real add → cognify → delete / rollback cycle and asserts the
graph-native path actually removed the right nodes/edges/vectors and detached
the shared ones. This suite is **authored in Part 2 but kept skipped** until
Part 1 lands the real adapter primitives (`supports_graph_native_provenance`,
the `get_*_delete_data_*` reads, `detach_provenance_refs_from_*`, `delete_edges`
on Ladybug; the LanceDB vector reads). Until then:

- the suite is marked skipped (e.g. `pytest.mark.skip(reason="needs Part 1
  adapter primitives")`);
- the Part 2 PR stays **draft** so it cannot merge ahead of its dependency;
- all the *unit-level* acceptance criteria are met today against the Part 0
  fakes, which require no real backend.

## How the marker is the routing authority

`markers.py` is the single decision point. Two functions, both reading/writing
one well-known marker node through the existing `GraphDBInterface` node CRUD —
no Part 1 primitives required:

- `is_graph_native_graph(graph_engine)` — returns `True` iff the graph carries
  the marker node (`GRAPH_NATIVE_MARKER_NODE_ID`, type
  `GraphNativeProvenanceMarker`) with `delete_mode == "graph_native"`. **Any read
  error returns `False`**, so routing fails safe onto the ledger.
- `ensure_graph_native_for_new_graph(graph_engine)` — marks a brand-new
  (`is_empty()`) graph as graph-native and returns `True`; returns `False` for a
  populated-but-unmarked graph (an old graph that must stay on the ledger).
  Idempotent.

This gives the clean cut Part 2 needs: every entry point in Phase 4 calls
`is_graph_native_graph` once and branches. A graph is graph-native if and only
if it was created after Part 2 marked it at first write (Phase 3). No
per-artifact inspection, no migration, no ambiguity — and old graphs are
*structurally* incapable of taking the new path until Part 3 backfills the
marker.

## The planner's retry-safe ordering

`execute_ref_removal` (`provenance_delete_planner.py`) is the heart of all three
`UnifiedStoreEngine` ops. Its ordering is what makes a partial failure safe:

```
read provenance + payloads   (caller: get_*_delete_data_by_*)
 → compute vector ids         (from snapshots only — never relational rows)
 → delete vectors             (unowned artifacts first)
 → detach surviving artifacts (strip the ref in the graph; idempotent)
 → delete unowned artifacts   (graph; idempotent)
```

Because **every graph mutation happens after the vector deletes**:

- a failed vector delete leaves **graph provenance untouched**, so a retry
  re-reads the identical state and converges;
- the detach and delete steps are idempotent (removing an absent ref / deleting
  an absent node is a no-op), so a retry after a partial *graph* mutation also
  converges to the clean final state;
- unsupported-capability errors from the reads propagate to the caller
  **before any mutation** — the engine never half-deletes when a backend can't
  answer a read.

The survival predicates differ per op (defined in `unified_store_engine.py`):
source-ref delete survives on another `source_ref`; dataset delete survives on
another dataset id; rollback survives on any `source_ref` or any other
`source_run_ref`. The planner is otherwise op-agnostic.

This is exercised directly by
`test_vector_failure_leaves_graph_untouched_then_retry_converges` and
`test_unsupported_capability_propagates_before_mutation`.

## Acceptance criteria

All unit-level criteria are met today against the Part 0 fakes
(`test_unified_graph_native_delete.py`, `test_fake_graph_vector_store.py`):

1. `delete_by_source_ref` / `delete_by_dataset_id` / `rollback_by_pipeline_run_id`
   pass vs the Part 0 fakes: **vectors deleted first**, refs removed from shared
   artifacts (detach), only unowned artifacts hard-deleted, no-candidate request
   is a no-op.
2. Edge/triplet vector ids use the Part 0 helpers (`EdgeType.id_for`,
   `generate_node_id`); orchestration **never reads relational `Node`/`Edge`
   rows**.
3. An injected vector-delete failure raises with **graph provenance untouched**,
   and a retry converges.
4. Unsupported capabilities propagate **before any mutation**.
5. Graph-native `add_data_points` attaches node/edge source refs incl. custom
   edges and writes **no** `upsert_nodes`/`upsert_edges`.
6. Writes without dataset/data-item stay **non-provenanced**; writes without
   `pipeline_run_id` attach refs but create **no run ownership**.
7. Graph-native delete / dataset-delete / `forget(memory_only=True)` /
   deprecated `delete()` / rollback route through the unified boundary, while old
   unmarked graphs keep the ledger path.
8. Ledger consumers read graph provenance **only in their graph-native branch**.

**Gated on Part 1:** the default-stack integration gate (Phase 6). It is
authored in Part 2 but **kept skipped** until Part 1 implements the real Ladybug
+ LanceDB primitives, and the **PR stays draft** until then.

## First backend scope

Inherited from Part 0: **Ladybug (graph) + LanceDB (vector) + SQLite
(relational)**. Other graph/vector backends keep
`supports_graph_native_provenance() == False`, so
`UnifiedStoreEngine.supports_graph_native_delete()` is `False` and they fall back
to the ledger path regardless of the marker. (As a defensive belt-and-braces,
the Phase 4 branch can require both `is_graph_native_graph(...)` **and**
`unified.supports_graph_native_delete()` before taking the unified boundary.)

## Out of scope (Part 3)

- Backfilling on-graph provenance for ledger-era graphs and flipping them to
  `graph_native`.
- Removing the ledger consumers (groups (a) in `ledger_consumer_inventory.md`)
  and dropping the `nodes`/`edges` tables and the legacy
  `GraphRelationshipLedger`.
- Re-sourcing the memify global-context analytics read (consumer 19) off the
  graph.
