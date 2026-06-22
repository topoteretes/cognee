# Ledger-consumer inventory

Verified by reading every file. "The ledger" = the relational `nodes`/`edges` tables (SQLAlchemy `cognee.modules.graph.models.Node`/`Edge`) plus the legacy `GraphRelationshipLedger` (`cognee/modules/graph/legacy/`). A **consumer** is a site that **READS** the ledger to drive behaviour. Writers (`upsert_nodes`/`upsert_edges`, `record_data_in_legacy_ledger`) and pure deleters of ledger rows are noted separately where relevant because Part 3 must retire them too, but the task's focus — read-driven behaviour Part 2 must re-point — is captured in groups (a) and (b).

## Consumer table

| # | file:line | function | reads ledger for | live delete/rollback path? | migrate? |
|---|-----------|----------|------------------|----------------------------|----------|
| 1 | `cognee/modules/graph/methods/get_data_related_nodes.py:13` / `:37` | `get_data_related_nodes` / `get_global_data_related_nodes` | `select(Node)` — finds nodes uniquely owned by `(dataset_id, data_id)` to hard-delete (per-data delete). Multi-user vs single-DB variants. | YES (data delete) | (a) |
| 2 | `cognee/modules/graph/methods/get_data_related_edges.py:13` / `:40` | `get_data_related_edges` / `get_global_data_related_edges` | `select(Edge)` — uniquely-owned edges for per-data delete. | YES (data delete) | (a) |
| 3 | `cognee/modules/graph/methods/get_dataset_related_nodes.py:11` / `:19` | `get_dataset_related_nodes` / `get_global_dataset_related_nodes` | `select(Node)` — nodes to hard-delete for whole-dataset delete. | YES (dataset delete) | (a) |
| 4 | `cognee/modules/graph/methods/get_dataset_related_edges.py:11` / `:16` | `get_dataset_related_edges` / `get_global_dataset_related_edges` | `select(Edge)` — edges for whole-dataset delete. | YES (dataset delete) | (a) |
| 5 | `cognee/modules/graph/methods/has_data_related_nodes.py:10` | `has_data_related_nodes` | `select(Node).limit(1)` — branch decision: if no ledger rows, `datasets.delete_data` falls back to `legacy_delete` (subgraph delete); else `delete_data_nodes_and_edges`. | YES (delete routing) | (a) |
| 6 | `cognee/modules/graph/methods/get_shared_slugs_losing_dataset_anchor.py:13` | `get_shared_slugs_losing_dataset_anchor` | `select(Node.slug)` — shared slugs that survive but lose their `(dataset_id,*)` anchor, so their `belongs_to_set` tags get scoped-detagged. Single-DB delete only. | YES (data delete detag) | (a) |
| 7 | `cognee/modules/graph/methods/get_orphaned_nodeset_labels_for_dataset.py:13` | `get_orphaned_nodeset_labels_for_dataset` | `select(Node.label)` — NodeSet labels the dataset is fully losing; tag list for the scoped detag above. | YES (data delete detag) | (a) |
| 8 | `cognee/modules/graph/methods/delete_from_graph_and_vector.py:18-175` | `delete_from_graph_and_vector` (+ `_get_deleted_edge_retrieval_text`) | Consumes `Node`/`Edge` rows passed in — reads `node.slug`, `node.type`, `node.label`, `node.indexed_fields`, `edge.slug`, `edge.relationship_name`, `edge.source_node_id`, `edge.destination_node_id`, `edge.attributes` to issue graph+vector deletes. Also CALLS `mark_ledger_*_as_deleted`. THE central delete executor for both data and dataset and rollback flows. | YES (all delete + rollback) | (a) |
| 9 | `cognee/modules/graph/methods/delete_data_nodes_and_edges.py:29` | `delete_data_nodes_and_edges` | Orchestrator: calls consumers 1,2,5,6,7,8,10,11 + `delete_data_related_*`. Per-data delete entry point. | YES (data delete) | (a) |
| 10 | `cognee/modules/graph/methods/delete_dataset_nodes_and_edges.py:21` | `delete_dataset_nodes_and_edges` | Orchestrator: calls consumers 3,4,8,10,11 + `delete_dataset_related_*`. Dataset delete entry point. | YES (dataset delete) | (a) |
| 11 | `cognee/modules/graph/legacy/has_nodes_in_legacy_ledger.py:16` | `has_nodes_in_legacy_ledger` (+ `confirm_nodes_in_graph`) | `select(GraphRelationshipLedger.node_label, .source_node_id)` — per-node flag: is this node a legacy (pre-relational-ledger) artifact, so `delete_from_graph_and_vector` skips it. Reads `Node` only as input list. | YES (all delete + rollback) | (a) |
| 12 | `cognee/modules/graph/legacy/has_edges_in_legacy_ledger.py:11` | `has_edges_in_legacy_ledger` | `select(GraphRelationshipLedger.creator_function)` — per-edge legacy flag, same role for edges. | YES (all delete + rollback) | (a) |
| 13 | `cognee/modules/cognify/rollback.py:42` (queries at `:66`, `:80`, `:103-114`, `:122-136`) | `cognify_rollback_handler` | `select(Node)`/`select(Edge)` filtered by `pipeline_run_id` + `dataset_id` to find run-introduced artifacts; aliased `select(distinct(slug))` to exclude still-shared slugs; then `delete(Node)`/`delete(Edge)`. Calls consumers 8,11,12. THE rollback path. | YES (rollback) | (a) |
| 14 | `cognee/modules/graph/methods/delete_data_related_nodes.py:10` | `delete_data_related_nodes` | `delete(Node)` — removes relational ownership rows after graph/vector delete. (Deleter, but on the live delete path; Part 3 retires.) | YES (data delete cleanup) | (a) |
| 15 | `cognee/modules/graph/methods/delete_data_related_edges.py:10` | `delete_data_related_edges` | `delete(Edge)` — ledger ownership cleanup, data delete. | YES (data delete cleanup) | (a) |
| 16 | `cognee/modules/graph/methods/delete_dataset_related_nodes.py:12` | `delete_dataset_related_nodes` | `select(Node)` then `delete(Node)` — ledger ownership cleanup, dataset delete. | YES (dataset delete cleanup) | (a) |
| 17 | `cognee/modules/graph/methods/delete_dataset_related_edges.py:12` | `delete_dataset_related_edges` | `select(Edge)` then `delete(Edge)` — ledger ownership cleanup, dataset delete. | YES (dataset delete cleanup) | (a) |
| 18 | `cognee/modules/graph/legacy/mark_ledger_as_deleted.py:15` / `:45` | `mark_ledger_nodes_as_deleted` / `mark_ledger_edges_as_deleted` | `update(GraphRelationshipLedger ... deleted_at)` — soft-marks legacy ledger entries after delete. Called by consumer 8. | YES (all delete + rollback tail) | (a) |
| 19 | `cognee/modules/graph/methods/get_global_context_graph_inputs.py:13` (queries `:54`, `:199`, `:234`) | `get_dataset_text_summary_ids`, `load_dataset_graph_entity_input` + helpers | `select(Node.slug)`, joins `Node`/`Edge` on `made_from`/`contains` to compute summary→entity maps and IDF/entity counts. Drives memify global-context-index bucketing. | NO (memify analytics) | (b) |
| 20 | `cognee/api/v1/datasets/datasets.py:182-185`, `:115`, `:170` | `Datasets.delete_data` / `delete` | Indirectly via consumers 5,9,10 (and `legacy_delete` fallback). Public delete API entry. | YES (delete API surface) | (a)* indirect |
| 21 | `cognee/api/v1/forget/forget.py:254-264`, `:326-333` | forget handlers | Call consumers 9 and 10 (`delete_data_nodes_and_edges`, `delete_dataset_nodes_and_edges`). | YES (forget = delete) | (a)* indirect |
| 22 | `cognee/tasks/ingestion/resolve_dlt_sources.py:330-367` | dlt orphan cleanup | Calls consumers 5 and 9 to delete orphaned data. | YES (delete) | (a)* indirect |
| 23 | `cognee/modules/migrations/versions/namespace_entity_type_node_ids.py:545-665` | `_migrate_ledger_edges` / `_migrate_ledger_nodes` / `_migrate_ledger` | `select(Node...)`, `select(Edge...)`, `update(GraphRelationshipLedger)` — one-off PK/endpoint remap migration of all three ledgers. | NO (migration) | (c) |

\* indirect = doesn't read the ledger directly, but is a live delete/rollback caller of group-(a) consumers; must keep working when (a) is re-pointed.

### Test-only consumers (incidental, group c)
- `cognee/tests/migrations/test_migration_lockstep.py:43` — imports `Edge, Node`.
- `cognee/tests/unit/modules/graph/test_graph_methods.py:33` — imports `Node, Edge`; tests delete/get methods.
- `cognee/tests/integration/tasks/test_cognify_rollback_recovery.py:27` — imports `Edge, Node`; rollback test.
- `cognee/tests/backwards_compatibility/phase2_verify.py:58` — imports `Edge, Node`.
- `cognee/tests/unit/modules/cognify/test_rollback.py:94-156` — monkeypatches `has_nodes_in_legacy_ledger`, `has_edges_in_legacy_ledger`, `delete_from_graph_and_vector`.
- `cognee/tests/test_delete_default_graph_with_legacy_data_1.py:22/477`, `..._2.py:21/441`, `test_delete_two_users_same_dataset.py:26/321`, `test_delete_two_users_with_legacy_data.py:25/329` — call `record_data_in_legacy_ledger` (the ONLY callers of that writer; it has no production caller).
- `cognee/tests/test_delete_default_graph_non_mocked.py:12-27` — calls `get_data_related_nodes`/`get_data_related_edges`.
- `cognee/tests/unit/modules/graph/test_relational_upserts.py`, `tests/unit/tasks/storage/test_add_data_points.py`, `tests/unit/api/v1/forget/test_forget_memory_only.py`, `tests/unit/modules/graph/test_delete_detag_nodeset.py`, `tests/unit/modules/graph/test_global_context_graph_inputs.py` — exercise upsert/delete/global-context paths.

### Writers (not read-consumers, but Part 3 retires)
- `cognee/modules/graph/methods/upsert_nodes.py` / `upsert_edges.py` — write `nodes`/`edges` rows. Callers: `tasks/storage/add_data_points.py:99/108/118`, `tasks/ingestion/extract_dlt_fk_edges.py:266`, `tasks/codingagents/coding_rule_associations.py:131`. These must instead stamp graph-native provenance (`source_refs`/`source_run_refs`/`dataset_ids`/`delete_mode`) per the Part 0 contract.
- `cognee/modules/graph/legacy/record_data_in_legacy_ledger.py` — writes `GraphRelationshipLedger`; only test callers; effectively dead in production.

## Migration classification

### (a) Delete / rollback core — MUST migrate for Part 2/3
These read the ledger (or pass ledger rows / delete ledger rows) to decide and execute what gets removed. Part 2 re-points them to the graph-native read primitives (`get_nodes/edges_delete_data_by_source_ref` / `_by_dataset_id` / `_by_source_run_ref` on `GraphDBInterface`, and `delete_by_source_ref` / `delete_by_dataset_id` / `rollback_by_pipeline_run_id` on `GraphVectorStoreInterface`). Part 3 deletes them.

- **Selection/discovery (read `select(Node)`/`select(Edge)`):** consumers 1, 2, 3, 4, 5, 6, 7 — superseded by `NodeDeleteData`/`EdgeDeleteData` returned from the new graph read primitives keyed on `source_ref`/`dataset_id`.
- **Rollback discovery (read by `pipeline_run_id`):** consumer 13 (`cognify_rollback_handler`) — superseded by `rollback_by_pipeline_run_id` / `get_*_delete_data_by_source_run_ref` (source-run refs).
- **Delete executor (consumes Node/Edge fields):** consumer 8 (`delete_from_graph_and_vector`) — its inputs become `NodeDeleteData`/`EdgeDeleteData`; its work moves behind the `GraphVectorStoreInterface` `delete_by_*` methods returning `ProvenanceDeleteResult`.
- **Legacy-flag checks:** consumers 11, 12 (`has_nodes/edges_in_legacy_ledger`) — replaced by the `delete_mode` / `provenance_version` markers stamped on graph artifacts (`DELETE_MODE_GRAPH_NATIVE` vs `DELETE_MODE_LEDGER`).
- **Ledger row cleanup / soft-mark (writes/deletes ledger):** consumers 14, 15, 16, 17, 18 — become unnecessary once provenance is on the graph; Part 3 deletes them.
- **Orchestrators / API surface (indirect, must keep working):** consumers 9, 10, 20, 21, 22.

### (b) Other live reads — non-delete, must be re-pointed or kept
- **Consumer 19** (`get_global_context_graph_inputs` + memify `global_context_index` callers `bucketing/graph/inputs.py:8`, `update.py:9`). Reads `nodes`/`edges` to compute summary→entity links and IDF/entity counts for memify global-context bucketing. It is NOT on the delete/rollback path, but it depends on the relational ledger existing. When Part 3 retires the `nodes`/`edges` tables, this analytics read must be re-sourced from the graph (or kept alive by a compatibility shim). Flag it for Part 2/3 even though it isn't delete/rollback.

### (c) Tests / migrations / incidental
- **Migration:** consumer 23 (`namespace_entity_type_node_ids.py` `_migrate_ledger*`). A frozen, version-pinned one-off Alembic-style migration. Do not re-point; it intentionally operates on the historical relational ledger shape. Part 3 may leave it as-is (historical) or gate it.
- **All test files** listed under "Test-only consumers" above. Update/retire alongside the code they cover; `record_data_in_legacy_ledger` and its 4 legacy-data test callers can be deleted with the legacy ledger.
- **Writers** (`upsert_nodes`/`upsert_edges` and their 4 task callers) are not read-consumers but are the producers of the relational ledger; Part 1/2 repurposes them to stamp graph-native provenance and Part 3 drops the relational writes.

## Key cross-references to the Part 0 contract
- The `(dataset_id, data_id)` ownership that consumers 1-7 read maps to **source refs** (`provenance.make_source_ref`, stored under `SOURCE_REFS_KEY`); "uniquely owned ⇒ hard delete" becomes "removing the last `source_ref`".
- The `pipeline_run_id` filter in consumer 13 maps to **source-run refs** (`make_source_run_ref`, `SOURCE_RUN_REFS_KEY`).
- The `is_legacy_*` flags from consumers 11/12 map to `DELETE_MODE_KEY` (`DELETE_MODE_LEDGER` vs `DELETE_MODE_GRAPH_NATIVE`) and `PROVENANCE_VERSION_KEY`.
- Dataset-scoped delete (consumers 3,4) needs `DATASET_IDS_KEY` because source refs are opaque hashes.
- `delete_from_graph_and_vector` (consumer 8) field usage (`slug`, `type`, `label`, `indexed_fields`, edge identity, `edge_text`) is exactly mirrored by `NodeDeleteData` / `EdgeDeleteData` / `EdgeIdentity`.
