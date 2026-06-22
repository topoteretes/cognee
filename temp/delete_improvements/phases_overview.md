# Graph-native delete & rollback — four-part plan

## Problem

Today `delete` and `rollback` do not trust the graph. To know which nodes and
edges to remove they read a **relational ledger**: the `nodes` and `edges`
tables (`cognee/modules/graph/models/Node.py`, `Edge.py`), written by
`upsert_nodes` / `upsert_edges` and keyed by
`(tenant_id, user_id, dataset_id, data_id, pipeline_run_id, slug)`. An older
`GraphRelationshipLedger` table (`cognee/modules/graph/legacy/`) backs a
compatibility path for graphs created before the current ledger.

- **Delete** (`cognee/modules/graph/methods/delete_data_nodes_and_edges.py`,
  `delete_dataset_nodes_and_edges.py`) selects ledger rows by
  `(dataset_id, data_id)`, computes which slugs are uniquely owned, deletes
  those from the graph + vector stores via `delete_from_graph_and_vector`, then
  deletes the relational rows.
- **Rollback** (`cognee/modules/cognify/rollback.py`) selects ledger rows by
  `(pipeline_run_id, dataset_id)`, computes uniqueness, deletes graph/vector
  artifacts, deletes the rows, and resets `pipeline_status`.

The ledger is a second source of truth that must be kept consistent with the
graph. We want **provenance to live on the graph itself** so delete and
rollback are graph-native, and then **retire the ledger**.

## Target contract

Each graph node/edge (and its vector payload) carries provenance directly:

| Property (`provenance/constants.py`) | Meaning |
| --- | --- |
| `provenance_version` (=1) | On-graph provenance layout version. |
| `delete_mode` (`"graph_native"` / `"ledger"`) | Whether this artifact can be deleted graph-natively or still needs the ledger. |
| `source_refs` | Set of `make_source_ref(dataset_id, data_id)` tokens — ingestion sources owning the artifact. |
| `source_run_refs` | Set of `make_source_run_ref(dataset_id, pipeline_run_id)` tokens — pipeline runs that touched it. |
| `dataset_ids` | Datasets the artifact belongs to (source refs are opaque, so dataset-scoped delete needs this). |

An artifact is hard-deleted only when its **last** owning `source_ref` is
removed; shared artifacts are *detached* (one ref stripped) and survive. This
reproduces the ledger's "unique by slug" logic without the ledger. Rollback
strips a `source_run_ref` and hard-deletes only what the run solely introduced.

## The four parts

### Part 0 — lock the contract (this issue)
Shared types, helpers, marker constants, and interfaces. **No runtime change.**
- `cognee/modules/graph/provenance/`: `constants`, `exceptions`
  (`UnsupportedProvenanceCapability`), `refs` (`make_source_ref`,
  `make_source_run_ref`), `snapshots` (`EdgeIdentity`, `NodeDeleteData`,
  `EdgeDeleteData`), `results` (`ProvenanceDeleteResult`).
- `GraphDBInterface` provenance read primitives (raising defaults, not abstract).
- `GraphVectorStoreInterface` (`delete_by_source_ref`, `delete_by_dataset_id`,
  `rollback_by_pipeline_run_id`) with raising defaults.
- Backend-agnostic capability tests (for Part 1) and test-only fake stores (for
  Part 2).
- Verified write-site and ledger-consumer inventories (`write_site_inventory.md`,
  `ledger_consumer_inventory.md`).
- First backend scope locked: **Ladybug graph + LanceDB vector + SQLite**.

See `part0_contract.md` for the full contract.

### Part 1 — storage primitives (follow-up plan, parallel with Part 2)
Implement the contract on the first backends:
- Stamp `source_refs` / `source_run_refs` / `dataset_ids` / `delete_mode` /
  `provenance_version` onto nodes & edges at every write site (see
  `write_site_inventory.md`), and onto vector payloads.
- Implement the `GraphDBInterface` provenance read primitives on **Ladybug**,
  and the vector-side reads on **LanceDB**.
- Make the Part 0 capability tests pass against the real Ladybug + LanceDB +
  SQLite adapters.

### Part 2 — graph-native delete & rollback for new graphs (follow-up plan, parallel with Part 1)
Implement `GraphVectorStoreInterface` over the Part 1 primitives and re-point
delete/rollback to it **for graphs written under `delete_mode="graph_native"`**,
keeping the ledger path as fallback for older graphs. Validate against the Part 0
fake stores (reference semantics) and then the real backends.

### Part 3 — migrate old graphs and retire the ledger (depends on Parts 1 & 2)
Backfill on-graph provenance for ledger-era graphs (using the migration mapping
in `part0_contract.md`), flip them to `delete_mode="graph_native"`, remove the
ledger consumers listed in `ledger_consumer_inventory.md`, and drop the `nodes`
/ `edges` tables and the legacy `GraphRelationshipLedger`.

## Dependencies

```
Part 0  ──┬──▶ Part 1 ──┐
          └──▶ Part 2 ──┴──▶ Part 3 (also needs the Part 0 migration mapping)
```

Part 1 and Part 2 both build on Part 0 and proceed in parallel. Part 3 needs
both, plus the migration mapping defined in Part 0.

## Acceptance for Part 0

- Helpers, dataclasses, marker constants, and interfaces import cleanly.
- New interface methods raise `UnsupportedProvenanceCapability` by default,
  never silently pass, and are not `@abstractmethod`.
- No live delete, rollback, add, search, or retrieval path calls the new methods.
- Part 1 plan agreed and its capability tests can run against real adapters.
- Part 2 plan agreed and its tests can run against the fake stores.
