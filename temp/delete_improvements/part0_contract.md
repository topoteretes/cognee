# Part 0 — the provenance contract

This is the authoritative spec for the shared types Part 1 and Part 2 build on.
Everything here ships in this issue as a skeleton with **no runtime behaviour
change**: importing it has no side effects and no live code path calls it.

## Package layout

```
cognee/modules/graph/provenance/
  __init__.py        re-exports the public contract
  constants.py       marker values + property-key names
  exceptions.py      UnsupportedProvenanceCapability
  refs.py            make_source_ref / make_source_run_ref
  snapshots.py       EdgeIdentity / NodeDeleteData / EdgeDeleteData
  results.py         ProvenanceDeleteResult
```

Interfaces:
- `cognee/infrastructure/databases/graph/graph_db_interface.py` — provenance
  read primitives added to `GraphDBInterface`.
- `cognee/infrastructure/databases/unified/graph_vector_store_interface.py` —
  new `GraphVectorStoreInterface`.

`cognee.modules.graph` is a namespace package (no `__init__.py`) and the
provenance modules depend only on the stdlib, so importing them from
`infrastructure` introduces no import cycle.

## Marker constants & property keys (`constants.py`)

| Name | Value | Role |
| --- | --- | --- |
| `PROVENANCE_VERSION` | `1` | On-graph provenance layout version. |
| `DELETE_MODE_GRAPH_NATIVE` | `"graph_native"` | Artifact is deletable without the ledger. |
| `DELETE_MODE_LEDGER` | `"ledger"` | Legacy artifact; provenance still in the ledger. |
| `PROVENANCE_VERSION_KEY` | `"provenance_version"` | Property holding the version int. |
| `DELETE_MODE_KEY` | `"delete_mode"` | Property holding the delete mode. |
| `SOURCE_REFS_KEY` | `"source_refs"` | `List[str]` of source refs. |
| `SOURCE_RUN_REFS_KEY` | `"source_run_refs"` | `List[str]` of source-run refs. |
| `DATASET_IDS_KEY` | `"dataset_ids"` | `List[str]` dataset ids for dataset-scoped delete. |

## Refs (`refs.py`)

- `make_source_ref(dataset_id, data_id) -> str` — opaque, deterministic token
  identifying one ingestion source (a `(dataset_id, data_id)` pair).
- `make_source_run_ref(dataset_id, pipeline_run_id) -> str` — opaque,
  deterministic token identifying one pipeline run.

Both are `uuid5(NAMESPACE_OID, ...)` over a **namespaced** string with distinct
prefixes (`cognee:source-ref:v1:` vs `cognee:source-run-ref:v1:`), so a source
ref and a source-run ref can never collide even if a `data_id` equals a
`pipeline_run_id`. Refs are opaque tokens — callers never parse them; the tuples
are the source of truth.

## Snapshot dataclasses (`snapshots.py`)

Graph-native replacements for the ledger rows that delete/rollback read. Fields
were chosen by walking what `delete_from_graph_and_vector.py` actually consumes
off a `Node`/`Edge` row:

- **`EdgeIdentity`** — `(source_node_id, relationship_name, target_node_id)`.
  This is what triplet ids and `EdgeType` retrieval ids are derived from.
- **`NodeDeleteData`** — `node_id` (the ledger `slug`), `node_type` (→ vector
  collection `f"{type}_{field}"`), `label` (→ NodeSet detag), `indexed_fields`
  (→ which vector collections to clean), `source_refs`, `source_run_refs`.
- **`EdgeDeleteData`** — `identity`, `edge_retrieval_text` (→ `EdgeType` +
  triplet vector ids; falls back to `relationship_name`), `source_refs`,
  `source_run_refs`.

All frozen so they dedup in sets, mirroring the slug-dedup the ledger flow does.

## Result (`results.py`)

`ProvenanceDeleteResult(nodes_deleted, edges_deleted, nodes_detached,
edges_detached)` — returned by every `GraphVectorStoreInterface` op. "deleted"
= hard-deleted (last ref removed); "detached" = survived because another
source ref / run still owns it (only the targeted ref stripped).

## Error (`exceptions.py`)

`UnsupportedProvenanceCapability(capability, backend=None)` subclasses
`NotImplementedError`, so existing `except NotImplementedError` handlers keep
working while new code can catch the narrower type. Every new interface method's
default raises it.

## Interface additions

### `GraphDBInterface` — provenance read primitives (raising defaults)

Not `@abstractmethod` (existing adapters must keep instantiating). Each raises
`UnsupportedProvenanceCapability` until Part 1 implements it.

- `supports_graph_native_provenance() -> bool` (default `False`)
- `get_nodes_delete_data_by_source_ref(source_ref) -> List[NodeDeleteData]`
- `get_edges_delete_data_by_source_ref(source_ref) -> List[EdgeDeleteData]`
- `get_nodes_delete_data_by_dataset_id(dataset_id) -> List[NodeDeleteData]`
- `get_edges_delete_data_by_dataset_id(dataset_id) -> List[EdgeDeleteData]`
- `get_nodes_delete_data_by_source_run_ref(source_run_ref) -> List[NodeDeleteData]`
- `get_edges_delete_data_by_source_run_ref(source_run_ref) -> List[EdgeDeleteData]`

### `GraphVectorStoreInterface` (new)

A plain class (not ABC) so it can be mixed into engines and subclassed by test
fakes. Lives by `UnifiedStoreEngine` because it spans graph + vector.

- `supports_graph_native_delete() -> bool` (default `False`)
- `delete_by_source_ref(source_ref) -> ProvenanceDeleteResult`
- `delete_by_dataset_id(dataset_id) -> ProvenanceDeleteResult`
- `rollback_by_pipeline_run_id(pipeline_run_id, dataset_id) -> ProvenanceDeleteResult`

All three raise `UnsupportedProvenanceCapability` by default.

## Migration mapping (for Part 3)

Each ledger row maps to on-graph provenance for its `slug` (the graph node id):

| Ledger column | On-graph provenance |
| --- | --- |
| `slug` | graph node id / edge identity to stamp |
| `(dataset_id, data_id)` | add `make_source_ref(dataset_id, data_id)` to `source_refs` |
| `(dataset_id, pipeline_run_id)` | add `make_source_run_ref(dataset_id, pipeline_run_id)` to `source_run_refs` |
| `dataset_id` | add to `dataset_ids` |
| — | set `provenance_version = 1`, `delete_mode = "graph_native"` after backfill |

A node/edge with multiple ledger rows accumulates multiple refs. Backfill is
idempotent (refs are deterministic; adding an existing ref is a no-op). After a
graph is fully backfilled and flipped to `graph_native`, its ledger rows and the
legacy `GraphRelationshipLedger` entries are removable.

## First backend scope

Locked to **Ladybug (graph) + LanceDB (vector) + SQLite (relational)** — the
default local stack. Part 1 implements the read primitives on Ladybug/LanceDB;
other graph/vector backends keep raising `UnsupportedProvenanceCapability` (and
fall back to the ledger) until separately implemented.

## Tests shipped in Part 0

- `cognee/tests/unit/modules/graph/provenance/test_provenance_capabilities.py`
  — backend-agnostic capability tests: raising defaults, ref determinism,
  constant/dataclass stability, and a parametrizable adapter harness Part 1
  plugs real backends into.
- `cognee/tests/unit/modules/graph/provenance/test_fake_graph_vector_store.py`
  — `FakeGraphVectorStore` reference semantics (ref-counted delete / detach /
  rollback) that Part 2 develops against.
