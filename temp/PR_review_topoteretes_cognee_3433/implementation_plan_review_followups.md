# Implementation Plan - PR Review Follow-up Fixes

## Goal

Close the remaining PR-review issues that should be fixed before merge, in an
order that keeps the implementation readable and avoids repeated renames.

This plan fixes:

- `"graph_native" vs "graph provenance" naming split`
- `delete_data resolves the graph engine and reads the marker twice`
- `Graph-provenance memory read silently falls back to relational on any error`
- `Real-adapter capability & integration tests are not backend-agnostic`

The scalar-string durability issue is already resolved. The indexed-field,
hybrid-backend, and full-scan issues stay as explicit follow-up work because the
current recommendations say not to change those paths in this PR.

## Current State

- The on-disk graph metadata token is `delete_mode="graph_native"`, but the code
  concept is graph provenance.
- New tests and comments still use `graph_native` in file names, function names,
  and one raw metadata assertion.
- `datasets.delete_data` reads the graph-provenance marker, then
  `delete_data_nodes_and_edges` reads it again on the graph-provenance path.
- `_read_memory_graph_provenance` catches broad import and graph read failures
  and returns `None`, causing relational fallback on graph-provenance graphs.
- The real adapter capability tests directly instantiate `LadybugAdapter`, so
  future adapter implementers do not have a shared contract suite to run.

## Desired State

- `graph_native` appears only as the documented storage token value and in the
  one constant contract test that pins that value.
- Feature, test, and helper names use `graph_provenance`.
- The API delete route does not perform a duplicate marker read on the
  graph-provenance path.
- Memory provenance falls back to relational only when the graph is intentionally
  not marked for graph provenance. Unexpected graph-provenance read failures
  surface.
- Backend-neutral graph-provenance graph-adapter contract tests exist, and
  Ladybug is registered as the first provider.
- A backend-neutral unified graph-provenance contract exists for the
  `UnifiedStoreEngine` delete/rollback path, with Ladybug registered as the
  first graph provider.
- Ladybug/Kuzu-specific storage, delimiter, and checkpoint tests remain separate
  from backend-neutral contract tests.

## Implementation Phases Overview

- **Phase 1 - Normalize Graph-Provenance Naming** ✅ DONE
  - Rename new test files/functions/comments from `graph_native` to
    `graph_provenance`.
  - Keep the stored token unchanged.

- **Phase 2 - Remove the Duplicate Delete Marker Read** ✅ DONE
  - Add one small exported graph-method helper for graph-provenance data-item
    delete.
  - Use it from both the API ledger gate and `delete_data_nodes_and_edges`.

- **Phase 3 - Make Memory Provenance Fail Closed** ✅ DONE
  - Stop swallowing unexpected graph-provenance read errors.
  - Add a unit test proving relational fallback is not used after such failures.

- **Phase 4 - Extract Backend-Neutral Graph-Provenance Contract Tests** ✅ DONE
  - Move common real-adapter provenance behavior into a fixture-parametrized
    graph-adapter contract suite.
  - Add a small fixture-parametrized unified-engine contract for delete/rollback
    behavior.
  - Register Ladybug as the first graph provider.
  - Keep Ladybug-only storage and checkpoint tests in Ladybug-specific files.

- **Phase 5 - Run Focused Validation**
  - Run renamed unit/integration tests and focused ruff checks.

## Implementation Guardrails

- Do not rename the stored metadata value `"graph_native"`.
- Do not rename `GRAPH_DELETE_MODE_GRAPH_PROVENANCE`.
- Do not add a migration or compatibility layer for old draft storage shapes.
- Do not expand this pass into indexing, full-scan optimization, hybrid backend
  atomicity, or malformed-row vector hardening.
- Keep routing cleanup small. Do not change the destructive-path safety check in
  `delete_data_nodes_and_edges`.
- Keep the contract-test extraction lightweight: fixture-parametrized test files
  only, no plugin system, no adapter registry abstraction outside tests.

## Phase 1 - Normalize Graph-Provenance Naming ✅ DONE

### Objective

Remove reader-facing `graph_native` terminology before more tests and helpers are
added.

### Changes

Rename these test files:

- `cognee/tests/unit/modules/graph/test_graph_native_delete_routing.py`
  -> `cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py`
- `cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_native_delete.py`
  -> `cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_provenance_delete.py`
- `cognee/tests/integration/tasks/test_graph_native_delete_default_stack.py`
  -> `cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py`
- `cognee/tests/integration/tasks/test_graph_native_delete_part2.py`
  -> `cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py`

Update references in touched code/tests:

- Rename test functions, helpers, comments, and docstrings that describe the
  feature from `graph_native` / graph-native to `graph_provenance` / graph
  provenance.
- Replace the raw assertion
  `metadata.get("delete_mode") == "graph_native"` with
  `metadata.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_PROVENANCE`.
- Keep `test_graph_provenance_marker_constants_import_cleanly` asserting
  `GRAPH_DELETE_MODE_GRAPH_PROVENANCE == "graph_native"` because that is the
  storage-token contract.
- Update local test command references in the new plan and review docs only when
  those references are touched during this pass.

### Rationale

Doing this first prevents later phases from adding more files and tests that
need the same rename.

### Completion

- Renamed the four graph-provenance delete test files.
- Replaced the remaining raw test assertion for `delete_mode="graph_native"` with
  `GRAPH_DELETE_MODE_KEY` and `GRAPH_DELETE_MODE_GRAPH_PROVENANCE`.
- Updated the rollback recovery comment to point at the renamed default-stack
  integration test.
- Verified `graph_native` remains only in the storage-token constant and its
  contract assertion.

### Demo

Run the renamed routing and unified tests:

```bash
uv run pytest cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py -q
uv run pytest cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_provenance_delete.py -q
```

Run the renamed integration files if Ladybug is available:

```bash
uv run pytest cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py -q
```

## Phase 2 - Remove the Duplicate Delete Marker Read ✅ DONE

### Objective

Make the graph-provenance delete decision happen once on the API graph-provenance
path while preserving the inner safety boundary.

### Changes

Create `cognee/modules/graph/methods/try_delete_data_by_graph_provenance.py`:

- Add one function:
  - `try_delete_data_by_graph_provenance(dataset_id: UUID, data_id: UUID) -> bool`
  - It assumes dataset/data authorization has already happened.
  - It gets the unified engine.
  - It returns `False` when `unified.supports_graph_provenance_delete()` is
    false.
  - It calls `stores_provenance_in_graph(unified.graph)`.
  - It returns `False` when the graph is not marked for graph provenance.
  - It calls `unified.delete_by_source_ref(make_source_ref_key(dataset_id, data_id))`
    and returns `True` when the graph is marked.
  - It allows unexpected marker/read/delete errors to propagate.

Update `cognee/modules/graph/methods/__init__.py`:

- Export `try_delete_data_by_graph_provenance` beside the other graph delete
  helpers.

In `cognee/modules/graph/methods/delete_data_nodes_and_edges.py`:

- Import `try_delete_data_by_graph_provenance` directly from its module, not from
  the package `__init__`.
- Delete the inline unified-engine marker-routing block.
- After authorization, call `try_delete_data_by_graph_provenance(dataset_id, data_id)`.
- Return early when it returns `True`.
- Keep the existing relational-ledger cleanup path unchanged when it returns
  `False`.

In `cognee/api/v1/datasets/datasets.py`:

- Remove the inline `get_graph_engine` and `stores_provenance_in_graph` imports.
- In the existing `data`-found delete branch:
  - First call `has_data_related_nodes(dataset_id, data_id)`.
  - If it returns true, call `delete_data_nodes_and_edges`.
  - If it returns false, call `try_delete_data_by_graph_provenance`.
  - If the helper returns false, call `legacy_delete(data, "soft")`.
- Keep the current `data`-missing custom-graph branch calling
  `delete_data_nodes_and_edges`.
- This avoids duplicate graph-provenance marker reads:
  - graph-provenance graphs have no ledger rows, so the API calls the helper and
    the helper reads the marker once before deleting;
  - old ledger graphs with related nodes go straight to `delete_data_nodes_and_edges`,
    where the inner safety-boundary helper reads the marker once before falling
    through to the ledger cleanup;
  - old graphs with no ledger rows call the helper once, get `False`, then use
    `legacy_delete`.

Update routing tests in
`cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py`:

- Add direct helper tests:
  - marked graph-provenance graph -> calls `unified.delete_by_source_ref` and
    returns `True`
  - unsupported unified engine -> returns `False`
  - supported but unmarked graph -> returns `False`
- Assert `delete_data_nodes_and_edges` uses the shared helper after authorization
  and returns before the relational-ledger path when the helper returns `True`.
- Add a public API route test where `has_data_related_nodes` is false and the
  graph-provenance helper returns true; assert `legacy_delete` is not called.
- Add a public API route test where `has_data_related_nodes` is false and the
  helper returns false; assert `legacy_delete(data, "soft")` is called.

### Rationale

The API route only needs a yes/no operation: "did graph provenance delete this
data item?" A shared graph-method helper keeps that operation in the graph layer,
keeps authorization at the existing callers, and avoids exposing private helpers
or adding status objects/flags to `delete_data_nodes_and_edges`.

### Completion

- Added `try_delete_data_by_graph_provenance` as a small exported graph-method
  helper.
- Replaced the inline graph-provenance routing block in
  `delete_data_nodes_and_edges`.
- Updated `datasets.delete_data` so the API checks ledger rows first and calls
  the helper only on the ledger-free branch.
- Added routing tests for the helper, the inner delete method, and the three API
  branches: ledger rows, ledger-free graph-provenance, and ledger-free legacy.

### Demo

```bash
uv run pytest cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py -q
uv run pytest cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py -q
```

## Phase 3 - Make Memory Provenance Fail Closed ✅ DONE

### Objective

Avoid returning an empty or relational memory view when graph-provenance reads
unexpectedly fail.

### Changes

In `cognee/api/v1/visualize/memory_provenance.py`:

- Keep `_read_memory_graph_provenance(..., dataset_ids=None)` returning `None`
  when no dataset ids are supplied.
- Keep returning `None` when `stores_provenance_in_graph(graph)` returns false.
- Remove the broad `except Exception: return None` around:
  - internal graph-provenance imports
  - `get_unified_engine()`
  - `stores_provenance_in_graph(graph)`
  - `find_node_source_refs_by_dataset`
  - `find_edge_source_refs_by_dataset`
  - `get_graph_data`
- Move the graph-provenance imports to module scope or import them directly
  without a broad fallback inside `_read_memory_graph_provenance`.
- Let unexpected errors from those operations propagate.
- Keep relational fallback in `get_memory_provenance_graph` only for the
  explicit `memory is None` result.

Update `cognee/tests/unit/api/test_memory_provenance_graph_provenance.py`:

- Add a test where `stores_provenance_in_graph` returns false and relational
  fallback remains allowed by the caller.
- Add a test where graph provenance is marked but `graph.get_graph_data` raises;
  assert `get_memory_provenance_graph(include_memory=True)` raises and does not
  call `_read_memory_relational`.

### Rationale

`stores_provenance_in_graph` already treats unexpected marker failures as
fail-closed. The memory read path should follow the same rule: marked
graph-provenance graphs have no relational ledger to fall back to.

### Completion

- Removed broad graph-provenance import/read fallbacks from
  `_read_memory_graph_provenance`.
- Kept the explicit unmarked-graph case returning `None`, so relational fallback
  still works for ledger-backed graphs.
- Added tests proving marked graph read errors propagate and the public caller
  does not call relational fallback after such errors.

### Demo

```bash
uv run pytest cognee/tests/unit/api/test_memory_provenance_graph_provenance.py -q
```

## Phase 4 - Extract Backend-Neutral Graph-Provenance Contract Tests ✅ DONE

### Objective

Give future graph-provenance adapter implementers a shared ground-truth test
suite. A colleague should be able to add their provider to these fixtures and
know whether the graph adapter and unified delete/rollback path satisfy the
graph-provenance contract. Keep storage-shape assertions out of the shared
contract.

### Changes

Create
`cognee/tests/integration/infrastructure/graph/test_graph_provenance_adapter_contract.py`:

- Add a lightweight provider fixture:
  - `graph_provenance_adapter(request, tmp_path)`
  - The first provider is `ladybug`.
  - The provider constructs `LadybugAdapter(str(tmp_path / "graph_db"))`.
  - The provider closes the adapter after each test.
  - If Ladybug is unavailable, skip only the Ladybug provider.
- Test only backend-neutral graph provenance behavior:
  - `get_graph_metadata` starts empty.
  - `set_graph_metadata` round-trips and merge-updates marker keys.
  - Graph metadata does not make the user graph non-empty.
  - node attach/remove source refs materializes and recalculates
    `source_ref_keys`, `source_dataset_ids`, `source_run_ids`, and
    `source_run_refs`.
  - edge attach/remove source refs follows the same derived-field rules.
  - attach without `pipeline_run_id` is not rollbackable by run.
  - node and edge lookup by exact valid source ref returns only matching owners.
  - dataset and pipeline-run source-ref lookups return the expected owners.
  - delete snapshots for nodes and edges include properties, indexed fields,
    edge text fallback, and missing-artifact omission.
  - `delete_edge_triples` deletes only the requested relationship and preserves
    endpoint nodes.
  - `belongs_to_set` detag behavior works for scoped, unscoped, and empty tag
    calls.
  - rewriting existing nodes and edges preserves provenance while updating
    content.
  - `get_edges_created_since` returns edges after the checkpoint, oldest first,
    respects `limit`, includes endpoint node metadata, and excludes edges at or
    before the checkpoint.
  - folded `add_nodes` and `add_edges` stamp provenance in the write operation.
  - folded `add_edges` preserves multiple owners.
  - folded writes with no `source_ref_key` stamp no provenance.
  - Model A run-ref behavior is preserved when a new run re-touches an existing
    node or edge source ref.
  - concurrent folded node writes for different source refs preserve both keys.
- Use only public adapter contract methods and provenance dataclasses/helpers.
- Do not call `adapter.query`, `_write_node_provenance`,
  `_write_edge_provenance`, `_raw_node_source_ref_keys`, or
  `_raw_edge_source_ref_keys` in this contract file.
- Do not assert raw Ladybug storage strings, Kuzu schema, Cypher behavior,
  private helper behavior, or checkpoint/reopen behavior in this contract file.
- Split mixed tests instead of moving them wholesale:
  - keep public folded-write assertions in the contract suite;
  - move raw `source_ref_keys == f"|{key}|"` assertions to the Ladybug storage
    suite;
  - keep delimiter/lookalike token stress in the Ladybug storage suite because
    it uses raw storage tokens that are not valid public source refs.

Create
`cognee/tests/integration/infrastructure/graph/test_graph_provenance_unified_contract.py`:

- Add a lightweight provider fixture:
  - `graph_provenance_unified_engine(request, tmp_path)`
  - The first provider is `ladybug+recording_vector`.
  - The provider constructs `LadybugAdapter(str(tmp_path / "graph_db"))`.
  - The provider wraps it in `UnifiedStoreEngine` with a small
    `RecordingVectorEngine`.
  - The provider closes the graph adapter after each test.
  - If Ladybug is unavailable, skip only the Ladybug provider.
- Test backend-neutral unified graph-provenance behavior:
  - `delete_by_source_ref` deletes unowned nodes and edges, detaches the removed
    ref from shared artifacts, and asks the vector engine to delete unowned node
    and triplet vectors.
  - `delete_by_dataset_id` removes only the dataset's refs and preserves
    cross-dataset artifacts.
  - `rollback_by_pipeline_run_id` removes only refs introduced by that run and
    preserves artifacts that a prior run still owns.
  - a vector delete failure leaves graph refs discoverable, and retry converges.
  - deleting an unowned `NodeSet` triggers graph/vector `belongs_to_set` detag
    cleanup.
- Use the recording vector only to prove the unified planner calls the vector
  contract correctly. Real vector adapter CRUD/detag behavior remains covered by
  the vector adapter capability tests.
- Do not duplicate the full default-stack `cognee.add -> cognee.cognify ->
  delete_data` product tests in this contract file.

Reshape and move
`cognee/tests/unit/infrastructure/databases/graph/test_ladybug_provenance_capabilities.py`
to
`cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_storage.py`:

- Keep Ladybug-specific tests there:
  - raw scalar storage shape for folded node provenance
  - raw scalar storage shape for folded edge provenance
  - raw scalar storage shape for multiple edge owners
  - raw delimiter lookalike token filtering
  - long raw delimited source-ref round trip seeded through Ladybug storage
    helpers
- Move duplicated backend-neutral contract cases into the new contract file.
- Keep the remaining storage-specific tests under `integration/` because they
  use the real Ladybug/Kuzu adapter.
- Keep the storage file small; do not create a second behavioral contract there.

Keep
`cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_checkpoint.py`
Ladybug-specific.

Keep task-level integration tests default-stack specific:

- `cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py`
- `cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py`

Those tests prove the product default stack, not every adapter implementation.
The new contract files are the ground truth for future adapter work.

Adapter readiness means all of these pass for the target implementation:

- the graph-adapter contract with the target graph provider registered;
- the unified graph-provenance contract with the target graph provider
  registered;
- the target vector adapter's provenance delete/detag capability tests;
- the product-flow integration tests for any supported default stack.

### Rationale

Future backends need one place to prove their graph adapter implements graph
provenance and one small real-adapter path that proves those methods work
through `UnifiedStoreEngine`. The contract tests should describe behavior, not
Ladybug storage details. Storage quirks stay in Ladybug files so the shared
suite does not become adapter-shaped.

### Completion

- Added `test_graph_provenance_adapter_contract.py` as the shared public graph
  adapter contract, with Ladybug registered as the first provider.
- Added `test_graph_provenance_unified_contract.py` as the shared
  `UnifiedStoreEngine` graph-provenance delete/rollback contract, using a real
  graph adapter and recording vector engine.
- Moved Ladybug/Kuzu scalar-string storage checks into
  `test_ladybug_provenance_storage.py`.
- Removed the old `test_ladybug_provenance_capabilities.py` file so the real
  adapter tests no longer mix backend-neutral contract behavior with Ladybug
  storage details.

### Demo

Run the new contract suite:

```bash
uv run pytest cognee/tests/integration/infrastructure/graph/test_graph_provenance_adapter_contract.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_graph_provenance_unified_contract.py -q
```

Run Ladybug-specific storage and checkpoint suites:

```bash
uv run pytest cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_storage.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_checkpoint.py -q
```

Run default-stack integration:

```bash
uv run pytest cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py -q
```

## Phase 5 - Run Focused Validation

### Objective

Prove the review-followup fixes work without broad formatting or unrelated test
churn.

### Commands

```bash
uv run pytest cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py -q
uv run pytest cognee/tests/unit/api/test_memory_provenance_graph_provenance.py -q
uv run pytest cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_provenance_delete.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_graph_provenance_adapter_contract.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_graph_provenance_unified_contract.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_storage.py -q
uv run pytest cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_checkpoint.py -q
uv run pytest cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py -q
```

Run ruff on touched files:

```bash
uv run ruff check cognee/api/v1/datasets/datasets.py cognee/modules/graph/methods/delete_data_nodes_and_edges.py cognee/api/v1/visualize/memory_provenance.py cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py cognee/tests/unit/api/test_memory_provenance_graph_provenance.py cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_provenance_delete.py cognee/tests/integration/infrastructure/graph/test_graph_provenance_adapter_contract.py cognee/tests/integration/infrastructure/graph/test_graph_provenance_unified_contract.py cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_storage.py cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_checkpoint.py cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py
uv run ruff format --check cognee/api/v1/datasets/datasets.py cognee/modules/graph/methods/delete_data_nodes_and_edges.py cognee/api/v1/visualize/memory_provenance.py cognee/tests/unit/modules/graph/test_graph_provenance_delete_routing.py cognee/tests/unit/api/test_memory_provenance_graph_provenance.py cognee/tests/unit/infrastructure/databases/provenance/test_unified_graph_provenance_delete.py cognee/tests/integration/infrastructure/graph/test_graph_provenance_adapter_contract.py cognee/tests/integration/infrastructure/graph/test_graph_provenance_unified_contract.py cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_storage.py cognee/tests/integration/infrastructure/graph/test_ladybug_provenance_checkpoint.py cognee/tests/integration/tasks/test_graph_provenance_delete_default_stack.py cognee/tests/integration/tasks/test_graph_provenance_delete_part2.py
```

## Non-Goals

- Do not change the on-disk `delete_mode="graph_native"` token.
- Do not rename `GRAPH_DELETE_MODE_GRAPH_PROVENANCE`.
- Do not implement graph-provenance support for Neo4j, Postgres, Neptune, or
  hybrid backends in this PR.
- Do not add provenance indexing or targeted EdgeType cleanup in this PR.
- Do not harden malformed/out-of-band vector cleanup in this PR.
- Do not make hybrid graph-provenance writes atomic in this PR.

## Merge Gates

- `rg "graph_native_delete|graph-native|Graph-native" cognee/tests cognee`
  returns no reader-facing feature names, except historical comments that are
  intentionally not touched in this PR.
- Raw `"graph_native"` remains only in the provenance constant contract and
  storage-token documentation.
- Graph-provenance data delete reads the marker once on the ledger-free API
  route.
- Graph-provenance memory read failures surface instead of falling back to the
  relational ledger.
- A future graph adapter can opt into the graph-adapter and unified contracts by
  adding provider entries.
- A future adapter combo has a clear readiness checklist: graph contract,
  unified graph-provenance contract, relevant vector capability tests, and any
  product-flow integration tests for supported default stacks.
- Ladybug storage and checkpoint regressions still pass.
