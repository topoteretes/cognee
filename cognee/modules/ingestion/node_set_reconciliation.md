# Node-set reconciliation: ownership-contract handoff

Status: **work order 1 incomplete; metadata moves are not implemented**.
Baseline: `c5e5a0b0eea6d546b0a1cedf43020b5a19d6d7c1`, on `dev`.
The selected release scope remains every supported document variant on the
default embedded graph/vector stores. This note records a concrete attribution
blocker, not a narrower release recommendation.

## Reproduction

`test_completed_readd_replaces_node_set_without_reextraction` in
`cognee/tests/integration/tasks/test_add_ingestion_regression.py` executes the
reported `add(alpha) → cognify → add(beta) → cognify` sequence with real embedded
Ladybug/Kuzu and LanceDB storage. It snapshots chunk properties, native source
refs, exact stored vectors, document identity, and processing stamps. LLM and
embedding calls are forbidden during the repeated add/cognify stage; query
embedding is restored separately.

The observed result is still `alpha`: its CHUNKS filter finds the chunk, while
`beta` returns nothing. The relational tags also remain `alpha`. This exact
symptom is a targeted expected failure after all snapshot/sentinel assertions
have passed. The test includes the desired new/old-scope assertions, but does
**not** prove the feature works. Remove the targeted xfail after integration.

The completed-status branch in
`modules/pipelines/operations/run_tasks_data_item.py` returns before
`tasks/ingestion/ingest_data.py:merge_existing_row_metadata` runs. Both the stored
content-hash lookup and the pinned-ID lookup reach this branch. Reconciliation
belongs before that return for permanent add only; resetting cognify stamps
would violate the task.

## Proposed internal discovery result

The owning package is `cognee.modules.ingestion`. The following is a proposed
read-only contract, not an exported API or an executable mutation plan. Do not
instantiate a complete result until every affected artifact has verified owners.

| Field | Contents and invariant |
| --- | --- |
| `dataset_id`, `data_id`, `route` | Authorized selected dataset, canonical processed input, and `cognify_route_for(Data)`; never derive the route from user metadata. |
| `nodes` | Exact ID-keyed `NodeDeleteData` snapshots: stored properties, indexed fields, full source refs, run refs and dataset refs. |
| `edges` | Exact `EdgeIdentity`-keyed `EdgeDeleteData` snapshots, including membership edges and complete co-owner source refs. |
| `owner_contributions` | Per node/edge: owner source-ref key, user tags, protected tags, and the stored evidence proving this contribution. Keep moved refs separate from remaining refs. |
| `owners` | One targeted relational snapshot of all referenced canonical `Data` rows; same dataset, ownership, system metadata and completion stamps verified. No fabricated code-doc rows. |
| `derived_owners` | Stored code symbol/file/document links or DLT source/row/schema links that prove an artifact's contribution belongs to the processed input. A tag match or transitive entity reachability is insufficient. |
| `protected_memberships` | System memberships resolved from their owning constants and strict session pin reads; unavailable pin state is an error. |
| `display_metadata` | Existing `belongs_to_set`, `source_node_set` and serialized `external_metadata` fields to patch, retaining all unrelated fields. |
| `indexed_payloads` | Every affected existing collection/ID, payload schema and stored payload; derive collection names from stored indexed fields. Require complete rows before mutation. |
| `required_capabilities` | Working native heterogeneous field patches and membership-owner deltas, exact ownership reads, and vector-preserving payload replacement on the configured default stores. Inherited no-ops are not capabilities. |

Reuse `data_source_ref_key_selector`, `parse_source_ref_key`, `NodeDeleteData`,
`EdgeDeleteData`, `EdgeIdentity` and the current trusted routing policy. The
existing chunk-ownership walker is a **write-time** collector over extraction
objects, not a stored-owner discovery service. It cannot be replayed here.

## Concrete blocker: existing code-doc extracted relationships

The real Ladybug fixture
`test_memify_code_docs_lose_shared_output_owner_evidence` in
`cognee/tests/unit/tasks/code_docs/test_code_docs_round_trip.py` demonstrates:

1. The graph is already marked as graph-provenance-backed.
2. Stored code-doc chunks from two repositories have distinct, valid
   repo/file/symbol metadata and `describes` links.
3. Both chunks contain the same entities; only one chunk's extraction produced
   a relationship between them.
4. `add_data_points` receives the real memify context shape, with `data_item={}`.
   The chunks, shared entity and relationship are stored with **zero source
   refs**. The enabled edge-evidence buffer captures **zero records**.
5. The private per-chunk relationship production record is absent from stored
   node properties.

Named owning seams:

- `memify_pipelines/cognify_code_docs.py:cognify_code_docs_pipeline` seeds
  `run_memify_tasks` with `data=[{}]`, including the entity-extraction mode.
- `tasks/storage/add_data_points.py` computes `data_id` from `ctx.data_item`;
  without one, native stamping and relational ownership registration are skipped.
- `modules/provenance/edge_evidence/buffer.py:ProvenanceBuffer.capture` also
  requires a resolved data ID. Its sidecar cannot repair this missing evidence.
- `tasks/code_docs/add_code_doc_edges.py` stores the `describes` link, but that
  link proves a chunk's symbol identity, not ownership of relationships between
  shared extracted entities.

The chunk/document/symbol links support deterministic discovery of code-doc
chunks and file/module tag unions. They do **not** prove which chunk emitted a
relationship between two shared entities. Both chunks can contain both endpoints
without both having produced the edge. The production identity is private and
not persisted. Traversing entities would guess ownership and could retain an old
scope or remove another owner's contribution.

**Upgrade boundary requiring maintainer review:** extend the owning code-doc
storage path to stamp extracted artifacts with contributions from the owning
processed code input, using stored symbol/file ownership and exact write-time
chunk production evidence. Do not fabricate a relational document. A deterministic
backfill may recover document/chunk/file links when stored identities prove them;
it cannot recover the missing shared-relationship producer from the fixture above.
Those ambiguous existing relationships need an explicit legacy refusal boundary.
No backfill or release-wide refusal policy is implemented in this patch.

Per the ordered-work blocker rule, dependent mutation work is stopped until a
complete owner-link/upgrade contract is established. A blanket code or DLT
refusal is not a solution and has not been added.

## Route coverage checklist for continuation

| Route | Evidence checked | Remaining exit proof |
| --- | --- | --- |
| Standard text | Real original sequence; document/chunk v1/v2 native refs and vector snapshots. | Shared entities/types/summaries, owner contribution planning, successful move. |
| File/media/custom chunkers | Trusted standard dispatch and chunk ownership hierarchy cover stored document subclasses. | Parameterized dispatch, real file-derived and custom-chunker fixtures. |
| Code file | Stable staged repo identity includes the full Data ID; current file route does not forward tags into extraction. | Stored scope and co-owner discovery, no parser, CODE filters. |
| Repository and code-docs | Stored repo/file/symbol identity and file/module unions checked; missing code-doc shared-relationship refs reproduced. | Reviewed owner-link contract, deterministic upgrade boundaries, shared code output, CODE and CHUNKS moves. |
| Connector document rows | Existing per-item union/PATCH seam identified. | Explicit clear, sibling isolation, structure metadata preservation. |
| Current DLT manifest | Emitter stamps source context on schema/value/time output; source-scoped table/column IDs and `is_row_of`/`is_table_of` links checked. | Complete stored source contributions, row-specific tags, shared values/timestamps and pre-manifest refusal. |
| Bridged QA/trace/distilled | Scope remains selected; no fixture proof in this patch. | Strict pins, protected tags, exact attribution, conflict-without-writes proof. |

Additional impact-sweep seam: the edge-evidence sidecar also depends on the
canonical processed input ID. It is enabled by default but is not complete
ownership evidence for code-doc memify. Readers additionally include
`graph_report_retriever`, visualization/preprocessor compact payloads, and brains
summaries; vector-only or relational-only tagging would leave them inconsistent.
MCP permanent remember still has truthiness gates that lose explicit `[]`;
forwarding remains part of work order 7.

## Milestone exit status

- **1: incomplete.** Ordinary symptom and the code-doc attribution defect are
  reproduced. The proposed result shape and route checklist are recorded.
  Complete representative co-owner discovery and the reviewed upgrade boundary
  are not established.
- **2–8: not started.** No batch contracts, planner, reconciliation integration,
  successful variant moves, compatibility changes, scale/heartbeat measurements,
  or feature documentation have been implemented.

Resume at work order 1. The original feature acceptance remains unsatisfied.

## Verification in this handoff

- `uv run pytest cognee/tests/integration/tasks/test_add_ingestion_regression.py
  -q --tb=short --show-capture=no`: 11 passed, 1 targeted expected failure for
  the original node-set symptom, using real default embedded workers.
- `uv run pytest
  cognee/tests/unit/tasks/code_docs/test_code_docs_round_trip.py
  cognee/tests/unit/modules/pipelines/test_run_tasks_data_item_pinned_id.py
  cognee/tests/unit/api/test_incremental_node_sets.py
  -q --tb=short --show-capture=no`: 11 passed.
- `uv run ruff check .`, changed-path `ruff format --check`, `git diff --check`,
  and `uv run pre-commit run --all-files`: passed. Repository-wide `ruff format
  .` ran; its unrelated connector Markdown formatting was reverted.
- Duplicate-code checking ran with pylint at the base and branch for
  `cognee/tests/integration/tasks` and `cognee/tests/unit/tasks/code_docs`.
  Existing fixture duplication was reported; no reported block includes added
  test lines. The new tests reuse the existing module fixtures and hashed-vector
  utility rather than copying their setup.
- `uv run pytest cognee/tests/unit/ -q --tb=short --show-capture=no` initially
  failed collection because optional dev/evals/Neo4j dependencies were missing.
  After installing those repository extras, the full run reached an assertion
  failure and did not finish before the command timeout (900 seconds).
  A diagnostic run with `-x --timeout=60` completed with 789 passed, 54 skipped,
  and one failure in
  `api/recall/test_recall_search_type_default.py::test_time_scoped_question_routes_to_temporal`:
  `HYBRID_COMPLETION` was selected instead of `TEMPORAL`. The exact test also
  fails on the untouched base in a detached worktree with the same environment.
  This is not a full-suite pass; the rest remains unverified.

No 500-node reconciliation, operation-count, timing, heartbeat, pin-conflict,
cross-store retry, API schema, or forwarding acceptance proof was performed.
No successful-move result or release readiness is claimed. A real Linear key
was not supplied by the task or its linked GitHub issue; none was invented.
