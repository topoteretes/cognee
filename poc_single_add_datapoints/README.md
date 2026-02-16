# POC Cognify

This folder contains a proof-of-concept (POC) pipeline and demos that modify the standard
`cognee.cognify()` flow to avoid double `add_data_points()` calls, instead creating graph
entities and relations attached to document chunks.

## POC vs. Default Cognify Pipeline

Both pipelines use the same task order, but they differ in how graph entities are persisted.

- **Task order (both):** `classify_documents` -> `extract_chunks_from_documents` ->
  `extract_graph_from_data` -> `summarize_text` -> `add_data_points`
- **Default cognify:** `extract_graph_from_data` calls `expand_with_nodes_and_edges`, then
  writes graph nodes via `add_data_points` and graph edges via graph DB + indexing before the
  final `add_data_points` task runs.
- **POC:** `extract_graph_from_data` is called with `use_single_add_datapoints_poc=True`,
  which switches to `poc_expand_with_nodes_and_edges`. This attaches `GraphEntity` objects
  (with `relations`) to each chunk’s `contains` list and **does not write to storage** in
  that step. A single `add_data_points` call later persists both chunks and graph entities.
- **Ontology and edge de-duplication:** Both paths keep ontology resolution and existing-edge
  checks, but the POC defers persistence to the final step.
- **POC toggle parameter:** The `use_single_add_datapoints_poc` flag is passed
  through the task builder into `extract_graph_from_data` to enable the single-add path.

## What Changed (vs. default cognify pipeline)

- **Custom pipeline entrypoint:** `poc_single_add_datapoints_pipeline.py` defines `poc_cognify()`
  and wires `extract_graph_from_data` from this folder.
- **Graph extraction no longer writes to DB (in POC mode):** `poc_extract_graph_from_data.py`
  mirrors the default task but, when `use_single_add_datapoints_poc=True`, **stops before DB
  writes**. It extracts chunk graphs and enriches `DocumentChunk.contains` without calling
  `add_data_points()` or graph-edge upserts.
- **Single `add_data_points()` call:** The POC pipeline performs `add_data_points` once
  (as a task after summarization), preventing the “extract graph” stage from inserting
  data points a second time.
- **Chunk-level graph entities with relations:** `poc_expand_with_nodes_and_edges.py`
  builds `GraphEntity`/`GraphEntityType` objects (with `relations` populated) and links
  them into each chunk’s `contains`. The POC keeps global node maps for cross-chunk
  deduplication, while computing per-chunk diffs to attach only newly created nodes
  to each chunk.
- **Ontology-aware behavior preserved:** Ontology resolver selection and existing-edge
  de-duplication are retained from the standard pipeline via `get_ontology_*` helpers
  and `retrieve_existing_edges()`.

## Line-Level Differences (Current Code)

This section lists the exact line ranges that differ between the two file pairings.

- `cognee/api/v1/cognify/cognify.py` vs `poc_single_add_datapoints/poc_single_add_datapoints_pipeline.py`
  - Import source for `extract_graph_from_data`: line 26 vs line 26.
  - Entry function name: `cognify` line 41 vs `poc_cognify` line 41.
  - POC-only parameter `use_single_add_datapoints_poc`: present at line 56 (POC only).
  - Passing the POC flag into task builder: line 228 (POC only).
  - `get_default_tasks` signature includes POC flag: line 258 (POC only).
  - `Task(extract_graph_from_data, ...)` includes POC flag: line 298 (POC only).

- `cognee/tasks/graph/extract_graph_from_data.py` vs `poc_single_add_datapoints/poc_extract_graph_from_data.py`
  - POC import: `poc_expand_with_nodes_and_edges` at line 31 (POC only).
  - `integrate_chunk_graphs` signature adds `use_single_add_datapoints_poc`: line 66 (POC only).
  - POC branching in graph integration: lines 117–151 (POC only); default writes graph nodes/edges
    unconditionally at lines 115–145.
  - `extract_graph_from_data` signature adds `use_single_add_datapoints_poc`: line 162 (POC only).
  - Call into `integrate_chunk_graphs` passes POC flag: line 222 (POC only).

## POC Pipeline Flow

1. `classify_documents`
2. `extract_chunks_from_documents`
3. `poc_extract_graph_from_data` (extract + attach graph entities to chunks, no DB writes)
4. `summarize_text`
5. `add_data_points` (single insertion point for both chunk and graph entities)

## Demo Scripts

- `simple_document_qa_demo.py` — Runs the POC flow on `data/alice_in_wonderland.txt`.
- `ontology_demo_example.py` — Demonstrates ontology-driven graph extraction using
  `ontology_input_example/basic_ontology.owl`, comparing standard vs. POC output.

## Files Added for This POC

- `poc_single_add_datapoints_pipeline.py` — POC pipeline entrypoint and task wiring
  (`poc_cognify`, `get_default_tasks`, `get_temporal_tasks`).
- `poc_extract_graph_from_data.py` — POC version of `extract_graph_from_data` that can
  avoid DB writes and attach graph entities to chunks when the POC flag is enabled.
- `poc_expand_with_nodes_and_edges.py` (in `cognee/modules/graph/utils/`) — Builds
  `GraphEntity`/`GraphEntityType` with `relations` and links them into chunk `contains`.

## Inputs & Outputs

- `data/` — Input documents used by demos.
- `ontology_input_example/` — Sample ontology files for ontology-based extraction.
- `results/` — Example HTML visualizations produced by the demos (generated artifacts).
