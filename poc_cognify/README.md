# POC Cognify

This folder contains a proof-of-concept (POC) pipeline and demos that modify the standard
`cognee.cognify()` flow to avoid double `add_data_points()` calls, instead creating graph
entities and relations attached to document chunks.

## What Changed (vs. default cognify pipeline)

- **Custom pipeline entrypoint:** `poc_cognify.py` defines `cognify_single_add_datapoints()`
  using `run_custom_pipeline()` with a dedicated pipeline name (`cognify_single_add_datapoints`).
- **Graph extraction no longer writes to DB:** `poc_extract_graph_from_data.py` mirrors
  `tasks.graph.extract_graph_from_data`, but **stops before DB writes**. It extracts
  chunk graphs and enriches `DocumentChunk.contains` without calling `add_data_points()`
  or graph-edge upserts.
- **Single `add_data_points()` call:** The POC pipeline performs `add_data_points` once
  (as a task after summarization), preventing the “extract graph” stage from inserting
  data points a second time.
- **Chunk-level graph entities with relations:** `poc_expand_with_nodes_and_edges.py`
  builds `GraphEntity`/`GraphEntityType` objects (with `relations` populated) and links
  them into each chunk’s `contains`. Maps are reset per chunk to keep additions scoped.
- **Ontology-aware behavior preserved:** Ontology resolver selection and existing-edge
  de-duplication are retained from the standard pipeline via `get_ontology_*` helpers
  and `retrieve_existing_edges()`.
- Inside `poc_cognify.py`, `poc_extract_graph_from_data.py` and `poc_expand_with_nodes_and_edges.py`
  are regions that group code that is the same as it is in cognify, or contains minor changes like
  moving default_tasks directly into pipeline

## POC Pipeline Flow

1. `classify_documents`
2. `extract_chunks_from_documents`
3. `poc_extract_graph_from_data` (extract + attach graph entities to chunks, no DB writes)
4. `summarize_text`
5. `add_data_points` (single insertion point for both chunk and graph entities)

## Demo Scripts

- `simple_example.py` — Minimal text example; runs both standard and POC flows and writes
  HTML graph outputs to `results/`.
- `simple_document_qa_demo.py` — Runs the POC flow on `data/alice_in_wonderland.txt`.
- `ontology_demo_example.py` — Demonstrates ontology-driven graph extraction using
  `ontology_input_example/basic_ontology.owl`, comparing standard vs. POC output.

## Inputs & Outputs

- `data/` — Input documents used by demos.
- `ontology_input_example/` — Sample ontology files for ontology-based extraction.
- `results/` — Example HTML visualizations produced by the demos (generated artifacts).
