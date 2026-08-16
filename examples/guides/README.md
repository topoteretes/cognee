# Cognee Guides

Short, focused how-to scripts — one concept each. Most use the v1.0 memory
API (`remember`, `recall`, `forget`, `improve`); the lower-level `add`,
`cognify`, `search`, and `prune` calls appear only where the guide is about
pipeline internals.

For the "advanced companion" versions of several guides (real documents,
longer sessions, PASSED/FAILED verdicts) see
[`../advanced_guides/`](../advanced_guides/).

## Getting started

| Script | What you'll learn |
|---|---|
| [`simple_cognee_example.py`](simple_cognee_example.py) | Canonical `remember → recall` pipeline (start here) |
| [`recall_core.py`](recall_core.py) | `recall` semantics and parameters |
| [`improve_quickstart.py`](improve_quickstart.py) | The `improve` step of the memory API |
| [`agent_memory_quickstart.py`](agent_memory_quickstart.py) | Wrap an LLM agent with cognee memory (two agents, two memory types) |
| [`start_local_ui_frontend_example.py`](start_local_ui_frontend_example.py) | Launch the cognee UI alongside the API server via `cognee.start_ui()` |

## Session memory & tuning

| Script | What you'll learn |
|---|---|
| [`sessions.py`](sessions.py) | Keep two `session_id` conversations apart in recall |
| [`session_distillation.py`](session_distillation.py) | Distill a session's stated preferences into learned guidance |
| [`memory_provenance.py`](memory_provenance.py) | Trace a fact back to the file it came from (`get_memory_provenance_graph()`) |
| [`importance_weight.py`](importance_weight.py) | Boost specific nodes in retrieval ranking |
| [`global_context_index.py`](global_context_index.py) | Dataset-level summary prepended to retrieval context |
| [`references_example.py`](references_example.py) | Lightweight evidence references appended to recall answers |

## Customizing extraction

| Script | What you'll learn |
|---|---|
| [`custom_prompts.py`](custom_prompts.py) | Override the LLM prompts used in the pipeline |
| [`custom_data_models.py`](custom_data_models.py) | Custom `DataPoint` subclasses |
| [`custom_graph_model.py`](custom_graph_model.py) | Custom graph model used by extraction |
| [`custom_tasks_and_pipelines.py`](custom_tasks_and_pipelines.py) | Author your own tasks and compose them |
| [`ontology_quickstart.py`](ontology_quickstart.py) | Supply a hand-written ontology (see [`ontology_input_example/`](ontology_input_example/)) |
| [`consolidate_entity_descriptions_example.py`](consolidate_entity_descriptions_example.py) | Merge near-duplicate entity descriptions |
| [`entity_deduplication.py`](entity_deduplication.py) | Deduplicate entities that share a surface form |

## Retrieval features

| Script | What you'll learn |
|---|---|
| [`temporal_recall.py`](temporal_recall.py) | Time-bounded recall queries (`temporal_cognify=True`) |
| [`nodeset_grouping_example.py`](nodeset_grouping_example.py) | Group nodes into named sets for filtered retrieval |
| [`schema_inventory.py`](schema_inventory.py) | Schema and entity inventory in the rendered graph |
| [`semantic_memory_map.py`](semantic_memory_map.py) | Lay out the knowledge graph by meaning (2-D projection) |
| [`code_graph_example.py`](code_graph_example.py) | Build a code knowledge graph with enola and query it with `SearchType.CODE` |

## Backends & storage

| Script | What you'll learn |
|---|---|
| [`ladybug_example.py`](ladybug_example.py) | Ladybug (default) graph backend |
| [`neo4j_example.py`](neo4j_example.py) | Neo4j graph backend |
| [`neptune_analytics_example.py`](neptune_analytics_example.py) | AWS Neptune Analytics graph backend |
| [`pgvector_example.py`](pgvector_example.py) | Postgres + pgvector as vector (and relational) store |
| [`s3_storage.py`](s3_storage.py) | Store data and metadata on S3 |

## Ingestion

| Script | What you'll learn |
|---|---|
| [`web_url_content_ingestion_example.py`](web_url_content_ingestion_example.py) | Ingest a web page with a custom loader and CSS rules |
| [`video_processing_example.py`](video_processing_example.py) | Ingest video content |
| [`multimedia_audio_image_processing_example.py`](multimedia_audio_image_processing_example.py) | Audio + image ingestion (data in [`multimedia_audio_image_processing_example_data/`](multimedia_audio_image_processing_example_data/)) |
| [`image_ocr_extraction_check.py`](image_ocr_extraction_check.py) | Inspect what an image becomes: vision transcription plus OCR (`cognee[rapidocr]`) |

## Observability & plumbing

| Script | What you'll learn |
|---|---|
| [`langfuse_telemetry.py`](langfuse_telemetry.py) | Send cognee's OpenTelemetry traces to Langfuse natively |
| [`low_level_llm.py`](low_level_llm.py) | Direct access to the LLM gateway (skip pipelines) |
| [`graph_visualization.py`](graph_visualization.py) | Render the knowledge graph to interactive HTML (`visualize_graph`) |

## Running a guide

```bash
uv run python examples/guides/<guide_name>.py
```

Requires `LLM_API_KEY` in `.env` (copy `.env.template`). Backend-specific
guides need their extra, e.g. `uv pip install "cognee[postgres]"` for
[`pgvector_example.py`](pgvector_example.py); see each script's header for
its exact prerequisites.
