# Cognee Examples

This directory contains 60+ runnable example scripts that demonstrate cognee's
features end-to-end. They double as the smoke-test corpus that the team uses
to verify behaviour across the SDK.

> **New here?** Start with [`demos/simple_cognee_example.py`](demos/simple_cognee_example.py) (the canonical `add → cognify → search` flow), then follow the quickstart map below.

## 🚀 Quickstart map (5 examples to start with)

| Example | What you'll learn |
|---|---|
| [`demos/simple_cognee_example.py`](demos/simple_cognee_example.py) | Canonical `add → cognify → search` pipeline |
| [`demos/remember_recall_improve_example.py`](demos/remember_recall_improve_example.py) | The V2 memory API (`remember`, `recall`, `improve`, `forget`) |
| [`guides/agent_memory_quickstart.py`](guides/agent_memory_quickstart.py) | Wrap an LLM agent with cognee memory |
| [`guides/graph_visualization.py`](guides/graph_visualization.py) | Render the resulting knowledge graph |
| [`demos/start_local_ui_frontend_example.py`](demos/start_local_ui_frontend_example.py) | Launch the cognee UI alongside the API server |

## 📁 Top-level layout

| Folder | Purpose | Count |
|---|---|---|
| [`configurations/`](configurations/) | Database & permissions configuration recipes | 8 |
| [`custom_pipelines/`](custom_pipelines/) | Build your own pipeline / extend `cognify` | 7 |
| [`database_examples/`](database_examples/) | Smoke tests per supported backend | 5 |
| [`demos/`](demos/) | Feature demos — broadest coverage | 21 |
| [`guides/`](guides/) | Short focused how-to guides | 13 |
| [`pocs/`](pocs/) | Research-grade proofs of concept (entity disambiguation, canonicalization, prefetch) | 7 |

## 🔧 `configurations/` — backend & permissions setup

### Database configuration
| Script | Demonstrates |
|---|---|
| [`database_examples/ladybug_graph_database_configuration.py`](configurations/database_examples/ladybug_graph_database_configuration.py) | Ladybug (default) graph backend |
| [`database_examples/neo4j_graph_database_configuration.py`](configurations/database_examples/neo4j_graph_database_configuration.py) | Neo4j graph backend |
| [`database_examples/neptune_analytics_aws_database_configuration.py`](configurations/database_examples/neptune_analytics_aws_database_configuration.py) | AWS Neptune Analytics graph backend |
| [`database_examples/pgvector_postgres_vector_database_configuration.py`](configurations/database_examples/pgvector_postgres_vector_database_configuration.py) | Postgres + pgvector hybrid (vector + relational in one) |

### Permissions / multi-tenancy (set `ENABLE_BACKEND_ACCESS_CONTROL=True`)
| Script | Demonstrates |
|---|---|
| [`permissions_example/tenant_role_setup_example.py`](configurations/permissions_example/tenant_role_setup_example.py) | Create tenants and assign roles |
| [`permissions_example/tenant_role_constraints_example.py`](configurations/permissions_example/tenant_role_constraints_example.py) | Constrain what a role can do |
| [`permissions_example/user_permissions_and_access_control_example.py`](configurations/permissions_example/user_permissions_and_access_control_example.py) | Per-user dataset access control |
| [`permissions_example/data_access_control_example.py`](configurations/permissions_example/data_access_control_example.py) | Filter retrieval by ACL |

## 🔄 `custom_pipelines/` — extend cognify

| Script | Demonstrates |
|---|---|
| [`custom_cognify_pipeline_example.py`](custom_pipelines/custom_cognify_pipeline_example.py) | Replace the default `cognify` task list with your own |
| [`memify_coding_agent_rule_extraction_example.py`](custom_pipelines/memify_coding_agent_rule_extraction_example.py) | Distill coding-agent traces into reusable rules |
| [`relational_database_to_knowledge_graph_migration_example.py`](custom_pipelines/relational_database_to_knowledge_graph_migration_example.py) | Lift a SQL schema + data into a knowledge graph |
| [`agentic_reasoning_procurement_example.py`](custom_pipelines/agentic_reasoning_procurement_example.py) | Multi-step reasoning over a procurement dataset |
| [`dynamic_steps_resume_analysis_hr_example.py`](custom_pipelines/dynamic_steps_resume_analysis_hr_example.py) | Pipeline that branches based on resume content |
| [`organizational_hierarchy/organizational_hierarchy_pipeline_example.py`](custom_pipelines/organizational_hierarchy/organizational_hierarchy_pipeline_example.py) | Org-chart ingestion (high-level API) |
| [`organizational_hierarchy/organizational_hierarchy_pipeline_low_level_example.py`](custom_pipelines/organizational_hierarchy/organizational_hierarchy_pipeline_low_level_example.py) | Same dataset via the low-level Task API |

## 🗄️ `database_examples/` — smoke tests per backend

| Script | Demonstrates |
|---|---|
| [`ladybug_example.py`](database_examples/ladybug_example.py) | Ladybug (default) — graph |
| [`neo4j_example.py`](database_examples/neo4j_example.py) | Neo4j — graph |
| [`neptune_analytics_example.py`](database_examples/neptune_analytics_example.py) | Neptune Analytics — graph |
| [`chromadb_example.py`](database_examples/chromadb_example.py) | ChromaDB — vector |
| [`pgvector_example.py`](database_examples/pgvector_example.py) | pgvector — vector + relational |

## 🎯 `demos/` — feature breadth

| Script | Demonstrates |
|---|---|
| [`simple_cognee_example.py`](demos/simple_cognee_example.py) | Canonical pipeline (start here) |
| [`comprehensive_example/cognee_comprehensive_example.py`](demos/comprehensive_example/cognee_comprehensive_example.py) | End-to-end with most features stitched together |
| [`remember_recall_improve_example.py`](demos/remember_recall_improve_example.py) | V2 memory API (`remember`, `recall`, `improve`, `forget`) |
| [`conversation_session_persistence_example.py`](demos/conversation_session_persistence_example.py) | Session memory persisted across runs |
| [`session_feedback_example.py`](demos/session_feedback_example.py) | Capturing thumbs-up/down feedback on retrieval |
| [`session_feedback_lifecycle_demo/backend/app.py`](demos/session_feedback_lifecycle_demo/backend/app.py) | Full feedback-loop backend (FastAPI + cognee) |
| [`feedback_score_shifting_example.py`](demos/feedback_score_shifting_example.py) | How feedback nudges retrieval scores |
| [`custom_graph_model_entity_schema_definition.py`](demos/custom_graph_model_entity_schema_definition.py) | Define your own entity schema for graph extraction |
| [`custom_pipeline_single_object_example.py`](demos/custom_pipeline_single_object_example.py) | Run a custom pipeline on a single object |
| [`dynamic_multiple_weighted_edges_example.py`](demos/dynamic_multiple_weighted_edges_example.py) | Many-to-many edges with per-edge weights |
| [`nodeset_grouping_example.py`](demos/nodeset_grouping_example.py) | Group nodes into named sets for filtered retrieval |
| [`temporal_awareness_example/temporal_awareness_example.py`](demos/temporal_awareness_example/temporal_awareness_example.py) | Time-aware retrieval (`Event` model) |
| [`ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py`](demos/ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py) | Ontology as a constraining vocabulary for extraction |
| [`web_url_content_ingestion_example.py`](demos/web_url_content_ingestion_example.py) | Crawl a URL and cognify the content |
| [`dlt_ingestion_example.py`](demos/dlt_ingestion_example.py) | Ingest via [dlt](https://dlthub.com/) sources |
| [`multimedia_processing/multimedia_audio_image_processing_example.py`](demos/multimedia_processing/multimedia_audio_image_processing_example.py) | Audio + image ingestion |
| [`simple_document_qa/simple_document_qa_demo.py`](demos/simple_document_qa/simple_document_qa_demo.py) | Q&A over a single document |
| [`simple_relational_database_migration_example/simple_relational_database_migration_example.py`](demos/simple_relational_database_migration_example/simple_relational_database_migration_example.py) | SQL → graph (small schema) |
| [`complex_relational_database_migration_example/complex_relational_database_migration_example.py`](demos/complex_relational_database_migration_example/complex_relational_database_migration_example.py) | SQL → graph (richer schema) |
| [`pipeline_api_proposal.py`](demos/pipeline_api_proposal.py) | Proposal-style API exploration |
| [`start_local_ui_frontend_example.py`](demos/start_local_ui_frontend_example.py) | Spin up cognee UI + backend |

## 📘 `guides/` — focused how-tos

| Script | Demonstrates |
|---|---|
| [`agent_memory_quickstart.py`](guides/agent_memory_quickstart.py) | Wrap an LLM agent with cognee memory |
| [`improve_quickstart.py`](guides/improve_quickstart.py) | The `improve` step in the V2 API |
| [`recall_core.py`](guides/recall_core.py) | `recall` semantics and parameters |
| [`temporal_recall.py`](guides/temporal_recall.py) | Time-bounded recall queries |
| [`ontology_quickstart.py`](guides/ontology_quickstart.py) | Supply a hand-written ontology |
| [`custom_data_models.py`](guides/custom_data_models.py) | Custom `DataPoint` subclasses |
| [`custom_graph_model.py`](guides/custom_graph_model.py) | Custom graph model used by extraction |
| [`custom_prompts.py`](guides/custom_prompts.py) | Override the LLM prompts used in the pipeline |
| [`custom_tasks_and_pipelines.py`](guides/custom_tasks_and_pipelines.py) | Author your own tasks and compose them |
| [`importance_weight.py`](guides/importance_weight.py) | Boost specific nodes in retrieval ranking |
| [`graph_visualization.py`](guides/graph_visualization.py) | Render the resulting knowledge graph |
| [`low_level_llm.py`](guides/low_level_llm.py) | Direct access to the LLM gateway (skip pipelines) |
| [`s3_storage.py`](guides/s3_storage.py) | Store data and metadata on S3 |
| [`consolidate_entity_descriptions_example.py`](guides/consolidate_entity_descriptions_example.py) | Merge near-duplicate entity descriptions |

## 🧪 `pocs/` — research / proofs of concept

These are exploratory scripts; conventions evolve faster here than in `guides/` or `demos/`.

| Script | Demonstrates |
|---|---|
| [`disambiguation/disambiguate_entities.py`](pocs/disambiguation/disambiguate_entities.py) | Entity-disambiguation primitive |
| [`disambiguation/disambiguate_entities_example.py`](pocs/disambiguation/disambiguate_entities_example.py) | End-to-end disambiguation flow |
| [`disambiguation/extract_graph_from_data_with_entity_disambiguation.py`](pocs/disambiguation/extract_graph_from_data_with_entity_disambiguation.py) | Graph extraction with disambiguation enabled |
| [`post_extraction_canonicalization/post_extraction_canonicalization.py`](pocs/post_extraction_canonicalization/post_extraction_canonicalization.py) | Canonicalize entities after extraction |
| [`post_extraction_canonicalization/post_extraction_canonicalization_example.py`](pocs/post_extraction_canonicalization/post_extraction_canonicalization_example.py) | Worked example for the canonicalization flow |
| [`prefetch_disambiguation/prefetch_disambiguation.py`](pocs/prefetch_disambiguation/prefetch_disambiguation.py) | Prefetch candidates before disambiguation |
| [`prefetch_disambiguation/prefetch_disambiguation_example.py`](pocs/prefetch_disambiguation/prefetch_disambiguation_example.py) | Worked example combining prefetch + disambiguation |

## 🔍 By feature (cross-folder index)

Same scripts, indexed by what they demonstrate.

### Memory API (V2: remember / recall / improve / forget)
- [`demos/remember_recall_improve_example.py`](demos/remember_recall_improve_example.py)
- [`guides/agent_memory_quickstart.py`](guides/agent_memory_quickstart.py)
- [`guides/improve_quickstart.py`](guides/improve_quickstart.py)
- [`guides/recall_core.py`](guides/recall_core.py)

### Session memory & feedback
- [`demos/conversation_session_persistence_example.py`](demos/conversation_session_persistence_example.py)
- [`demos/session_feedback_example.py`](demos/session_feedback_example.py)
- [`demos/session_feedback_lifecycle_demo/backend/app.py`](demos/session_feedback_lifecycle_demo/backend/app.py)
- [`demos/feedback_score_shifting_example.py`](demos/feedback_score_shifting_example.py)
- [`guides/importance_weight.py`](guides/importance_weight.py)

### Temporal awareness
- [`demos/temporal_awareness_example/temporal_awareness_example.py`](demos/temporal_awareness_example/temporal_awareness_example.py)
- [`guides/temporal_recall.py`](guides/temporal_recall.py)

### Ontology
- [`demos/ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py`](demos/ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py)
- [`guides/ontology_quickstart.py`](guides/ontology_quickstart.py)

### Multimedia & non-text ingestion
- [`demos/multimedia_processing/multimedia_audio_image_processing_example.py`](demos/multimedia_processing/multimedia_audio_image_processing_example.py)
- [`demos/web_url_content_ingestion_example.py`](demos/web_url_content_ingestion_example.py)
- [`demos/dlt_ingestion_example.py`](demos/dlt_ingestion_example.py)

### SQL → knowledge graph
- [`custom_pipelines/relational_database_to_knowledge_graph_migration_example.py`](custom_pipelines/relational_database_to_knowledge_graph_migration_example.py)
- [`demos/simple_relational_database_migration_example/simple_relational_database_migration_example.py`](demos/simple_relational_database_migration_example/simple_relational_database_migration_example.py)
- [`demos/complex_relational_database_migration_example/complex_relational_database_migration_example.py`](demos/complex_relational_database_migration_example/complex_relational_database_migration_example.py)

### Custom pipelines / tasks
- [`custom_pipelines/custom_cognify_pipeline_example.py`](custom_pipelines/custom_cognify_pipeline_example.py)
- [`custom_pipelines/agentic_reasoning_procurement_example.py`](custom_pipelines/agentic_reasoning_procurement_example.py)
- [`custom_pipelines/dynamic_steps_resume_analysis_hr_example.py`](custom_pipelines/dynamic_steps_resume_analysis_hr_example.py)
- [`custom_pipelines/memify_coding_agent_rule_extraction_example.py`](custom_pipelines/memify_coding_agent_rule_extraction_example.py)
- [`custom_pipelines/organizational_hierarchy/`](custom_pipelines/organizational_hierarchy/) (high-level + low-level variants)
- [`guides/custom_data_models.py`](guides/custom_data_models.py)
- [`guides/custom_graph_model.py`](guides/custom_graph_model.py)
- [`guides/custom_prompts.py`](guides/custom_prompts.py)
- [`guides/custom_tasks_and_pipelines.py`](guides/custom_tasks_and_pipelines.py)
- [`demos/custom_pipeline_single_object_example.py`](demos/custom_pipeline_single_object_example.py)
- [`demos/custom_graph_model_entity_schema_definition.py`](demos/custom_graph_model_entity_schema_definition.py)

### Multi-tenant / permissions
- [`configurations/permissions_example/`](configurations/permissions_example/) (4 scripts)

### Backends
- [`database_examples/`](database_examples/) — 5 backends end-to-end
- [`configurations/database_examples/`](configurations/database_examples/) — 4 graph + 1 hybrid configurations

### Visualization & UI
- [`guides/graph_visualization.py`](guides/graph_visualization.py)
- [`demos/start_local_ui_frontend_example.py`](demos/start_local_ui_frontend_example.py)

### Storage backends
- [`guides/s3_storage.py`](guides/s3_storage.py)

### Disambiguation / canonicalization
- [`pocs/disambiguation/`](pocs/disambiguation/) (3 scripts)
- [`pocs/post_extraction_canonicalization/`](pocs/post_extraction_canonicalization/) (2 scripts)
- [`pocs/prefetch_disambiguation/`](pocs/prefetch_disambiguation/) (2 scripts)
- [`guides/consolidate_entity_descriptions_example.py`](guides/consolidate_entity_descriptions_example.py)

## ⚙️ Running an example

```bash
# Install dev environment
uv sync --dev --all-extras --reinstall

# Configure API keys (one-time)
cp .env.example .env
# edit .env: set LLM_API_KEY (your OpenAI key) at minimum

# Run any example
uv run python examples/demos/simple_cognee_example.py
```

For non-OpenAI providers (Anthropic, Bedrock, Ollama, fastembed, …) see
[the cognee docs](https://docs.cognee.ai) and `cognee/.env.example`.

## 🤝 Contributing a new example

1. Pick the right folder:
   - `demos/` — broad feature demonstration
   - `guides/` — small focused how-to
   - `custom_pipelines/` — custom pipeline composition
   - `pocs/` — research-shaped exploration
2. Self-contained: an example should run with `uv run python <path>` after `uv sync` and a configured `.env`. No global setup.
3. Add an entry to the appropriate table above.
4. If your example demonstrates a feature, add it to the **By feature** cross-index too.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the broader contribution flow.
