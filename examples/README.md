# Cognee Examples

Runnable example scripts demonstrating cognee end-to-end — 64 scripts across three folders.
They double as the smoke-test corpus the team uses to verify behaviour across the SDK.

> **New here?** Start with [`guides/simple_cognee_example.py`](guides/simple_cognee_example.py)
> (the canonical `remember → recall` flow), then follow the quickstart map below.

## 🚀 Quickstart map (5 examples to start with)

| Example | What you'll learn |
|---|---|
| [`guides/simple_cognee_example.py`](guides/simple_cognee_example.py) | Canonical `remember → recall` pipeline |
| [`advanced_guides/remember_recall_improve_example.py`](advanced_guides/remember_recall_improve_example.py) | The v1.0 memory API (`remember`, `recall`, `improve`, `forget`) |
| [`guides/agent_memory_quickstart.py`](guides/agent_memory_quickstart.py) | Wrap an LLM agent with cognee memory |
| [`guides/graph_visualization.py`](guides/graph_visualization.py) | Render the resulting knowledge graph |
| [`guides/sessions.py`](guides/sessions.py) | Session-scoped memory via `session_id` |

## 📁 Top-level layout

| Folder | What lives there | Count |
|---|---|---|
| [`guides/`](guides/) | One feature per script: concise, self-contained how-tos | 31 |
| [`advanced_guides/`](advanced_guides/) | Deeper takes on topics a guide already covers | 8 |
| [`demos/`](demos/) | Multiple features stitched into use cases, grouped by topic | 25 |

One line each: **guides teach a feature, advanced guides deepen a feature, demos combine
features.** See [Contributing](#-contributing-a-new-example) for the precise category rules.

## 📘 `guides/` — one feature per script

### Getting started
| Script | Demonstrates |
|---|---|
| [`simple_cognee_example.py`](guides/simple_cognee_example.py) | Canonical `remember → recall` flow (start here) |
| [`recall_core.py`](guides/recall_core.py) | `recall` semantics and parameters |
| [`improve_quickstart.py`](guides/improve_quickstart.py) | Graph enrichment before/after `improve()` |
| [`agent_memory_quickstart.py`](guides/agent_memory_quickstart.py) | Wrap an LLM agent with `@cognee.agent_memory` |

### Sessions & self-improvement
| Script | Demonstrates |
|---|---|
| [`sessions.py`](guides/sessions.py) | Session-scoped memory via `session_id` |
| [`session_distillation.py`](guides/session_distillation.py) | Distilling a session into durable preferences |
| [`global_context_index.py`](guides/global_context_index.py) | Building the index with `improve(build_global_context_index=True)` and updating it incrementally |
| [`global_context_index_recall.py`](guides/global_context_index_recall.py) | What `include_global_context_index` adds to `GRAPH_COMPLETION` retrieval |
| [`importance_weight.py`](guides/importance_weight.py) | Boosting specific memories in retrieval ranking |

### Retrieval
| Script | Demonstrates |
|---|---|
| [`truth_subspace_reranking.py`](guides/truth_subspace_reranking.py) | Teaching retrieval a preference — truth-weighted reranking on/off |
| [`temporal_recall.py`](guides/temporal_recall.py) | Time-bounded queries with `SearchType.TEMPORAL` |
| [`references_example.py`](guides/references_example.py) | `include_references` — answers with evidence |
| [`nodeset_grouping_example.py`](guides/nodeset_grouping_example.py) | `node_set` grouping for filtered retrieval |

### Graph modeling & extraction
| Script | Demonstrates |
|---|---|
| [`custom_graph_model.py`](guides/custom_graph_model.py) | `graph_model=` on `remember` |
| [`custom_data_models.py`](guides/custom_data_models.py) | Custom `DataPoint` subclasses and edges |
| [`custom_prompts.py`](guides/custom_prompts.py) | Overriding the extraction prompt |
| [`custom_tasks_and_pipelines.py`](guides/custom_tasks_and_pipelines.py) | Authoring tasks and composing a pipeline |
| [`ontology_quickstart.py`](guides/ontology_quickstart.py) | Grounding extraction in an OWL ontology |
| [`entity_deduplication.py`](guides/entity_deduplication.py) | Merging duplicate entities (dry-run, then real) |
| [`consolidate_entity_descriptions_example.py`](guides/consolidate_entity_descriptions_example.py) | Merging near-duplicate entity descriptions |
| [`low_level_llm.py`](guides/low_level_llm.py) | Direct LLM-gateway structured output |

### Ingestion
| Script | Demonstrates |
|---|---|
| [`web_url_content_ingestion_example.py`](guides/web_url_content_ingestion_example.py) | Ingesting a URL with `preferred_loaders` (needs network) |
| [`multimedia_audio_image_processing_example.py`](guides/multimedia_audio_image_processing_example.py) | Audio + image ingestion (bundled assets) |
| [`image_ocr_extraction.py`](guides/image_ocr_extraction.py) | Vision transcription + OCR text for an image |
| [`code_graph_example.py`](guides/code_graph_example.py) | Code-graph pipeline + `SearchType.CODE` |

### Visualization
| Script | Demonstrates |
|---|---|
| [`graph_visualization.py`](guides/graph_visualization.py) | Rendering the graph — all seeding modes |
| [`semantic_memory_map.py`](guides/semantic_memory_map.py) | The Semantic memory-map view |
| [`schema_inventory.py`](guides/schema_inventory.py) | Schema/entity inventory side panel |
| [`memory_provenance.py`](guides/memory_provenance.py) | The memory-provenance graph |

### Backends & deployment
| Script | Prerequisite |
|---|---|
| [`neptune_analytics_example.py`](guides/neptune_analytics_example.py) | AWS account + provisioned Neptune Analytics graph |
| [`local_ollama_example.py`](guides/local_ollama_example.py) | `ollama serve` + two pulled models — fully local |
| [`s3_storage.py`](guides/s3_storage.py) | Your S3 bucket + AWS credentials |

## 🎓 `advanced_guides/` — the same topic, deeper

Each script names the simpler guide it builds on and states what it adds.

| Script | Builds on | What it adds |
|---|---|---|
| [`remember_recall_improve_example.py`](advanced_guides/remember_recall_improve_example.py) | `guides/simple_cognee_example.py` + `guides/improve_quickstart.py` | Nine-step tour of the full v1.0 memory API |
| [`conversation_session_persistence_example.py`](advanced_guides/conversation_session_persistence_example.py) | `guides/sessions.py` | Recalls across two sessions, then persists both into the graph |
| [`session_distillation_demo.py`](advanced_guides/session_distillation_demo.py) | `guides/session_distillation.py` | Eight-message session, hybrid recall, post-distillation verification |
| [`global_context_index_smoke_demo.py`](advanced_guides/global_context_index_smoke_demo.py) | `guides/global_context_index.py` + `guides/global_context_index_recall.py` | 12-turn fixture, three-question sweep, pass/fail verdict |
| [`temporal_awareness_example/`](advanced_guides/temporal_awareness_example/) | `guides/temporal_recall.py` | Real biography documents instead of inline text |
| [`ontology_reference_vocabulary/`](advanced_guides/ontology_reference_vocabulary/) | `guides/ontology_quickstart.py` | Bundled OWL + texts as a constraining vocabulary |
| [`simple_document_qa/`](advanced_guides/simple_document_qa/) | `guides/simple_cognee_example.py` | Q&A over a real 150 KB document |
| [`truth_centroid_slots_demo.py`](advanced_guides/truth_centroid_slots_demo.py) | `guides/truth_subspace_reranking.py` | Centroid slots, epochs, and rebuilds behind truth-subspace reranking |

## 🎯 `demos/` — features combined into use cases

Every demo lives in a topic folder.

### [`comprehensive_example/`](demos/comprehensive_example/) — everything at once
| Script | Demonstrates |
|---|---|
| [`cognee_comprehensive_example.py`](demos/comprehensive_example/cognee_comprehensive_example.py) | Three sources, node sets, ontology, memify, filtered recall — stitched together |

### [`agentic/`](demos/agentic/) — agents reasoning over memory
| Script | Demonstrates |
|---|---|
| [`agentic_reasoning_procurement_example.py`](demos/agentic/agentic_reasoning_procurement_example.py) | Research-then-decide over `node_set`-categorized memory: scoped recalls per category, then an LLM decision justified by the evidence |

### [`sessions/`](demos/sessions/) — session memory in action
| Script | Demonstrates |
|---|---|
| [`session_flow_stepwise_demo.py`](demos/sessions/session_flow_stepwise_demo.py) | Narrated five-stage trace of the memory loop |
| [`live_session_context_feedback_demo.py`](demos/sessions/live_session_context_feedback_demo.py) | Learning lessons from conversation feedback, live |
| [`agentic_session_context_demo.py`](demos/sessions/agentic_session_context_demo.py) | Learning agent-profile lessons from tool/action traces |
| [`session_feedback_example.py`](demos/sessions/session_feedback_example.py) | The session feedback API surface (`get_session`, `add_feedback`, …) |
| [`session_feedback_lifecycle_demo/`](demos/sessions/session_feedback_lifecycle_demo/) | Full feedback-loop application (FastAPI backend + frontend) |

### [`feedback/`](demos/feedback/) — feedback signals and what they do to the graph/ranking
| Script | Demonstrates |
|---|---|
| [`contradiction_feedback_demo.py`](demos/feedback/contradiction_feedback_demo.py) | Contradiction detection + feedback, visualized step by step |
| [`feedback_score_shifting_example.py`](demos/feedback/feedback_score_shifting_example.py) | Feedback nudging retrieval scores, with a beta sweep |
| [`skill_feedback_loop/`](demos/feedback/skill_feedback_loop/) | Skills scored, improved, and re-applied in a loop |

### [`ingestion_and_migration/`](demos/ingestion_and_migration/) — getting external data in
| Script | Demonstrates |
|---|---|
| [`dlt_ingestion_example.py`](demos/ingestion_and_migration/dlt_ingestion_example.py) | Six [dlt](https://dlthub.com/) ingestion modes + ontology (needs `cognee[dlt]`) |
| [`simple_relational_database_migration_example/`](demos/ingestion_and_migration/simple_relational_database_migration_example/) | SQL → knowledge graph (small schema) |
| [`complex_relational_database_migration_example/`](demos/ingestion_and_migration/complex_relational_database_migration_example/) | SQL → knowledge graph (richer schema, optional ontology) |
| [`migrate_from_mem0/`](demos/ingestion_and_migration/migrate_from_mem0/) | Importing mem0 memories into cognee |
| [`migrate_from_letta_and_zep/`](demos/ingestion_and_migration/migrate_from_letta_and_zep/) | Importing Letta (MemGPT) agent files and Zep / Graphiti exports into cognee |

### [`custom_pipelines/`](demos/custom_pipelines/) — pipeline composition
| Script | Demonstrates |
|---|---|
| [`custom_cognify_pipeline_example.py`](demos/custom_pipelines/custom_cognify_pipeline_example.py) | Replacing the default `cognify` task list |
| [`custom_pipeline_single_object_example.py`](demos/custom_pipelines/custom_pipeline_single_object_example.py) | Deferred-call pipeline pattern with typed `DataPoint`s |
| [`memify_coding_agent_rule_extraction_example.py`](demos/custom_pipelines/memify_coding_agent_rule_extraction_example.py) | Distilling coding-agent traces into reusable rules |
| [`relational_database_to_knowledge_graph_migration_example.py`](demos/custom_pipelines/relational_database_to_knowledge_graph_migration_example.py) | Migration config + tuned recalls |
| [`dynamic_steps_resume_analysis_hr_example.py`](demos/custom_pipelines/dynamic_steps_resume_analysis_hr_example.py) | Self-coded run stages toggled per run, over a CV corpus |
| [`organizational_hierarchy/`](demos/custom_pipelines/organizational_hierarchy/) | Org-chart ingestion — high-level and low-level variants |

### [`permissions/`](demos/permissions/) — multi-tenancy (set `ENABLE_BACKEND_ACCESS_CONTROL=True`)
| Script | Demonstrates |
|---|---|
| [`tenant_role_setup_example.py`](demos/permissions/tenant_role_setup_example.py) | Creating tenants and assigning roles |
| [`tenant_role_constraints_example.py`](demos/permissions/tenant_role_constraints_example.py) | What a role may not do |
| [`user_permissions_and_access_control_example.py`](demos/permissions/user_permissions_and_access_control_example.py) | The full ACL surface across users, roles, tenants |
| [`data_access_control_example.py`](demos/permissions/data_access_control_example.py) | Retrieval filtered by ACL, `PermissionDeniedError` paths |

## ⚙️ Running an example

```bash
# Install dev environment
uv sync --dev --all-extras --reinstall

# Configure API keys (one-time)
cp .env.template .env
# edit .env: set LLM_API_KEY (your OpenAI key) at minimum

# Run any example
uv run python examples/guides/simple_cognee_example.py
```

For non-OpenAI providers (Anthropic, Bedrock, Ollama, fastembed, …) see
[the cognee docs](https://docs.cognee.ai), the [Ollama model matrix guide](../docs/ollama_models.md), and `.env.template`.

## 🤝 Contributing a new example

Pick the folder by these rules:

**`guides/`** — teaches exactly one functionality. Three criteria: **(1) single feature** —
one API surface, one lesson; **(2) concise** — one linear flow, readable top-to-bottom in one
sitting; **(3) self-contained** — runnable from the get-go, every input inline. Reading a
bundled file, a remote store, or a third-party account disqualifies it; a pip extra or a
startable local service (Neo4j, Postgres, Ollama) is fine as a documented prerequisite, and
writing output the script creates itself is always fine. *Coverage exception:* if a topic's
only possible script can't be self-contained (binary media, S3), it still becomes the topic's
basic guide.

**`advanced_guides/`** — a guide on a topic **that already has a simpler guide**, going deeper
while staying on that one topic. May be long and may read bundled files, but the docstring must
name the basic guide it builds on and state what it adds.

**`demos/`** — multiple cognee features stitched together, or a realistic scenario/use case.
Lives in a topic subfolder (`agentic/`, `sessions/`, `feedback/`, `ingestion_and_migration/`,
`custom_pipelines/`, `permissions/`) — never loose at the `demos/` root. Scenario folders keep
their own `data/`. If your demo really demonstrates one feature and its length is padding,
it's a guide that grew — trim it.

Research-grade proofs of concept don't belong in `examples/` — keep experiment drivers on a
branch or in the issue that tracks the research.

Then: make sure it runs with `uv run python <path>` after `uv sync` and a configured `.env`,
and add a row to the matching table in this README.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the broader contribution flow.
