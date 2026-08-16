# Cognee Demos

Feature demos — the broadest coverage of cognee's surface in one folder.
Where [`../guides/`](../guides/) shows one concept per script, demos stitch
several features together or showcase a single feature end-to-end with
visual output.

## Top-level scripts

| Script | Demonstrates |
|---|---|
| [`session_flow_stepwise_demo.py`](session_flow_stepwise_demo.py) | Step-by-step trace of the cognee 1.0 memory loop (`remember()` + `recall()`) |
| [`contradiction_feedback_demo.py`](contradiction_feedback_demo.py) | Contradiction detection + feedback, live and visualized step by step |
| [`session_feedback_example.py`](session_feedback_example.py) | Capturing thumbs-up/down feedback on retrieval |
| [`feedback_score_shifting_example.py`](feedback_score_shifting_example.py) | How feedback nudges retrieval scores |
| [`live_session_context_feedback_demo.py`](live_session_context_feedback_demo.py) | Live session-context feedback loop |
| [`agentic_session_context_demo.py`](agentic_session_context_demo.py) | The agentic counterpart: learning agent-profile lessons from an agent's own traces |
| [`session_context_growth_demo.py`](session_context_growth_demo.py) | Deterministic JSON demo of session-context growth |
| [`hybrid_context_only_demo.py`](hybrid_context_only_demo.py) | Inspecting HybridRetriever retrieval, context, and completion on a small corpus |
| [`truth_subspace_reranking_demo.py`](truth_subspace_reranking_demo.py) | Truth-subspace re-ranking and its effect on ranking (see also [`../advanced_guides/truth_centroid_slots_demo.py`](../advanced_guides/truth_centroid_slots_demo.py)) |
| [`custom_graph_model_entity_schema_definition.py`](custom_graph_model_entity_schema_definition.py) | Define your own entity schema for graph extraction |
| [`custom_pipeline_single_object_example.py`](custom_pipeline_single_object_example.py) | Run a custom pipeline on a single object (TaskSpec / BoundTask deferred-call pattern) |
| [`dynamic_multiple_weighted_edges_example.py`](dynamic_multiple_weighted_edges_example.py) | Many-to-many edges with per-edge weights |
| [`dlt_ingestion_example.py`](dlt_ingestion_example.py) | Ingest via [dlt](https://dlthub.com/) sources |
| [`local_ollama_example.py`](local_ollama_example.py) | Running cognee fully locally with Ollama (Llama 3.1 8B) |
| [`pipeline_api_proposal.py`](pipeline_api_proposal.py) | Proposal-style exploration of what a simplified pipeline API could look like |

## Sub-directory demos

| Demo | Demonstrates |
|---|---|
| [`comprehensive_example/`](comprehensive_example/) | End-to-end with most features stitched together |
| [`skill_feedback_loop/`](skill_feedback_loop/) | Ingest skills, record a weak run, and propose an improvement |
| [`session_feedback_lifecycle_demo/`](session_feedback_lifecycle_demo/) | Full feedback-loop application (FastAPI backend + frontend) |
| [`simple_relational_database_migration_example/`](simple_relational_database_migration_example/) | SQL → knowledge graph (small schema) |
| [`complex_relational_database_migration_example/`](complex_relational_database_migration_example/) | SQL → knowledge graph (richer schema) |

## Running

```bash
uv run python examples/demos/<script>.py
```

Requires `LLM_API_KEY` in `.env` (copy `.env.template`). The relational
migration demos expect a local Postgres database; `local_ollama_example.py`
expects a running Ollama instance. See each script's header for details.
