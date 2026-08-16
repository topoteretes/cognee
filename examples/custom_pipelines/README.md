# Cognee Custom Pipelines

Examples that replace or extend the default `cognify` pipeline — composing
your own task lists, adding custom tasks, or lifting whole relational
schemas into a knowledge graph. These use the lower-level `add` / `cognify` /
`search` API on purpose: the pipeline itself is the subject.

| Script | Demonstrates |
|---|---|
| [`custom_cognify_pipeline_example.py`](custom_cognify_pipeline_example.py) | Replace the default `cognify` task list with your own |
| [`custom_tasks_and_pipelines.py` (guide)](../guides/custom_tasks_and_pipelines.py) | Author your own tasks and compose them (start here) |
| [`memify_coding_agent_rule_extraction_example.py`](memify_coding_agent_rule_extraction_example.py) | Distill coding-agent traces into reusable rules |
| [`agentic_reasoning_procurement_example.py`](agentic_reasoning_procurement_example.py) | Multi-step reasoning over a procurement dataset (nodesets; requires Ladybug or Neo4j) |
| [`dynamic_steps_resume_analysis_hr_example.py`](dynamic_steps_resume_analysis_hr_example.py) | A pipeline whose steps branch based on the ingested content (resume screening) |
| [`relational_database_to_knowledge_graph_migration_example.py`](relational_database_to_knowledge_graph_migration_example.py) | Lift a SQL schema + data into a knowledge graph (uses a local Postgres migration database; no backend ACL) |
| [`organizational_hierarchy/organizational_hierarchy_pipeline_example.py`](organizational_hierarchy/organizational_hierarchy_pipeline_example.py) | Org-chart ingestion via the high-level API (bundled `data/companies.json`, `data/people.json`) |
| [`organizational_hierarchy/organizational_hierarchy_pipeline_low_level_example.py`](organizational_hierarchy/organizational_hierarchy_pipeline_low_level_example.py) | Same dataset through the low-level Task API |

For smaller single-object custom pipelines see
[`../demos/custom_pipeline_single_object_example.py`](../demos/custom_pipeline_single_object_example.py).

## Running

```bash
uv run python examples/custom_pipelines/<script>.py
```

Requires `LLM_API_KEY` in `.env` (copy `.env.template`). The migration
example additionally expects a local Postgres database (see the script
header); `agentic_reasoning_procurement_example.py` requires a Ladybug or
Neo4j graph database.
