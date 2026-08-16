---
name: cognee-community
description: Use when the user needs something that ships outside cognee core — community database adapters (Qdrant, Milvus, Weaviate, Redis, Pinecone, FalkorDB, Memgraph, DuckDB, NetworkX, …), data-source connectors (Slack, Gmail, Notion, Confluence, Google Drive), custom tasks/pipelines/retrievers (Exa, ScrapeGraph, codify), Keywords AI observability — or wants to contribute a package to the cognee-community repo.
---

# Use and contribute cognee-community packages

Community-maintained plugins live in a separate monorepo:
https://github.com/topoteretes/cognee-community. Everything installable is
under `packages/`; `experimental/` holds demos (n8n nodes, dlt demos,
bauplan, tower) that are not published packages. Each package publishes to
PyPI as `cognee-community-<family>-<kind>-<name>` and imports as the same
name with underscores.

## Package families

| Family | Packages |
|---|---|
| Vector adapters | azureaisearch, milvus, moss, opengauss, opensearch, pinecone, qdrant, redis, singlestore, turbopuffer, valkey, weaviate |
| Graph adapters | arcadedb, memgraph, networkx, pggraph, spanner, turbopuffer, turingdb |
| Hybrid (graph+vector in one DB) | arcadedb, duckdb, falkordb, helixdb |
| Connectors (data sources) | confluence, gmail, google-drive, notion, slack |
| Tasks / pipelines / retrievers | codify_tasks, codify_pipeline, code_retriever, exa_tasks, scrapegraph_tasks |
| Observability | keywordsai (`MONITORING_TOOL=keywordsai` + `KEYWORDSAI_API_KEY`) |

## Using a database adapter

Install, then **import the package's `register` module before cognee touches
any engine** — registration is what makes the provider name valid:

```python
uv pip install cognee-community-vector-adapter-qdrant
```

```python
import cognee
from cognee import config
from cognee_community_vector_adapter_qdrant import register  # noqa: F401

config.set_vector_db_config({
    "vector_db_provider": "qdrant",
    "vector_db_url": "http://localhost:6333",
    "vector_db_key": "...",
    "vector_dataset_database_handler": "qdrant",  # only if the adapter ships one
})
```

The `register.py` calls `use_vector_adapter(name, AdapterClass)` /
`use_graph_adapter(...)`. Setting `VECTOR_DB_PROVIDER`/`GRAPH_DATABASE_PROVIDER`
to a community name **without** the register import raises "Unsupported
vector database provider". Hybrid adapters (e.g. falkordb) register as both
graph and vector — set both configs to the same provider name.

**Multi-tenancy caveat**: with `ENABLE_BACKEND_ACCESS_CONTROL=true` (the
default), both backends must have a dataset-database handler or cognee raises
`EnvironmentError`. Community adapters that ship one (registered via
`use_dataset_database_handler` in their `register.py`): qdrant, moss,
singlestore, turbopuffer (vector + graph), falkordb, arcadedb, helixdb. All
other community adapters need `ENABLE_BACKEND_ACCESS_CONTROL=false`.

## Using a connector

Connectors expose a `dlt` source you hand straight to `remember()`; they
reuse core's DLT ingestion path, so snapshot sync and forget-on-delete work
with no core changes:

```python
from cognee_community_connector_slack import slack_export_source

await cognee.remember(
    slack_export_source("/path/to/slack-export"),
    dataset_name="team-slack-export",  # use a dedicated dataset
    max_rows_per_table=0,
)
```

Same shape for gmail ("ask my inbox"), notion, confluence, and google-drive
(incremental, forget-on-delete). Each package README documents its
credentials; always give a connector its own dataset.

## Verifying an install

Every package has `examples/example.py` (run `uv run python examples/example.py`
from the package dir) and a `tests/` directory. An LLM API key is still
required (`LLM_API_KEY`, OpenAI by default).

## Contributing a package

- **Branch from `main`** — unlike the core repo, cognee-community does not
  use a `dev` branch.
- Follow the existing structure: package dir under `packages/<family>/<name>/`
  with `pyproject.toml`, a `README.md` (install + usage), `examples/example.py`,
  and `tests/` that go beyond the example.
- New DB adapters implement `VectorDBInterface` / `GraphDBInterface` from
  core, expose a `register.py`, and should run the shared conformance tests
  in `packages/shared/contract_suite/` (vector_contract.py / graph_contract.py).
- Add a handler via `use_dataset_database_handler(...)` if the backend can
  isolate per user+dataset — that's what makes it work with access control on.
- Name it `cognee-community-<family>-<kind>-<name>` and add it to the tables
  in the repo README. Lint config is the repo-root `ruff.toml`.
