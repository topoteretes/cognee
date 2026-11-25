# Cognee architecture

## Project Overview

Cognee is an open-source AI memory platform that transforms raw data into persistent knowledge graphs for AI agents. It replaces traditional RAG (Retrieval-Augmented Generation) with an ECL (Extract, Cognify, Load) pipeline combining vector search, graph databases, and LLM-powered entity extraction.

**Requirements**: Python 3.10 - 3.14

## Architecture Overview

See the [repository layout](../AGENTS.md#repository-layout) for package locations and [memory internals](memory-internals.md) for operation flows and feature behavior.

### Key Architectural Patterns

#### 1. Pipeline-Based Processing
All data flows through task-based pipelines (`cognee/modules/pipelines/`). Tasks are composable units that can run sequentially or in parallel. Example pipeline tasks: `classify_documents`, `extract_graph_from_data`, `add_data_points`. The runner semantics (a task's `batch_size` batches the *previous* task's output, `enriches`, `ctx` injection, and which of the two `run_pipeline` functions to import) are in the `cognee/modules/pipelines/__init__.py` docstring; the index of all task implementations is `cognee/tasks/README.md`.

Abandoned-run recovery checks at API startup and every 5 minutes. The `COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS` setting (default 3600, nonnegative) is an eligibility floor, not proof of inactivity. Each new pipeline run holds an OS lock under `SYSTEM_ROOT_DIRECTORY/pipeline_run_locks` from its persisted start through its terminal record, including the background handoff. Recovery must acquire the existing ownership marker and re-read the latest run status before rolling back. Active dataset locks defer recovery too. Completed runs remove their markers; a crashed writer leaves its marker for recovery. Legacy rows and markers unavailable on this filesystem are skipped, so workers need the same persistent system directory with working OS file locks to recover each other's runs. Missing ownership is never inferred from run age alone. A run that a later run of the same pipeline has followed on its dataset is rolled back like any other, but its closing row is dated just after its own last row, so closing it late does not change the dataset's status.


#### 2. Interface-Based Database Adapters
Multiple backends are supported through adapter interfaces:
- **Graph**: Ladybug (default), Neo4j, Neptune, Postgres (demo) via `GraphDBInterface`
- **Vector**: LanceDB (default), PGVector, Neptune Analytics, Turso via `VectorDBInterface` (ChromaDB/Qdrant/Weaviate/Milvus via community adapters)
- **Relational**: SQLite (default), PostgreSQL

Key files:
- `cognee/infrastructure/databases/graph/graph_db_interface.py`
- `cognee/infrastructure/databases/vector/vector_db_interface.py`

#### 3. Multi-Tenant Access Control
User → Dataset → Data hierarchy with permission-based filtering. Enable with `ENABLE_BACKEND_ACCESS_CONTROL=True`. Each user+dataset combination can have isolated graph/vector databases — but only on backends with a dataset-database handler.

**Multi-tenancy support matrix** (source of truth: `cognee/infrastructure/databases/dataset_database_handler/supported_dataset_database_handlers.py`):

| Layer | Backend | Isolated per user+dataset? | Notes |
|---|---|---|---|
| Graph | Ladybug/Kuzu (default) | ✅ | embedded, one database per dataset |
| Graph | Neo4j | ✅ | default handler `neo4j`: one Neo4j database per dataset inside the DBMS — requires an edition with multi-database support (Enterprise/Aura). Two alternates: `neo4j_community` runs one Docker container per dataset, because Community edition serves exactly one database per server (needs Docker); `neo4j_aura_dev` provisions a whole Aura instance per dataset — dev/PoC only, not production-ready |
| Graph | Postgres | ✅ | default handler `postgres_graph`: one Postgres database per dataset. `postgres_graph_shared` instead gives each dataset a schema (`ds_<dataset_id>`) in cognee's main database, so no `CREATE DATABASE` privilege is needed. graph-on-Postgres is itself a demo feature (see [Postgres graph configuration](runtime-configuration.md#graph-databases)) |
| Graph | Turso | ✅ | one Turso database file per dataset; local only (no remote sync in this version) |
| Graph | Neptune, ladybug-remote | ❌ | requires `ENABLE_BACKEND_ACCESS_CONTROL=false` |
| Vector | LanceDB (default) | ✅ | |
| Vector | PGVector | ✅ | default handler `pgvector`: one Postgres database per dataset. `pgvector_shared` uses a schema (`ds_<dataset_id>`) in cognee's main database instead — no `CREATE DATABASE` privilege needed |
| Vector | Turso | ✅ | one Turso database file per dataset; exact cosine scan, no approximate index |
| Vector | Neptune Analytics | ❌ | requires `ENABLE_BACKEND_ACCESS_CONTROL=false` |
| Vector | Community adapters (ChromaDB, Qdrant, …) | ❌ | unless the adapter registers a handler via `use_dataset_database_handler()` |
| Relational | SQLite / Postgres / Turso | n/a — always shared | one relational DB holds users, ACLs, and the dataset-database registry; it is never isolated per dataset |

How it works:
- A default handler is derived from the configured provider (`GraphConfig.fill_derived` and the vector-config equivalent), so in-tree backends work with no handler setting at all. An explicit `GRAPH_DATASET_DATABASE_HANDLER` / `VECTOR_DATASET_DATABASE_HANDLER` is never overwritten by that derivation — that is how you reach the non-default handlers named above (`neo4j_community`, `neo4j_aura_dev`, `postgres_graph_shared`, `pgvector_shared`), each of which also requires its matching provider.
- **Both** the graph and vector backends must support isolation. If either doesn't, cognee raises an `EnvironmentError` naming the unsupported handler — with the flag on (its default), an unsupported backend is a hard error, not a silent fallback to shared databases. The fix is switching backends or setting `ENABLE_BACKEND_ACCESS_CONTROL=false`.
- New backends gain multi-tenancy by registering a `DatasetDatabaseHandlerInterface` implementation in the registry (or at runtime via `use_dataset_database_handler()`).
- The shared dataset routing index lives outside the isolated dataset stores, in the deployment's default vector store configured by `VECTOR_DB_URL`. See [dataset routing](retrieval-internals.md#search-retrieval) for how searches use the index.

### Layer Structure

```
API Layer (cognee/api/v1/)
    ↓
Memory API (remember, recall, improve, forget)
    ↓
Low level operations (add, cognify, search)
    ↓
Pipeline Orchestrator (cognee/modules/pipelines/)
    ↓
Task Execution Layer (cognee/tasks/)
    ↓
Domain Modules (graph, retrieval, ingestion, etc.)
    ↓
Infrastructure Adapters (LLM, databases)
    ↓
External Services (OpenAI, Ladybug, LanceDB, etc.)
```

### Core Data Models

#### Engine Models (`cognee/infrastructure/engine/models/`)
- **DataPoint** - Base class for all graph nodes (versioned, with metadata)
- **Edge** - Graph relationships (source, target, relationship type)
- **Triplet** - (Subject, Predicate, Object) representation

#### Graph Models (`cognee/shared/data_models.py`)
- **KnowledgeGraph** - Container for nodes and edges
- **Node** - Entity (id, name, type, description)
- **Edge** - Relationship (source_node_id, target_node_id, relationship_name)

### Key Infrastructure Components

#### LLM Gateway (`cognee/infrastructure/llm/LLMGateway.py`)
Unified interface for multiple LLM providers: OpenAI, Anthropic, Gemini, Ollama, Mistral, Bedrock. Structured output goes through litellm's schema-native response format by default (`STRUCTURED_OUTPUT_FRAMEWORK=litellm_native`); Instructor and BAML are optional extras.

#### Embedding Engines
Factory pattern for embeddings: `cognee/infrastructure/databases/vector/embeddings/get_embedding_engine.py`

#### Document Loaders
Support for PDF, CSV, images, audio, video, code files (DOCX/PPTX and other office formats through the `docs`/`docling` extras) in `cognee/infrastructure/loaders/` — core loaders in `core/`; loaders backed by third-party libraries in `external/` (pypdf and dlt are core deps; unstructured and beautiful_soup register only with their extras, `docs` / `scraping`; docling registers without its extra but raises on first use; advanced_pdf falls back to pypdf without `docs`); registry in `supported_loaders.py`

## Extension Points

### Adding New Functionality

1. **New Task Type**: Create task function in `cognee/tasks/`, return Task object, register in pipeline
2. **New Database Backend**: Implement `GraphDBInterface` or `VectorDBInterface` in `cognee/infrastructure/databases/`
3. **New LLM Provider**: Add configuration in LLM config (uses litellm)
4. **New Document Processor**: Implement `LoaderInterface` in `cognee/infrastructure/loaders/` and register it in `supported_loaders.py` there
5. **New Search Type**: Add to `SearchType` enum and implement retriever in `cognee/modules/retrieval/`
6. **Custom Graph Models**: Define Pydantic models extending `DataPoint` in your code
7. **Custom Chunker**: Subclass `Chunker` with its own `chunker_id`, add it with `register_chunker(name, cls)` (`cognee/modules/chunking/chunker_registry.py`) and name it in `DEFAULT_CHUNKER`; every call that names no chunker (SDK, CLI, API, cognee's own pipelines) then cuts with it. Register it in every process that reads the setting

## API Structure

FastAPI application with versioned routes under `/api/v1/` (routers registered in `cognee/api/client.py`):
- `/remember` - Store data in memory
- `/recall` - Query memory
- `/improve` - Graph enrichment/indexing
- `/forget` - Unified deletion
- `/add`, `/cognify`, `/search`, `/delete` - Low level operations
- `/datasets` - Dataset management
- `/users` - Authentication (when `REQUIRE_AUTHENTICATION` is effectively true; see [security settings](runtime-configuration.md#security-considerations))
- `/visualize` - Graph visualization server
- `/provenance` - Audit-ledger reads (entry/history, verify, statistics, export, anchor, status)

`DELETE /api/v1/search/history` clears the caller's own history listed by `GET /api/v1/search` and `GET /api/v1/recall`. The endpoint takes no user parameter. See [search history retention and deletion](memory-internals.md#search-history-retention-and-deletion) for SDK and CLI equivalents and the related deletion rules.

Request bodies accept both snake_case and camelCase (`cognee/api/DTO.py` sets `alias_generator=to_camel` with `populate_by_name=True`). There is no `/feedback` route — feedback is CLI- and SDK-only.

## Python SDK Entry Points

Main functions exported from `cognee/__init__.py`.

Memory API (primary):
- `remember(data, dataset_name="main_dataset", session_id=..., self_improvement=True)` - Store data
- `recall(query_text, query_type=None, datasets=..., top_k=15, session_id=...)` - Query memory
- `improve(dataset="main_dataset", session_ids=..., node_name=...)` - Enrich/index the graph
- `forget(data_id=..., dataset=..., dataset_id=..., everything=False, memory_only=False, clean_history=False)` - Remove data

Low level operations:
- `add(data, dataset_name)` - Ingest data
- `cognify(datasets)` - Build knowledge graph
- `search(query_text, query_type)` - Query knowledge
- `delete(data_id)` - Remove data (deprecated since 0.3.9)

Supporting:
- `config()` - Configuration management
- `datasets()` - Dataset operations
- `serve(url)` / `disconnect()` - Point the SDK at a running instance

All functions are async - use `await` or `asyncio.run()`. See `examples/advanced_guides/remember_recall_improve_example.py` for permanent memory, session memory, and the sync between them.

## Common Patterns

### Creating a Custom Pipeline Task
```python
from cognee.modules.pipelines.tasks.task import Task


async def my_custom_task(data):
    # Your logic here
    processed_data = process(data)
    return processed_data


# Use in pipeline
task = Task(my_custom_task)
```

### Accessing Databases Directly
```python
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async

graph_engine = await get_graph_engine()
vector_engine = await get_vector_engine_async()
```

### Using LLM Gateway
```python
from cognee.infrastructure.llm.LLMGateway import LLMGateway

response = await LLMGateway.acreate_structured_output(
    text_input="Your prompt", system_prompt="System instructions", response_model=YourPydanticModel
)
```

## Key Concepts

### Datasets
Datasets are project-level containers that support organization, permissions, and isolated processing workflows. Each user can have multiple datasets with different access permissions.

```python
# Create/use a dataset
await cognee.remember(data, dataset_name="my_project")
await cognee.recall("my question", datasets=["my_project"])
```

`remember()` and `add()` target the default dataset `main_dataset` when `dataset_name` is omitted.

### DataPoints
Atomic knowledge units that form the foundation of graph structures. All graph nodes extend the `DataPoint` base class with versioning and metadata support.

### Permissions System
Multi-tenant architecture with users, roles, and Access Control Lists (ACLs):
- Read, write, delete, and share permissions per dataset
- Enable with `ENABLE_BACKEND_ACCESS_CONTROL=True`
- Supports isolated graph/vector databases per user+dataset — backend support varies; see the multi-tenancy support matrix under "Multi-Tenant Access Control" above
