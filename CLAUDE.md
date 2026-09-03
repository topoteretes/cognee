# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognee is an open-source AI memory platform that transforms raw data into persistent knowledge graphs for AI agents. It replaces traditional RAG (Retrieval-Augmented Generation) with an ECL (Extract, Cognify, Load) pipeline combining vector search, graph databases, and LLM-powered entity extraction.

**Requirements**: Python 3.10 - 3.14

## Development Commands

### Setup
```bash
# Create virtual environment (recommended: uv)
uv venv && source .venv/bin/activate

# Install with pip or uv
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install with specific extras
uv pip install -e ".[postgres,neo4j,docs]"

# Set up pre-commit hooks
pre-commit install
```

### Available Installation Extras
- **postgres** / **postgres-binary** - PostgreSQL + PGVector support (also enables the Postgres session-cache backend, `CACHE_BACKEND=postgres`)
- **neo4j** - Neo4j graph database support
- **neptune** - AWS Neptune support
- **turso** - Turso vector database support
- **docs** - Document processing (unstructured library)
- **scraping** - Web scraping (Tavily, BeautifulSoup, Playwright; Keenable needs no extra — it uses the built-in httpx)
- **langchain** - LangChain integration
- **llama-index** - LlamaIndex integration
- **anthropic** - Anthropic Claude models
- **ollama** - Ollama local models
- **mistral** - Mistral AI models
- **groq** - Groq API support
- **llama-cpp** - Llama.cpp local inference
- **huggingface** - HuggingFace transformers
- **aws** - S3 storage backend
- **redis** - Redis caching
- **graphiti** - Graphiti-core integration
- **baml** - BAML structured output
- **dlt** - Data load tool (dlt) integration
- **docling** - Docling document processing, slim profile without torch (office/HTML/email/markdown/LaTeX formats)
- **docling-full** - Full docling install with torch-based ML models (adds PDF/image conversion through docling; conflicts with **codegraph** due to tree-sitter pins)
- **codegraph** - Code graph extraction
- **evals** - Evaluation tools
- **deepeval** - DeepEval testing framework
- **posthog** - PostHog analytics
- **tracing** - OpenTelemetry tracing
- **dev** - All development tools (pytest, ty, ruff, etc.)
- **debug** - Debugpy for debugging

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cognee --cov-report=html

# Run specific test file
pytest cognee/tests/test_custom_model.py

# Run specific test function
pytest cognee/tests/test_custom_model.py::test_function_name

# Run async tests
pytest -v cognee/tests/integration/

# Run unit tests only
pytest cognee/tests/unit/

# Run integration tests only
pytest cognee/tests/integration/
```

### Code Quality
```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Run both linting and formatting (pre-commit)
pre-commit run --all-files

# Type checking with ty
ty check .
```

### Running Cognee
```bash
# Using Python SDK
uv run python examples/guides/simple_cognee_example.py

# Using CLI (memory API — the primary surface)
cognee-cli remember "Your text here"   # also accepts file paths / URLs
cognee-cli recall "Your question"
cognee-cli improve -d my_project       # enrich/index the graph
cognee-cli forget --all                # NOTE: no confirmation prompt

# Low level operations (still ship; what the memory commands call underneath)
cognee-cli add "Your text here" && cognee-cli cognify
cognee-cli search "Your query"
cognee-cli delete --all                # prompts before deleting

# Launch full stack with UI
cognee-cli -ui
```

## Architecture Overview

### Core Workflow: remember → recall (+ improve / forget)

As of cognee 1.x the memory API is the primary surface. All functions are async.

1. **remember()** - Store data in memory. Without `session_id` it runs `add()` + `cognify()` and then `improve()` (`self_improvement=True` by default); with `session_id` it writes to the fast session cache and bridges into the graph in the background.
2. **recall()** - Query memory. Auto-routes to a search strategy unless `query_type` is passed (`auto_route=False` falls back to `HYBRID_COMPLETION`). A `session_id` reads the session cache first and falls through to the graph.
3. **improve()** - Enrich/index the graph: triplet embeddings, feedback weights, and (with `session_ids`) bridging session Q&A and distilled learnings into the permanent graph.
4. **forget()** - Unified deletion (`data_id` / `dataset` / `dataset_id` / `everything=True`, plus `memory_only=True` to drop graph+vectors but keep raw files).

#### Low level operations: add → cognify → search/memify

These still ship and are what the memory API calls underneath. Reach for them to drive one stage in isolation (custom pipeline tasks, stage-level debugging), not for ordinary ingestion or retrieval.

1. **add()** - Ingest data (files, URLs, text) into datasets
2. **cognify()** - Extract entities/relationships and build knowledge graph
3. **search()** - Query knowledge using various retrieval strategies
4. **memify()** - Enrich graph with additional context and rules

Note: Using Low level operations over core is useful in the following contexts.
1) functional_relationships= is completely unreachable from remember(). So Only cognify can constrain single-target relationships.
2) remember() hardcodes datasets_arg = [dataset_name]: always exactly one. Use cognify for this: cognify(datasets=["a","b","c"]) or datasets=None (every dataset the user owns.)
3) remember() always runs add() first. To rebuild a graph over data already in the DB — after forget(memory_only=True), or with a new graph_model/ontology, cognify() is the only path.
4) add() is like a staging area for cognify(). But remember automatically adds every time.
5) search() packs skills/tools/max_iter/code_query into retriever_specific_config for you. Using recall() you hand-build that dict yourself.
6) prune.prune_system(metadata=True) drops the relational DB (users, tenants, ACLs, the dataset_database registry, pipeline runs, search history.) forget() touches none of that. Full test teardown is prune's job.
Improve & Memify are virtually the same, though. So no reason not to use improve.

`cognee.delete` is deprecated (since 0.3.9, in favor of `datasets.delete_data`); `forget()` is the v1 replacement that unifies the old delete/prune/empty_dataset paths.

#### recall() vs search()

`recall()` wraps `search()` — its graph path calls the same authorized search — and adds three things: rule-based query routing when `query_type` is omitted (regex scoring, no LLM call, so auto-routing is free), session memory as a searchable source (`scope` = `graph` / `session` / `trace` / `session_context`; with a bare `session_id` a session hit short-circuits the graph search), and normalized results tagged with a `_source` key. Use `recall()` for ordinary retrieval. Drop to `search()` when you need the agentic extras as first-class parameters (`skills`, `tools`, `max_iter`, `code_query`, `node_type`), raw `SearchResult` objects instead of tagged entries, or a pinned `query_type` with no router in the path. Note `search(session_id=...)` only adds session history to the retrieval context — it never searches the session cache as a source; that is `recall()`-only. Full guide: `docs/recall-vs-search.md`.

### Key Architectural Patterns

#### 1. Pipeline-Based Processing
All data flows through task-based pipelines (`cognee/modules/pipelines/`). Tasks are composable units that can run sequentially or in parallel. Example pipeline tasks: `classify_documents`, `extract_graph_from_data`, `add_data_points`.

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
| Graph | Neo4j | ✅ | one Neo4j database per dataset inside the DBMS — requires an edition with multi-database support (Enterprise/Aura). A second handler, `neo4j_aura_dev`, provisions a whole Aura instance per dataset; dev/PoC only, not production-ready |
| Graph | Postgres | ✅ | graph-on-Postgres is itself a demo feature (see warning above) |
| Graph | Turso | ✅ | |
| Graph | Neptune, ladybug-remote | ❌ | requires `ENABLE_BACKEND_ACCESS_CONTROL=false` |
| Vector | LanceDB (default) | ✅ | |
| Vector | PGVector | ✅ | |
| Vector | Turso | ✅ | |
| Vector | Neptune Analytics | ❌ | requires `ENABLE_BACKEND_ACCESS_CONTROL=false` |
| Vector | Community adapters (ChromaDB, Qdrant, …) | ❌ | unless the adapter registers a handler via `use_dataset_database_handler()` |
| Relational | SQLite / Postgres | n/a — always shared | one relational DB holds users, ACLs, and the dataset-database registry; it is never isolated per dataset |

How it works:
- The handler is selected automatically from the configured provider (`GraphConfig.fill_derived` and the vector-config equivalent) — you never set it by hand for in-tree backends.
- **Both** the graph and vector backends must support isolation. If either doesn't, cognee raises an `EnvironmentError` naming the unsupported handler — with the flag on (its default), an unsupported backend is a hard error, not a silent fallback to shared databases. The fix is switching backends or setting `ENABLE_BACKEND_ACCESS_CONTROL=false`.
- New backends gain multi-tenancy by registering a `DatasetDatabaseHandlerInterface` implementation in the registry (or at runtime via `use_dataset_database_handler()`).

### Layer Structure

```
API Layer (cognee/api/v1/)
    ↓
Memory API (remember, recall, improve, forget)
    ↓
Low level operations (add, cognify, search, memify)
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

### Critical Data Flow Paths

#### REMEMBER / RECALL: Memory API
NOTE: This is how the memory API flow works under the hood; it's read as a flow of data. So remember calls add(), cognify(), and improve().
`remember(data)` → `add()` → `cognify()` → `improve()` (when `self_improvement=True`)
`remember(data, session_id=...)` → session cache → background `improve()` bridge
`recall(query)` → auto-route to a `SearchType` → `search()` → permission filter → results

Key files: `cognee/api/v1/remember/remember.py`, `cognee/api/v1/recall/recall.py`, `cognee/api/v1/improve/improve.py`, `cognee/api/v1/forget/forget.py`

The stages below are the Low level operations these call underneath.

#### ADD: Data Ingestion
`add()` → `resolve_data_directories` → `ingest_data` → `save_data_item_to_storage` → Create Dataset + Data records in relational DB

Key files: `cognee/api/v1/add/add.py`, `cognee/tasks/ingestion/ingest_data.py`

#### COGNIFY: Knowledge Graph Construction
`cognify()` → `classify_documents` → `extract_chunks_from_documents` → `extract_graph_from_data` (LLM extracts entities/relationships using Instructor) → `summarize_text` → `add_data_points` (store in graph + vector DBs)

Key files:
- `cognee/api/v1/cognify/cognify.py`
- `cognee/tasks/graph/extract_graph_from_data.py`
- `cognee/tasks/storage/add_data_points.py`

#### UPDATE: Chunk-Level Incremental Updates
`update(data_id, data, dataset_id)` diffs the new content against the stored processed text (`Data.raw_data_location`) and replaces only the chunks the edits touched: a paragraph-anchored multi-region diff (near-linear for any change shape, including repeated-line-heavy content) finds every disjoint changed span (so edits at the top, middle, and end of a document are three small regions, not one giant one), each span is expanded to chunk boundaries and re-chunked with the standard TextChunker (same boundary semantics as pipeline chunks, cut against the token budget recorded on the chunks it replaces — every chunk stores `max_chunk_tokens`, so documents stay self-consistent across config changes; legacy chunks without the field fall back to the current config; only a region's last chunk may be under-filled), replaced chunks (plus their summaries, chunk-orphaned entities, and triplet embeddings) are deleted, and only the new chunks run through LLM extraction. Chunks between regions are kept without being re-chunked, so their boundaries and ids cannot drift. Unaffected chunks keep their node ids, entities, summaries, and embeddings; surviving chunks are renumbered so `chunk_index` stays contiguous. Chunk identity is content-derived (`uuid5(doc : sha256(text) : occurrence)`, see `cognee/modules/chunking/chunk_id.py`), so unchanged content keeps its identity across edits. Both incremental and full fallback paths preserve `data_id`.

Runs under the per-dataset lock with the dataset-scoped database context (multi-tenant-safe). Falls back to the full delete + re-add + cognify flow when preconditions fail (first ingestion, non-text content, pre-v2 chunk ownership, or an unverified graph adapter — verified: Kuzu/Ladybug, Neo4j and the Postgres demo adapter; Neptune falls back); disable with `update(..., chunk_level_diff=False)` or the same query param on `PATCH /update`.

Key files:
- `cognee/api/v1/update/update.py` / `cognee/api/v1/update/incremental.py`
- `cognee/modules/chunking/incremental_chunking.py` (diff + balanced re-split, no-loss invariant)
- `cognee/modules/graph/methods/delete_chunks_incremental.py` (chunk-scoped orphan deletion)

#### SEARCH: Retrieval
`search(query_text, query_type)` → route to retriever type → filter by permissions → return results

Available search types (from `cognee/modules/search/types/SearchType.py`), passed as `query_type` to `recall()` or `search()`:
- **HYBRID_COMPLETION** (default) - Document passages plus entity neighbourhoods, then LLM completion
- **GRAPH_COMPLETION** - Graph traversal + LLM completion
- **GRAPH_SUMMARY_COMPLETION** - Uses pre-computed summaries with graph context
- **GRAPH_COMPLETION_COT** - Chain-of-thought reasoning over graph
- **GRAPH_COMPLETION_CONTEXT_EXTENSION** - Extended context graph retrieval
- **TRIPLET_COMPLETION** - Triplet-based (subject-predicate-object) search
- **RAG_COMPLETION** - Traditional RAG with chunks
- **CHUNKS** - Vector similarity search over chunks
- **CHUNKS_LEXICAL** - Lexical (keyword) search over chunks
- **SUMMARIES** - Search pre-computed document summaries
- **CYPHER** - Direct Cypher query execution (requires `ALLOW_CYPHER_QUERY=True`)
- **NATURAL_LANGUAGE** - Natural language to structured query
- **TEMPORAL** - Time-aware graph search
- **FEELING_LUCKY** - Automatic search type selection
- **CODING_RULES** - Code-specific search rules
- **SKILLS** - Semantic discovery of skill playbooks (metadata-only, no LLM; requires exactly one dataset)

`recall()` picks one of these automatically when `query_type` is omitted. The CLI is narrower: `cognee-cli recall --query-type` accepts only the choices in `cognee/cli/config.py:SEARCH_TYPE_CHOICES` and defaults to `HYBRID_COMPLETION`; the rest are SDK-only.

Key files:
- `cognee/api/v1/search/search.py`
- `cognee/modules/retrieval/context_providers/TripletSearchContextProvider.py`
- `cognee/modules/search/types/SearchType.py`

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
Unified interface for multiple LLM providers: OpenAI, Anthropic, Gemini, Ollama, Mistral, Bedrock. Uses Instructor for structured output extraction.

#### Embedding Engines
Factory pattern for embeddings: `cognee/infrastructure/databases/vector/embeddings/get_embedding_engine.py`

#### Document Loaders
Support for PDF, DOCX, CSV, images, audio, code files in `cognee/infrastructure/files/`

## Important Configuration

### Environment Setup
Copy `.env.template` to `.env` and configure:

```bash
# Minimal setup (defaults to OpenAI + local file-based databases)
LLM_API_KEY="your_openai_api_key"
LLM_MODEL="openai/gpt-5-mini"  # Default model
```

**Important**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both to avoid unexpected defaults.

Default databases (no extra setup needed):
- **Relational**: SQLite (metadata and state storage)
- **Vector**: LanceDB (embeddings for semantic search)
- **Graph**: Ladybug (knowledge graph and relationships)

All stored in `.venv` by default. Override with `DATA_ROOT_DIRECTORY` and `SYSTEM_ROOT_DIRECTORY`.

### Switching Databases

#### Relational Databases
```bash
# PostgreSQL (requires postgres extra: pip install cognee[postgres])
DB_PROVIDER=postgres
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=cognee
DB_PASSWORD=cognee
DB_NAME=cognee_db
```

#### Vector Databases
Supported in-tree: lancedb (default), pgvector, neptune_analytics, turso.
Others (ChromaDB, Qdrant, Weaviate, Milvus, …) are community adapters — install from
https://github.com/topoteretes/cognee-community and register via `use_vector_adapter`
before setting `VECTOR_DB_PROVIDER`, otherwise cognee raises
"Unsupported vector database provider".
```bash
# PGVector (requires postgres extra)
VECTOR_DB_PROVIDER=pgvector
VECTOR_DB_URL=postgresql://cognee:cognee@localhost:5432/cognee_db
```

#### Graph Databases
Supported: ladybug (default), neo4j, neptune, ladybug-remote, postgres_demo (demo; `postgres` is an accepted alias)
```bash
# Neo4j (requires neo4j extra: pip install cognee[neo4j])
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://localhost:7687
GRAPH_DATABASE_NAME=neo4j
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=yourpassword

# Remote Ladybug
GRAPH_DATABASE_PROVIDER=ladybug-remote
GRAPH_DATABASE_URL=http://localhost:8000
GRAPH_DATABASE_USERNAME=your_username
GRAPH_DATABASE_PASSWORD=your_password

# Postgres (requires postgres extra: pip install cognee[postgres])
# DEMO, not production-ready — see the warning below.
# Does not support raw Cypher queries, natural language search, or Graphiti.
# The legacy value `postgres` still resolves to this same adapter.
GRAPH_DATABASE_PROVIDER=postgres_demo
GRAPH_DATABASE_URL=postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db
```

> **⚠️ Warning:** Using Postgres as a graph store is currently a demo feature and is not
> production-ready. Use it to demo keeping relational metadata, PGVector, and graph
> state in a single Postgres service, but rely on a graph-native backend such as Kuzu or Neo4j
> for production workloads.
>
> Interested in further development or production use of Postgres as a graph database? Write to
> us at social@cognee.ai to explore the options.

#### Session Cache
```bash
# Session/conversation cache backend: sqlite (default), postgres, redis, fs, tapes
CACHE_BACKEND=sqlite
# Optional explicit SQLAlchemy URL for sqlite/postgres cache backends (overrides defaults)
CACHE_DB_URL=postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db
# Session-search execution mode: concurrent (default) or sequential
SESSION_SEARCH_MODE=concurrent
```

#### Session Search Modes

A session search (a `search()` with an active session cache) runs in one of two modes,
chosen deployment-wide by `SESSION_SEARCH_MODE`. There is no per-request override.

Both modes make the same two LLM calls per turn — one to analyze the turn for session
context, one to answer. They differ in how those calls are sequenced:

- **`concurrent`** (default) — analysis runs **concurrently** with retrieval and
  answering, so a turn costs one answer call of wall-clock time. Retrieval compensates
  for not having the analysis's rewritten query by running two lanes: the raw question,
  and a deterministic (LLM-free) rewrite built from the last two turns. Their results are
  merged by the retriever before context is formatted.
- **`sequential`** — analysis runs **first**, its rewritten query drives a single
  retrieval, and its context updates are applied before the answer is generated.

The practical difference: in sequential mode, guidance the user states this turn can
influence this turn's answer. In concurrent mode it applies from the next turn onward.

Concurrent mode applies only to `GraphCompletionRetriever`,
`HybridRetriever`, `CompletionRetriever` (`RAG_COMPLETION`), and `TripletRetriever`
(`TRIPLET_COMPLETION`), and only through `search()`. Calling a retriever's
`get_completion()` directly always takes the sequential path. Subclasses, batch queries,
`only_context`, `FEELING_LUCKY`, and sessionless calls fall back to sequential mode
automatically. With `AUTO_FEEDBACK=false`
neither mode analyzes the turn.

#### only_context and `context_format`

`only_context=True` returns the retrieval context instead of an LLM completion. By
default that is the bare context string and nothing else — no session guidance, no
conversation history, no rendered prompt — which is less than a real completion
receives. Pass `context_format="prompt"` to get the full envelope instead:

```python
result = await cognee.recall(
    "why did the migration stall?",
    query_type=SearchType.GRAPH_COMPLETION,  # pin the graph lane — with a bare
    session_id="s1",                         # session_id a session hit would
    only_context=True,                       # short-circuit it (see recall vs search)
    context_format="prompt",                 # default: "context"
)
```

The `"prompt"` shape returns `question`, `context`, `session_context` (the guidance
block plus conversation history), `user_prompt`, and `system_prompt` — the exact
strings `generate_completion` would have sent, built by the same code
(`build_session_prompt` in read-only mode plus `build_completion_prompts`). It makes no
LLM completion or turn-analysis call, writes nothing to the session, and records no QA
turn. It does make **one embedding call** — the conversation-history vector recall —
once per search, shared across the dataset fan-out. With `CACHING=false` the
`system_prompt` still carries the durable preference block, exactly as the real
sessionless completion does.

Search types that never send a single prompt from their template pair report the
session layer and leave `user_prompt`/`system_prompt` empty: the non-generative types
(`CHUNKS`, `SUMMARIES`, `CODE`, …) have no template, and `CYPHER` and
`AGENTIC_COMPLETION` opt out via `supports_prompt_preview = False` (Cypher never
prompts; the agentic loop answers through other templates). For `recall()`, an empty
retrieval yields zero items in either format, so the `on_empty` tools fallback still
fires.

Caveats. `context_format` only affects `only_context` calls. `POST /api/v1/search`
accepts `session_id`; without one the session layer is the default session's. And the
preview is knowingly unfaithful in one place: a real sequential turn first rewrites the
question (`effective_query`), and that rewrite fills `{{ question }}`, drives history
selection, and ranks the guidance block — concurrent mode also merges a second
retrieval lane. Producing the rewrite is an LLM call, so the preview uses the raw query
for all of them: it reports the prompt for the context actually retrieved, not a
replay of a full turn.

### Memory & Performance Tuning Flags

Four flags trade memory features for speed. Know what each turns off before flipping it:

| Flag (default) | Turns off when disabled | Cost of disabling |
|---|---|---|
| `PERSONALIZATION_ENABLED=false` | Per-user preference personalization: one `UserPreference` node per user+dataset with weighted `prefers` edges, retrieval ranking multiplied by those weights, stated-preference text injected into LLM prompts, the per-turn 1-5 rating question, and the `improve()` stage that folds ratings into weights | Off by default, so nothing is lost until you opt in. When on, ranking strength comes from `PERSONALIZATION_INFLUENCE` (default 0.3, valid range [0, 1] — out-of-range values are rejected at startup); personalization also needs a user and a single resolved dataset in context, so multi-dataset searches never personalize |
| `CACHING=true` | The entire session-memory layer: `remember(session_id=...)` raises, `recall()` loses session history and the session-cache short-circuit, `agent_memory` session options error, and `AUTO_FEEDBACK` becomes moot | You lose the fast session write path and self-improving memory — only the slower add+cognify path remains. Do not benchmark cognee with this off; that measures cognee with its memory layer removed |
| `AUTO_FEEDBACK=true` | The automatic per-turn analysis: one structured-output LLM call after each answered query that detects implicit feedback, guides later retrievals, and feeds `improve()`'s agent-context lessons | Memory stops self-tuning from conversation signals. Session store/recall itself keeps working — this is the flag to disable for low-latency reads, since the per-turn LLM call dominates default read latency |
| `DATASET_QUEUE_ENABLED=true` | The per-process cap on concurrent datasets (`DATASET_QUEUE_MAX_CONCURRENT`, default 6), subprocess-engine teardown on scope exit, and pinning of in-use engines against cache eviction | Saves minor per-operation overhead, but embedded engines become unbounded: file-lock leaks and mid-use engine eviction under parallel multi-dataset load. Safe only for single-dataset scripts |

`AUTO_FEEDBACK` is only consulted when `CACHING=true`. If reads feel slow on defaults, set `AUTO_FEEDBACK=false` and keep `CACHING=true` — that keeps session memory while removing the per-turn LLM call.

### LLM Provider Configuration

Supported providers: OpenAI (default), Azure OpenAI, Google Gemini, Anthropic, AWS Bedrock, Ollama, LM Studio, Custom (OpenAI-compatible APIs)

#### OpenAI (Recommended - Minimal Setup)
```bash
LLM_API_KEY="your_openai_api_key"
LLM_MODEL="openai/gpt-5-mini"  # default; or gpt-5, gpt-4o, gpt-4o-mini, etc.
LLM_PROVIDER="openai"
```

#### Azure OpenAI
```bash
LLM_PROVIDER="azure"
LLM_MODEL="azure/gpt-4o-mini"
LLM_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/openai/deployments/gpt-4o-mini"
LLM_API_KEY="your_azure_api_key"
LLM_API_VERSION="2024-12-01-preview"
```

#### Google Gemini (no extra required)
```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini/gemini-2.0-flash-exp"
LLM_API_KEY="your_gemini_api_key"
```

#### Anthropic Claude (requires anthropic extra)
```bash
LLM_PROVIDER="anthropic"
LLM_MODEL="claude-3-5-sonnet-20241022"
LLM_API_KEY="your_anthropic_api_key"
```

#### Ollama (Local - requires ollama extra)
```bash
LLM_PROVIDER="ollama"
LLM_MODEL="llama3.1:8b"
LLM_ENDPOINT="http://localhost:11434/v1"
LLM_API_KEY="ollama"
EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL="nomic-embed-text:latest"
EMBEDDING_ENDPOINT="http://localhost:11434/api/embed"
HUGGINGFACE_TOKENIZER="nomic-ai/nomic-embed-text-v1.5"
```

#### Custom / OpenRouter / vLLM
```bash
LLM_PROVIDER="custom"
LLM_MODEL="openrouter/deepseek/deepseek-r1"
LLM_ENDPOINT="https://openrouter.ai/api/v1"
LLM_API_KEY="your_api_key"
```
OpenRouter model ids change over time (the `:free` tier especially) — check
`https://openrouter.ai/api/v1/models` for a current slug. Embeddings are a
separate catalogue at `https://openrouter.ai/api/v1/embeddings/models` and
must be configured separately; see the OpenRouter block in `.env.template`.

#### AWS Bedrock (requires aws extra)
```bash
LLM_PROVIDER="bedrock"
LLM_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
# Optional for temporary credentials:
# AWS_SESSION_TOKEN="your_session_token"
```

#### LLM Rate Limiting
```bash
LLM_RATE_LIMIT_ENABLED=true
LLM_RATE_LIMIT_REQUESTS=60  # Requests per interval
LLM_RATE_LIMIT_INTERVAL=60  # Interval in seconds
```

#### Instructor Mode (Structured Output)
```bash
# LLM_INSTRUCTOR_MODE controls how structured data is extracted
# Each LLM has its own default (e.g., gpt-4o models use "json_schema_mode")
# Override if needed:
LLM_INSTRUCTOR_MODE="json_schema_mode"  # or "tool_call", "md_json", etc.
```

### Structured Output Framework
```bash
# litellm_native (default): plain litellm, schema-native response_format
# with prompted-JSON fallback — no instructor in the call path
STRUCTURED_OUTPUT_FRAMEWORK="litellm_native"

# Or use Instructor (legacy, via litellm)
STRUCTURED_OUTPUT_FRAMEWORK="instructor"

# Or use BAML (requires baml extra: pip install cognee[baml])
STRUCTURED_OUTPUT_FRAMEWORK="baml"
BAML_LLM_PROVIDER=openai
BAML_LLM_MODEL="gpt-4o-mini"
BAML_LLM_API_KEY="your_api_key"
```

### Storage Backend
```bash
# Local filesystem (default)
STORAGE_BACKEND="local"

# S3 (requires aws extra: pip install cognee[aws])
STORAGE_BACKEND="s3"
STORAGE_BUCKET_NAME="your-bucket-name"
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
DATA_ROOT_DIRECTORY="s3://your-bucket/cognee/data"
SYSTEM_ROOT_DIRECTORY="s3://your-bucket/cognee/system"
```

## Extension Points

### Adding New Functionality

1. **New Task Type**: Create task function in `cognee/tasks/`, return Task object, register in pipeline
2. **New Database Backend**: Implement `GraphDBInterface` or `VectorDBInterface` in `cognee/infrastructure/databases/`
3. **New LLM Provider**: Add configuration in LLM config (uses litellm)
4. **New Document Processor**: Extend loaders in `cognee/modules/data/processing/`
5. **New Search Type**: Add to `SearchType` enum and implement retriever in `cognee/modules/retrieval/`
6. **Custom Graph Models**: Define Pydantic models extending `DataPoint` in your code

### Working with Ontologies
Cognee supports ontology-based entity extraction to ground knowledge graphs in standardized semantic frameworks (e.g., OWL ontologies).

Configuration:
```bash
ONTOLOGY_RESOLVER=rdflib  # Default: uses rdflib and OWL files
MATCHING_STRATEGY=fuzzy   # Default: fuzzy matching with 80% similarity
ONTOLOGY_FILE_PATH=/path/to/your/ontology.owl  # Full path to ontology file
ONTOLOGY_MODE=annotate    # Default: enrich only. strict drops entities with no ontology grounding
```

`ONTOLOGY_MODE=strict` keeps an entity when either its type matches an ontology class or its name matches an individual, and drops the rest (plus their edges). It prunes only the graph — chunk text stays stored/embedded, so CHUNKS/RAG_COMPLETION can still surface dropped entities. It expects an ontology covering the corpus's vocabulary (a small ontology drops most entities; an aggregate dropped/retained count is logged), and an empty/missing ontology file with strict on is a hard error. The mode can also be set per call via `config={"ontology_config": {"ontology_mode": "strict", ...}}`.

Implementation: `cognee/modules/ontology/`

## Branching Strategy

**IMPORTANT**: Always branch from `dev`, not `main`. The `dev` branch is the active development branch.

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

**Core-team PRs must reference a Linear issue.** Put the issue key (e.g. `COG-123`)
in the PR title or the branch name so Linear links the PR to its ticket. This is
enforced by the `Require Linear issue` workflow (`linear-issue-check`), a required
status check. Fork / external-contributor PRs are exempt (the check skips them), so
this rule applies only to internal PRs.

## Code Style

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **String quotes**: Use double quotes `"` not single quotes `'` (enforced by ruff-format)
- **Pre-commit hooks**: Run ruff linting and formatting automatically
- **Type hints**: Encouraged (ty checks enabled)
- **Important**: Always run `pre-commit run --all-files` before committing to catch formatting issues

## Commit & PR Title Style
- **Subject line (required):**
  - The format is (type): (short summary)
  - Write summary as if it is giving an instruction (e.g., "Fix bug" instead of "Fixed bug")
  - 50 chars or less
  - Capitalize first char of summary
  - Do NOT end with a period
- **Body (optional):**
  - **Description:** Explain the motivation behind the change, what problem it solves, and any relevant background.
  - **Use the body to explain what and why, not how.** The body of the commit message should explain why the change was made and what problem it solves. You don't need to explain how the code works, as the code itself should be clear enough for that.
- **Include issue tracking numbers where applicable.** Reference an issue in at least the subject line (e.g., Fixes COG-24), making it easier to trace changes to their corresponding issue.
- **Separate the subject line from the body with a blank line.** This helps differentiate the short description from the detailed explanation. Generally, all commits should have separate subject and body.

## Testing Strategy

Tests are organized in `cognee/tests/`:
- `unit/` - Unit tests for individual modules
- `integration/` - Full pipeline integration tests
- `e2e/` - Full-stack end-to-end suites run per backend in CI (e.g. `e2e/incremental_update/` runs on LadybugDB + LanceDB, Postgres graph + PGVector, and Neo4j + LanceDB)
- `cli_tests/` - CLI command tests
- `tasks/` - Task-specific tests

When adding features, add corresponding tests. Integration tests should cover the full remember → recall flow (or add → cognify → search when the feature lives in one of those stages).

## API Structure

FastAPI application with versioned routes under `/api/v1/` (routers registered in `cognee/api/client.py`):
- `/remember` - Store data in memory
- `/recall` - Query memory
- `/improve` - Graph enrichment/indexing
- `/forget` - Unified deletion
- `/add`, `/cognify`, `/search`, `/memify`, `/delete` - Low level operations
- `/datasets` - Dataset management
- `/users` - Authentication (when `REQUIRE_AUTHENTICATION` is effectively true; see auth posture below)
- `/visualize` - Graph visualization server

Request bodies accept both snake_case and camelCase (`cognee/api/DTO.py` sets `alias_generator=to_camel` with `populate_by_name=True`). There is no `/feedback` route — feedback is CLI- and SDK-only.

## Python SDK Entry Points

Main functions exported from `cognee/__init__.py`.

Memory API (primary):
- `remember(data, dataset_name="main_dataset", session_id=..., self_improvement=True)` - Store data
- `recall(query_text, query_type=None, datasets=..., top_k=15, session_id=...)` - Query memory
- `improve(dataset="main_dataset", session_ids=..., node_name=...)` - Enrich/index the graph
- `forget(data_id=..., dataset=..., dataset_id=..., everything=False, memory_only=False)` - Remove data

Low level operations:
- `add(data, dataset_name)` - Ingest data
- `cognify(datasets)` - Build knowledge graph
- `search(query_text, query_type)` - Query knowledge
- `memify(extraction_tasks, enrichment_tasks)` - Enrich graph
- `delete(data_id)` - Remove data (deprecated since 0.3.9)

Supporting:
- `config()` - Configuration management
- `datasets()` - Dataset operations
- `serve(url)` / `disconnect()` - Point the SDK at a running instance

All functions are async - use `await` or `asyncio.run()`. See `examples/advanced_guides/remember_recall_improve_example.py` for permanent memory, session memory, and the sync between them.

## Security Considerations

Several security environment variables in `.env`:
- `ACCEPT_LOCAL_FILE_PATH` - Allow local file paths (default: True)
- `ALLOW_HTTP_REQUESTS` - Allow HTTP requests from Cognee (default: True)
- `ALLOW_CYPHER_QUERY` - Allow raw Cypher queries (default: True)
- `ENABLE_BACKEND_ACCESS_CONTROL` - Multi-tenant isolation (default: True). When `true`, API auth is required and per-user/dataset DB isolation is enabled. When `false`, single-user mode: shared DBs and auth off unless overridden.
- `REQUIRE_AUTHENTICATION` - Explicit auth override. Unset (default): follows `ENABLE_BACKEND_ACCESS_CONTROL`. `false` is ignored when `ENABLE_BACKEND_ACCESS_CONTROL=true`. For a single-user deployment with auth off, set `ENABLE_BACKEND_ACCESS_CONTROL=false` (and optionally `REQUIRE_AUTHENTICATION=false`).

For production deployments, review and tighten these settings.

## Common Patterns

### Creating a Custom Pipeline Task
```python
from cognee.modules.pipelines.tasks.Task import Task

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
from cognee.infrastructure.llm.get_llm_client import get_llm_client

llm_client = get_llm_client()
response = await llm_client.acreate_structured_output(
    text_input="Your prompt",
    system_prompt="System instructions",
    response_model=YourPydanticModel
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

`remember()`/`add()` without `dataset_name` target the default dataset `main_dataset`; `recall()`/`search()` span all accessible datasets unless one is given.

### DataPoints
Atomic knowledge units that form the foundation of graph structures. All graph nodes extend the `DataPoint` base class with versioning and metadata support.

### Contradiction Detection
Opt-in LLM check that runs as the last `cognify()` task (default **off**). After the graph is stored, it gathers the facts one hop from the entities this ingestion touched — new and pre-existing alike — asks an LLM which pairs cannot both be true, and records each confident conflict as a `contradicts` edge carrying both fact texts, the reason, and the confidence. It only adds edges (never rewrites or deletes) and swallows its own errors, so it can never break ingestion.

- **Enable**: set `CONTRADICTION_DETECTION=true`. When off, the cognify pipeline is unchanged.
- **Tuning** (env): `CONTRADICTION_CONFIDENCE_THRESHOLD` (default 0.5, minimum confidence to flag), `CONTRADICTION_MAX_FACTS` (default 500, cap on facts per LLM call).
- **Applies to `remember()` too** — and to session memory bridged back by `improve()` — since those build their graphs through `cognify()`. The exception is `remember(content_type="code")`, which runs the separate code-graph pipeline.
- **Scope / limitations**: only the 1-hop neighbourhood of the touched entities is compared; structural edges (`contains`, `is_part_of`, `made_from`, `exists_in`, `contradicts`) and edges with an unnamed endpoint are skipped; the temporal cognify path is not covered.

### Skills (Procedural Memory)
Dataset-scoped `SKILL.md` playbooks agents can discover, load on demand, execute, and improve from run history.

- **Ingest**: `remember(content_type="skills", dataset_name=...)` (folder, file, or inline via `skills_text`/`skill_name`) — requires an explicit dataset; re-ingest upserts (deterministic ids). HTTP: `POST /skills`.
- **Discover**: `SearchType.SKILLS` — one vector search over the `Skill_search_text` collection, no LLM, **metadata-only results** (never the procedure body; progressive disclosure). Requires exactly one dataset; skills outside that dataset's scope, inactive skills, and empty-scope legacy skills are filtered out. Missing collection returns `[]`, not an error.
- **Skill gate**: `recall()` runs a deterministic regex gate (`cognee/api/v1/recall/skill_gate.py`); procedural-sounding queries trigger a concurrent SKILLS lookup whose hits are appended tagged `source="skills"`. Additive and fail-safe; only fires when exactly one dataset is targeted. Disable with `SKILL_GATE_ENABLED=false`.
- **Execute**: `SearchType.AGENTIC_COMPLETION` with `skills=[...]` — LLM sees name+description, loads bodies via the `load_skill` tool (12k char cap).
- **Improve**: `SkillRun` records (via `remember()` skill-run entries) feed LLM-drafted `SkillImprovementProposal`s; preview then apply by proposal id (`/proposals` router).

### Code Files (cognify CODE route)
Supported code files (`.py`, `.go`, `.ts`, `.java`, `.rs`, … — the extension list lives on `code_loader`) are recognized at add time through the loader system: the code loader claims the file, stores it under its real extension, and `ingest_data` tags the record with `system_metadata = {"source": "code"}`. Cognify then routes such items down the CODE route, which runs the deterministic enola code graph pipeline per file — typed `CodeSymbol`/`CodeModule`/… nodes with `calls`/`imports`/`has_method` edges, **no LLM calls**.

- **Search**: code is searchable through `SearchType.CODE` only (deterministic graph operations via `code_query`: `query_facts`, `explore`, `traverse`, `find_path`, `impact_analysis`, `insights`, `architecture`, `delta`). Completion/chunk search types (`GRAPH_COMPLETION`, `CHUNKS`, `RAG_COMPLETION`) do not cover code — the route produces no chunks and no embeddings.
- **Diagrams**: add `"diagram": "mermaid"` (or `"dot"`, or `True`) to any `code_query` and the result carries a `diagram` block with deterministic diagram source (nodes shaped by kind, one subgraph per repository, seeds/focus/path highlighted). `{"operation": "architecture"}` is the module-level overview — symbol-to-symbol edges are rolled up into counted module-to-module edges, routes/storage/services hang off their modules — and it draws itself as Mermaid by default. Renderer: `cognee/modules/retrieval/code_graph_diagram.py`; no LLM, no network. Same option over REST (`code_query` on `POST /api/v1/search` and `/api/v1/recall` with `scope=["code"]`) and the CLI: `cognee-cli search "" -t CODE --code-query '{"operation": "architecture"}' --diagram-out arch.html` (`.html` renders Mermaid in a browser, `.svg/.png/.pdf` run Graphviz on DOT, other extensions get raw source; `--diagram mermaid|dot` prints the source in a fenced block).
- **enola version**: pinned (with per-platform SHA-256) in `cognee/tasks/code_graph/install_enola.py` and auto-installed to `~/.cognee/bin` on first use (`ENOLA_AUTO_INSTALL=false` opts out; `ENOLA_PATH` always wins). Cognee reads enola's documented snapshot contract (`facts.jsonl`, `insights.json`, `receipt.json`; `format_version` 1) and rejects a receipt with a format version it does not understand. Fact ids and resolved relation `target_id`s from the writer are used when present; explainer findings become `CodeInsight` nodes with `evidences` edges; the receipt's provenance/quality block is stamped on the `CodeRepository` node and reported by the `delta` operation. Bumping the pin means re-pinning the checksums and re-checking the known answers in `cognee/tests/test_code_graph_e2e.py`.
- **Opt-out per add**: `preferred_loaders={"text_loader": {}}` treats a code file as a plain document (chunking + LLM extraction).
- **Whole repositories**: `remember(content_type="code")` remains the repo-level path (cross-file edges); the CODE route is per-file.

### Provenance
Cognee has five provenance mechanisms. They answer different questions and are controlled by three unrelated flags — do not confuse them:

| # | Mechanism | Question it answers | Stored where | Flag (default) |
|---|---|---|---|---|
| 1 | Source stamping | who/which run wrote this node | `source_*` fields on the graph node | `COGNEE_PROVENANCE_MODE` (`lightweight`) |
| 2 | Graph source-refs | which documents own this node/edge (drives `forget()` delete/rollback) | source-ref keys on graph nodes/edges | always on (`cognee/infrastructure/databases/provenance/`) |
| 3 | Audit ledger | tamper-evident history for audits | `provenance_entries` table, hash-chained | `PROVENANCE_TRACKING` (**false**) |
| 4 | Memory-provenance projection | who can access what (tenant → user → dataset → data + ACL grants) | computed on request from the relational DB (`GET /v1/schema/provenance`) | n/a |
| 5 | Edge evidence | which document chunk supports this graph edge | `provenance_edge_evidence` table | `EDGE_EVIDENCE_ENABLED` (`true`) |

All three table-backed systems (2, 3, 5) identify a document by the same `make_source_ref_key(dataset_id, data_id)` key.

**Edge evidence** (5) is captured in memory during `add_data_points` and bulk-written once per data item (`EDGE_EVIDENCE_FLUSH_THRESHOLD`, default 10000, forces an earlier flush for huge documents). Search with `include_references=True` returns it as structured `EvidenceReference` objects. Rows are ignored at read time when their pipeline run did not complete or their document is gone, and swept when a document is deleted or its memory dropped with `forget(memory_only=True)`. Scope: only edges extracted from document chunks — contradiction edges, `improve()` enrichment, session bridging, and the code-graph route record no evidence yet (`evidence_kind` is the extension point). Implementation: `cognee/modules/provenance/edge_evidence/`.

### Permissions System
Multi-tenant architecture with users, roles, and Access Control Lists (ACLs):
- Read, write, delete, and share permissions per dataset
- Enable with `ENABLE_BACKEND_ACCESS_CONTROL=True`
- Supports isolated graph/vector databases per user+dataset — backend support varies; see the multi-tenancy support matrix under "Multi-Tenant Access Control" above

### Graph Visualization
Launch visualization server:
```bash
# Via CLI
cognee-cli -ui  # Launches full stack with UI at http://localhost:3000

# Via Python
from cognee.api.v1.visualize import visualization_server
shutdown = visualization_server(port=8080)  # synchronous; returns a shutdown callable
```

## Debugging & Troubleshooting

### Debug Configuration
- Set `LITELLM_LOG="DEBUG"` for verbose LLM logs (default: "ERROR")
- Enable debug mode: `ENV="development"` or `ENV="debug"`
- Disable telemetry: `TELEMETRY_DISABLED=1`
- Check logs in structured format (uses structlog)
- Use `debugpy` optional dependency for debugging: `pip install cognee[debug]`

### Common Issues

**Slow search/recall on default settings**
- Issue: Each answered query on the session path makes one structured-output LLM call for automatic feedback analysis
- Solution: Set `AUTO_FEEDBACK=false` (keep `CACHING=true` so session memory stays on); see "Memory & Performance Tuning Flags"

**Ollama + OpenAI Embeddings NoDataError**
- Issue: Mixing Ollama with OpenAI embeddings can cause errors
- Solution: Configure both LLM and embeddings to use the same provider, or ensure `HUGGINGFACE_TOKENIZER` is set when using Ollama

**LM Studio Structured Output**
- Issue: LM Studio requires explicit instructor mode
- Solution: Set `LLM_INSTRUCTOR_MODE="json_schema_mode"` (or appropriate mode)

**Default Provider Fallback**
- Issue: Configuring only LLM or only embeddings defaults the other to OpenAI
- Solution: Always configure both LLM and embedding providers, or ensure valid OpenAI API key

**Permission Denied on Search**
- Behavior: Returns empty list rather than error (prevents information leakage)
- Solution: Check dataset permissions and user access rights

**Database Connection Issues**
- Check: Verify database URLs, credentials, and that services are running
- Docker users: Use `DB_HOST=host.docker.internal` for local databases

**Rate Limiting Errors**
- Enable client-side rate limiting: `LLM_RATE_LIMIT_ENABLED=true`
- Adjust limits: `LLM_RATE_LIMIT_REQUESTS` and `LLM_RATE_LIMIT_INTERVAL`

## Resources

- [Documentation](https://docs.cognee.ai/)
- [Discord Community](https://discord.gg/NQPKmU5CCg)
- [GitHub Issues](https://github.com/topoteretes/cognee/issues)
- [Example Notebooks](examples/python/)
- [Research Paper](https://arxiv.org/abs/2505.24478) - Optimizing knowledge graphs for LLM reasoning
