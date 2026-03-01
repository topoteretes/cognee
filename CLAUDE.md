# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognee is an open-source AI memory platform that transforms raw data into persistent knowledge graphs for AI agents. It replaces traditional RAG (Retrieval-Augmented Generation) with an ECL (Extract, Cognify, Load) pipeline combining vector search, graph databases, and LLM-powered entity extraction.

**Requirements**: Python 3.9 - 3.12

## Development Commands

### Setup
```bash
# Create virtual environment (recommended: uv)
uv venv && source .venv/bin/activate

# Install with pip, poetry, or uv
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install with specific extras
uv pip install -e ".[postgres,neo4j,docs,chromadb]"

# Set up pre-commit hooks
pre-commit install
```

### Available Installation Extras
- **postgres** / **postgres-binary** - PostgreSQL + PGVector support
- **neo4j** - Neo4j graph database support
- **neptune** - AWS Neptune support
- **chromadb** - ChromaDB vector database
- **docs** - Document processing (unstructured library)
- **scraping** - Web scraping (Tavily, BeautifulSoup, Playwright)
- **langchain** - LangChain integration
- **llama-index** - LlamaIndex integration
- **anthropic** - Anthropic Claude models
- **gemini** - Google Gemini models
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
- **docling** - Docling document processing
- **codegraph** - Code graph extraction
- **evals** - Evaluation tools
- **deepeval** - DeepEval testing framework
- **posthog** - PostHog analytics
- **monitoring** - Sentry + Langfuse observability
- **distributed** - Modal distributed execution
- **dev** - All development tools (pytest, mypy, ruff, etc.)
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

# Type checking with mypy
mypy cognee/

# Run pylint
pylint cognee/
```

### Running Cognee
```bash
# Using Python SDK
python examples/python/simple_example.py

# Using CLI
cognee-cli add "Your text here"
cognee-cli cognify
cognee-cli search "Your query"
cognee-cli delete --all

# Launch full stack with UI
cognee-cli -ui
```

## Architecture Overview

### Core Workflow: add → cognify → search/memify

1. **add()** - Ingest data (files, URLs, text) into datasets
2. **cognify()** - Extract entities/relationships and build knowledge graph
3. **search()** - Query knowledge using various retrieval strategies
4. **memify()** - Enrich graph with additional context and rules

### Key Architectural Patterns

#### 1. Pipeline-Based Processing
All data flows through task-based pipelines (`cognee/modules/pipelines/`). Tasks are composable units that can run sequentially or in parallel. Example pipeline tasks: `classify_documents`, `extract_graph_from_data`, `add_data_points`.

#### 2. Interface-Based Database Adapters
Multiple backends are supported through adapter interfaces:
- **Graph**: Kuzu (default), Neo4j, Neptune via `GraphDBInterface`
- **Vector**: LanceDB (default), ChromaDB, PGVector via `VectorDBInterface`
- **Relational**: SQLite (default), PostgreSQL

Key files:
- `cognee/infrastructure/databases/graph/graph_db_interface.py`
- `cognee/infrastructure/databases/vector/vector_db_interface.py`

#### 3. Multi-Tenant Access Control
User → Dataset → Data hierarchy with permission-based filtering. Enable with `ENABLE_BACKEND_ACCESS_CONTROL=True`. Each user+dataset combination can have isolated graph/vector databases (when using supported backends: Kuzu, LanceDB, SQLite, Postgres).

### Layer Structure

```
API Layer (cognee/api/v1/)
    ↓
Main Functions (add, cognify, search, memify)
    ↓
Pipeline Orchestrator (cognee/modules/pipelines/)
    ↓
Task Execution Layer (cognee/tasks/)
    ↓
Domain Modules (graph, retrieval, ingestion, etc.)
    ↓
Infrastructure Adapters (LLM, databases)
    ↓
External Services (OpenAI, Kuzu, LanceDB, etc.)
```

### Critical Data Flow Paths

#### ADD: Data Ingestion
`add()` → `resolve_data_directories` → `ingest_data` → `save_data_item_to_storage` → Create Dataset + Data records in relational DB

Key files: `cognee/api/v1/add/add.py`, `cognee/tasks/ingestion/ingest_data.py`

#### COGNIFY: Knowledge Graph Construction
`cognify()` → `classify_documents` → `extract_chunks_from_documents` → `extract_graph_from_data` (LLM extracts entities/relationships using Instructor) → `summarize_text` → `add_data_points` (store in graph + vector DBs)

Key files:
- `cognee/api/v1/cognify/cognify.py`
- `cognee/tasks/graph/extract_graph_from_data.py`
- `cognee/tasks/storage/add_data_points.py`

#### SEARCH: Retrieval
`search(query_text, query_type)` → route to retriever type → filter by permissions → return results

Available search types (from `cognee/modules/search/types/SearchType.py`):
- **GRAPH_COMPLETION** (default) - Graph traversal + LLM completion
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
LLM_MODEL="openai/gpt-4o-mini"  # Default model
```

**Important**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both to avoid unexpected defaults.

Default databases (no extra setup needed):
- **Relational**: SQLite (metadata and state storage)
- **Vector**: LanceDB (embeddings for semantic search)
- **Graph**: Kuzu (knowledge graph and relationships)

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
Supported: lancedb (default), pgvector, chromadb, qdrant, weaviate, milvus
```bash
# ChromaDB (requires chromadb extra)
VECTOR_DB_PROVIDER=chromadb

# PGVector (requires postgres extra)
VECTOR_DB_PROVIDER=pgvector
VECTOR_DB_URL=postgresql://cognee:cognee@localhost:5432/cognee_db
```

#### Graph Databases
Supported: kuzu (default), neo4j, neptune, kuzu-remote
```bash
# Neo4j (requires neo4j extra: pip install cognee[neo4j])
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://localhost:7687
GRAPH_DATABASE_NAME=neo4j
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=yourpassword

# Remote Kuzu
GRAPH_DATABASE_PROVIDER=kuzu-remote
GRAPH_DATABASE_URL=http://localhost:8000
GRAPH_DATABASE_USERNAME=your_username
GRAPH_DATABASE_PASSWORD=your_password
```

### LLM Provider Configuration

Supported providers: OpenAI (default), Azure OpenAI, Google Gemini, Anthropic, AWS Bedrock, Ollama, LM Studio, Custom (OpenAI-compatible APIs)

#### OpenAI (Recommended - Minimal Setup)
```bash
LLM_API_KEY="your_openai_api_key"
LLM_MODEL="openai/gpt-4o-mini"  # or gpt-4o, gpt-4-turbo, etc.
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

#### Google Gemini (requires gemini extra)
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
LLM_MODEL="openrouter/google/gemini-2.0-flash-lite-preview-02-05:free"
LLM_ENDPOINT="https://openrouter.ai/api/v1"
LLM_API_KEY="your_api_key"
```

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
# Use Instructor (default, via litellm)
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
```

Implementation: `cognee/modules/ontology/`

## Branching Strategy

**IMPORTANT**: Always branch from `dev`, not `main`. The `dev` branch is the active development branch.

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

## Code Style

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **String quotes**: Use double quotes `"` not single quotes `'` (enforced by ruff-format)
- **Pre-commit hooks**: Run ruff linting and formatting automatically
- **Type hints**: Encouraged (mypy checks enabled)
- **Important**: Always run `pre-commit run --all-files` before committing to catch formatting issues

## Testing Strategy

Tests are organized in `cognee/tests/`:
- `unit/` - Unit tests for individual modules
- `integration/` - Full pipeline integration tests
- `cli_tests/` - CLI command tests
- `tasks/` - Task-specific tests

When adding features, add corresponding tests. Integration tests should cover the full add → cognify → search flow.

## API Structure

FastAPI application with versioned routes under `cognee/api/v1/`:
- `/add` - Data ingestion
- `/cognify` - Knowledge graph processing
- `/search` - Query interface
- `/memify` - Graph enrichment
- `/datasets` - Dataset management
- `/users` - Authentication (if `REQUIRE_AUTHENTICATION=True`)
- `/visualize` - Graph visualization server

## Python SDK Entry Points

Main functions exported from `cognee/__init__.py`:
- `add(data, dataset_name)` - Ingest data
- `cognify(datasets)` - Build knowledge graph
- `search(query_text, query_type)` - Query knowledge
- `memify(extraction_tasks, enrichment_tasks)` - Enrich graph
- `delete(data_id)` - Remove data
- `config()` - Configuration management
- `datasets()` - Dataset operations

All functions are async - use `await` or `asyncio.run()`.

## Security Considerations

Several security environment variables in `.env`:
- `ACCEPT_LOCAL_FILE_PATH` - Allow local file paths (default: True)
- `ALLOW_HTTP_REQUESTS` - Allow HTTP requests from Cognee (default: True)
- `ALLOW_CYPHER_QUERY` - Allow raw Cypher queries (default: True)
- `REQUIRE_AUTHENTICATION` - Enable API authentication (default: False)
- `ENABLE_BACKEND_ACCESS_CONTROL` - Multi-tenant isolation (default: True)

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
from cognee.infrastructure.databases.vector import get_vector_engine

graph_engine = await get_graph_engine()
vector_engine = await get_vector_engine()
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
await cognee.add(data, dataset_name="my_project")
await cognee.cognify(datasets=["my_project"])
```

### DataPoints
Atomic knowledge units that form the foundation of graph structures. All graph nodes extend the `DataPoint` base class with versioning and metadata support.

### Permissions System
Multi-tenant architecture with users, roles, and Access Control Lists (ACLs):
- Read, write, delete, and share permissions per dataset
- Enable with `ENABLE_BACKEND_ACCESS_CONTROL=True`
- Supports isolated databases per user+dataset (Kuzu, LanceDB, SQLite, Postgres)

### Graph Visualization
Launch visualization server:
```bash
# Via CLI
cognee-cli -ui  # Launches full stack with UI at http://localhost:3000

# Via Python
from cognee.api.v1.visualize import start_visualization_server
await start_visualization_server(port=8080)
```

## Debugging & Troubleshooting

### Debug Configuration
- Set `LITELLM_LOG="DEBUG"` for verbose LLM logs (default: "ERROR")
- Enable debug mode: `ENV="development"` or `ENV="debug"`
- Disable telemetry: `TELEMETRY_DISABLED=1`
- Check logs in structured format (uses structlog)
- Use `debugpy` optional dependency for debugging: `pip install cognee[debug]`

### Common Issues

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
