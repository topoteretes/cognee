---
name: cognee-integrations
description: Use when the user wants to connect cognee to external services — switching LLM or embedding providers (OpenAI, Azure, Gemini, Anthropic, Ollama, OpenRouter), changing databases (Postgres, PGVector, Neo4j, Neptune, Turso), S3 storage, or the MCP server for IDE integration.
---

# Set up cognee integrations

All integration config is environment variables (`.env`). The authoritative,
always-current list with commented examples is `.env.template` at the repo
root — check it before inventing variable names. Install the matching extra
before switching a backend (e.g. `pip install cognee[postgres]`).

## LLM providers

Default is OpenAI (`LLM_API_KEY` is all you need). To switch, set
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and (where relevant)
`LLM_ENDPOINT` / `LLM_API_VERSION`:

- **Azure OpenAI**: `LLM_PROVIDER=azure`, `LLM_MODEL=azure/gpt-4o-mini`, endpoint + api version required.
- **Gemini** (no extra needed): `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini/gemini-2.0-flash-exp`.
- **Anthropic** (`cognee[anthropic]`): `LLM_PROVIDER=anthropic`, model e.g. `claude-3-5-sonnet-20241022`.
- **Ollama, local** (`cognee[ollama]`): `LLM_PROVIDER=ollama`, `LLM_ENDPOINT=http://localhost:11434/v1`, and set the embedding block + `HUGGINGFACE_TOKENIZER` too.
- **Custom / OpenRouter / vLLM**: `LLM_PROVIDER=custom` with the provider's OpenAI-compatible endpoint.
- **AWS Bedrock** (`cognee[aws]`): `LLM_PROVIDER=bedrock` + AWS credentials/region.

**The classic trap**: LLM and embeddings are configured independently
(`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_ENDPOINT`,
`EMBEDDING_API_KEY`). Configuring only one leaves the other on OpenAI —
either keep a valid OpenAI key or configure both.

## Databases

- **Relational** (`DB_PROVIDER`): sqlite (default) or postgres
  (`cognee[postgres]`; host/port/user/password/name via `DB_*` vars).
- **Vector** (`VECTOR_DB_PROVIDER`): lancedb (default), pgvector
  (`cognee[postgres]`, needs `VECTOR_DB_URL`), neptune_analytics
  (`cognee[neptune]`), turso (`cognee[turso]`). Anything else (ChromaDB,
  Qdrant, Weaviate, Milvus, …) lives in community adapters — install from
  https://github.com/topoteretes/cognee-community and register with
  `use_vector_adapter` before use; setting `VECTOR_DB_PROVIDER` alone raises
  "Unsupported vector database provider".
- **Graph** (`GRAPH_DATABASE_PROVIDER`): ladybug (default), neo4j
  (`cognee[neo4j]`, bolt URL + credentials), neptune (`cognee[neptune]`),
  ladybug-remote, postgres (no raw Cypher / natural-language search).

The repo `docker-compose.yml` ships ready-to-use `postgres` (pgvector) and
`neo4j` profiles with matching default credentials. From a container, reach
host services with `DB_HOST=host.docker.internal`.

## Storage, cache, and the rest

- **S3 storage** (`cognee[aws]`): `STORAGE_BACKEND=s3` + bucket/credentials,
  and point `DATA_ROOT_DIRECTORY`/`SYSTEM_ROOT_DIRECTORY` at `s3://` paths.
- **Session cache**: `CACHE_BACKEND` = sqlite (default) | postgres | redis | fs | tapes.
- **Ontologies**: `ONTOLOGY_FILE_PATH` to an OWL file, resolver/matching via
  `ONTOLOGY_RESOLVER` / `MATCHING_STRATEGY`.

## MCP server (IDE integration)

`docker compose --profile mcp up` starts the MCP server on port 8001
(SSE transport), built from `cognee-mcp/`. Point Cursor / Claude Desktop /
Claude Code at it to use cognee memory from the IDE. Configure its `DB_*` env
to match the main service so both see the same data.

## After changing providers mid-project

Embeddings from different models are not comparable — after switching the
embedding provider or model, reset local state (`cognee-cli forget --all` or
`await cognee.forget(everything=True)`) and re-ingest with `remember()`.

To drop just the graph and vectors while keeping the ingested files, use
`await cognee.forget(dataset="my_project", memory_only=True)` — the dataset can
then be rebuilt under the new embedding model without re-uploading anything.
