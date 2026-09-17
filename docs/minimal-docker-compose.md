# Minimal docker-compose for a local try-out

Try the Cognee API server with a single copy-pasteable file — no cloning, no
building. It uses the prebuilt [`cognee/cognee`](https://hub.docker.com/r/cognee/cognee)
image with the default local databases (SQLite, LanceDB, Ladybug), so the only
thing you need to provide is an LLM API key.

## Prerequisites

- Docker with the Compose plugin (Docker Desktop, Colima, or any OCI-compatible
  runtime — see [Docker & Colima Setup](docker-colima-setup.md))
- An OpenAI API key (the default LLM and embedding provider)

## 1. Save this as `docker-compose.yml` in an empty directory

```yaml
services:
  cognee:
    image: cognee/cognee:main
    ports:
      - "8000:8000"
    environment:
      LLM_API_KEY: ${LLM_API_KEY:?set LLM_API_KEY to your OpenAI API key}
      # Single-user try-out: no auth, shared local databases.
      # Remove this line (or set it to true) for multi-tenant mode,
      # which requires authentication on every API call.
      ENABLE_BACKEND_ACCESS_CONTROL: "false"
```

## 2. Start it

```bash
export LLM_API_KEY="sk-..."   # your OpenAI API key
docker compose up
```

## 3. Verify it works

```bash
curl http://localhost:8000/health
```

Then open <http://localhost:8000/docs> for the interactive API reference and
send your first requests:

```bash
# Ingest a text file
echo "Cognee turns documents into AI memory." > note.txt
curl -X POST http://localhost:8000/api/v1/add \
  -F "data=@note.txt" \
  -F "datasetName=main_dataset"

# Build the knowledge graph
curl -X POST http://localhost:8000/api/v1/cognify \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["main_dataset"]}'

# Search it
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"searchType": "GRAPH_COMPLETION", "query": "What does Cognee do?", "datasets": ["main_dataset"]}'
```

## Keeping data across restarts

The minimal file above stores everything inside the container, so removing the
container removes your data. To persist it, point Cognee's data directories at
a named volume:

```yaml
services:
  cognee:
    image: cognee/cognee:main
    ports:
      - "8000:8000"
    environment:
      LLM_API_KEY: ${LLM_API_KEY:?set LLM_API_KEY to your OpenAI API key}
      ENABLE_BACKEND_ACCESS_CONTROL: "false"
      DATA_ROOT_DIRECTORY: /cognee-data/data
      SYSTEM_ROOT_DIRECTORY: /cognee-data/system
    volumes:
      - cognee_data:/cognee-data

volumes:
  cognee_data:
```

## Going further

- **Other LLM providers** (Anthropic, Gemini, Ollama, …): add the matching
  `LLM_PROVIDER` / `LLM_MODEL` / `LLM_ENDPOINT` variables — see
  [`.env.template`](../.env.template) for the full list.
- **UI, MCP server, Postgres, Neo4j**: the repository's
  [`docker-compose.yml`](../docker-compose.yml) provides these as opt-in
  profiles — see [Run with Docker](../README.md#run-with-docker) in the README.
- **Production**: multi-tenant mode (`ENABLE_BACKEND_ACCESS_CONTROL=true`, the
  default) requires authentication and isolates data per user and dataset.
  Review the security variables in [`.env.template`](../.env.template) before
  exposing the API.
