---
name: cognee-docker
description: Use when the user wants to run cognee with Docker or docker compose — trying it out from the prebuilt image, starting the API server in a container, or bringing up the full stack (UI, MCP, Postgres, Neo4j) with compose profiles.
---

# Start cognee from the Docker image

## Fastest path: prebuilt image, one file

For a local try-out, do NOT clone or build anything. Follow
`docs/minimal-docker-compose.md`: save this as `docker-compose.yml` in an
empty directory:

```yaml
services:
  cognee:
    image: cognee/cognee:main
    ports:
      - "8000:8000"
    environment:
      LLM_API_KEY: ${LLM_API_KEY:?set LLM_API_KEY to your OpenAI API key}
      # Single-user try-out: no auth, shared local databases.
      ENABLE_BACKEND_ACCESS_CONTROL: "false"
```

Then:

```bash
export LLM_API_KEY="sk-..."   # OpenAI key (default LLM + embedding provider)
docker compose up
curl http://localhost:8000/health
```

Interactive API reference: http://localhost:8000/docs. First requests:

```bash
echo "Cognee turns documents into AI memory." > note.txt
curl -X POST http://localhost:8000/api/v1/add -F "data=@note.txt" -F "datasetName=main_dataset"
curl -X POST http://localhost:8000/api/v1/cognify -H "Content-Type: application/json" -d '{"datasets": ["main_dataset"]}'
curl -X POST http://localhost:8000/api/v1/search -H "Content-Type: application/json" -d '{"searchType": "GRAPH_COMPLETION", "query": "What does Cognee do?", "datasets": ["main_dataset"]}'
```

Data lives inside the container by default. To persist it, set
`DATA_ROOT_DIRECTORY=/cognee-data/data` and
`SYSTEM_ROOT_DIRECTORY=/cognee-data/system` and mount a named volume at
`/cognee-data` (full example in `docs/minimal-docker-compose.md`).

## Full stack from the repo

The repository's `docker-compose.yml` builds from source and adds opt-in
profiles. From the repo root (needs a `.env` with at least `LLM_API_KEY`;
copy `.env.template`):

```bash
docker compose up                                  # API server only, port 8000
docker compose --profile ui up                     # + frontend on port 3000
docker compose --profile mcp up                    # + MCP server on port 8001
docker compose --profile postgres --profile neo4j up   # + databases
```

Postgres profile: pgvector/pg17, user/password/db `cognee`/`cognee`/`cognee_db`
on 5432. Neo4j profile: `neo4j`/`pleaseletmein` on 7474/7687. When cognee runs
in a container and the database on the host, use `DB_HOST=host.docker.internal`.

## Gotchas

- With `ENABLE_BACKEND_ACCESS_CONTROL` unset (defaults to true), every API
  call requires authentication — the single-user try-out sets it to `false`.
- The image defaults to OpenAI for both LLM and embeddings; configuring only
  one of them leaves the other on OpenAI, so keep a valid OpenAI key or
  configure both (see the cognee-integrations skill).
