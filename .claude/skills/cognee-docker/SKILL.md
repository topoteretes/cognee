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
# remember = ingest + build the graph in one call (multipart form)
curl -X POST http://localhost:8000/api/v1/remember -F "data=@note.txt" -F "datasetName=main_dataset"
# recall = query it (JSON)
curl -X POST http://localhost:8000/api/v1/recall -H "Content-Type: application/json" \
  -d '{"query": "What does Cognee do?", "datasets": ["main_dataset"]}'
```

`/api/v1/recall` takes the question as `query`. Omit `search_type` (or pass
`null`) and the query is auto-routed by the same rule-based router the SDK
`recall()` uses, with `HYBRID_COMPLETION` as the fallback; pass a value such as
`"search_type": "GRAPH_COMPLETION"` to pin a strategy. The rule table is in
`docs/recall-vs-search.md`.

Request DTOs accept both `snake_case` and `camelCase` for every field
(`alias_generator=to_camel` + `populate_by_name` in `cognee/api/DTO.py`), so
`search_type` and `searchType` are equally valid.

The legacy `/api/v1/add` + `/api/v1/cognify` + `/api/v1/search` endpoints still
exist and are what `remember`/`recall` call underneath; use them only when you
need a single stage on its own. `/api/v1/improve` and `/api/v1/forget` complete
the memory API.

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
