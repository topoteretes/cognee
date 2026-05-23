# Cognee on Coolify

Coolify can deploy Cognee directly from the repository's Docker Compose stack.

## Recommended setup

- **Resource type:** Docker Compose
- **Repository root:** the repo root
- **Compose file:** `docker-compose.yml`
- **Public service:** `cognee`
- **Public port:** `8000`
- **Optional services:** `postgres`, `neo4j`, `redis`, and `frontend` via compose profiles
- **Do not expose:** `5678` in production unless you specifically need the debugger

## Environment variables

At a minimum, set:

- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_PROVIDER`

If you use the built-in Postgres service, also set:

- `DB_PROVIDER=postgres`
- `DB_HOST=postgres`
- `DB_PORT=5432`
- `DB_USERNAME=cognee`
- `DB_PASSWORD=<your postgres password>`
- `DB_NAME=cognee_db`

If you use another vector store, set `VECTOR_DB_PROVIDER` to the provider you want.

For a public deployment, set `CORS_ALLOWED_ORIGINS` to your real frontend or dashboard domain instead of `*`.

## Notes for Coolify

- Coolify can route to the `cognee` service on port `8000` and handle TLS separately.
- If you want the UI too, enable the `ui` profile and expose the `frontend` service on port `3000`.
- For a production-style setup, prefer the `postgres` profile over the default file-based databases.
