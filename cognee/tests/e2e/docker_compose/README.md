# Docker Compose full-stack e2e

A real end-to-end test for the `docker-compose.yml` deployment. It replaces the
old `up -d → sleep 30 → down` placeholder in
[`.github/workflows/docker_compose.yml`](../../../../.github/workflows/docker_compose.yml).

## What it covers

| Test | Asserts |
| --- | --- |
| `test_golden_flow_api` | `health → login → add → datasets → data` against the API on `:8000` (cognify + search when an LLM key is provided). |
| `test_mcp_health_and_tool_call` | `cognee-mcp` `/health` on `:8001` **and** one real MCP tool call (`list_datasets_json`) over SSE. |
| `test_postgres_persistence_across_recreate` | Data added via the API survives a Postgres container **force-recreate** — proving the `postgres_data` volume is required. |
| `test_service_logs_are_traceback_free` | No service emitted an unhandled Python traceback. |

The LLM-dependent leg is **off by default**, so the PR-blocking run never calls a
real model ("mock LLM by default").

## Run it locally

```bash
# 1. Bring the stack up with the same profiles CI uses.
cp .env.template .env            # set LLM_API_KEY only if you want the LLM leg
docker compose --profile postgres --profile mcp up -d --build

# 2. Run the suite (compose-driving tests need COGNEE_E2E_MANAGE_COMPOSE=1).
COGNEE_E2E_MANAGE_COMPOSE=1 \
  uv run --no-project --with pytest --with requests --with mcp \
  python -m pytest cognee/tests/e2e/docker_compose -v
```

## Configuration (all via env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `COGNEE_API_URL` | `http://localhost:8000` | Main API base URL. |
| `COGNEE_MCP_URL` | `http://localhost:8001` | MCP service base URL. |
| `COGNEE_E2E_RUN_LLM` | `0` | Run the cognify/search leg (needs a real LLM key). |
| `COGNEE_E2E_MANAGE_COMPOSE` | `0` | Allow the suite to drive `docker compose` (persistence + log tests). |
| `COGNEE_E2E_COMPOSE_PROFILES` | `postgres,mcp` | Profiles passed to `docker compose`. |
| `COGNEE_E2E_STARTUP_TIMEOUT` | `300` | Seconds to wait for a service to become healthy. |
