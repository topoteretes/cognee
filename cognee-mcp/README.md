<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>
  <br />
  <strong>Cognee MCP – Read‑only Knowledge Search Server</strong>
  <p>Expose your Cognee knowledge bases to any Model Context Protocol client (LibreChat, IDEs, terminals) through a lightweight two‑tool server.</p>
</div>

---

## 1. Overview

This fork focuses the MCP server on a single job: answering search queries against datasets that were already built via the Cognee UI/API. The server is API-only (no direct Python imports) and only exposes two tools:

1. `list_datasets` – enumerate available knowledge bases (IDs, names, created timestamps).
2. `search` – send natural language queries to the Cognee API and return ranked evidence for the calling LLM.

Because the server is read-only, it pairs cleanly with LibreChat, IDE copilots, or custom MCP-aware agents that need GraphRAG-style answers without managing ingestion pipelines.

---

## 2. Architecture & Requirements

- **Cognee API backend**: must be running (FastAPI server). For multi-user isolation, enable `ENABLE_BACKEND_ACCESS_CONTROL=true` on the backend. If you start clean, configure multi-user mode from day one so dataset permissions stay scoped per LibreChat user.
- **Frontend / ingestion**: handled elsewhere (e.g., Cognee UI). The MCP server never mutates data.
- **Transport**: choose `stdio`, `sse`, or `streamable-http`. LibreChat typically uses `sse` or `streamable-http`.
- **Networking**: MCP container needs network access to the Cognee API container (Docker bridge networking recommended on Unraid).

---

## 3. Quick Start (local CLI)

```bash
git clone https://github.com/topoteretes/cognee.git
cd cognee/cognee-mcp
uv sync --dev --all-extras --reinstall
source .venv/bin/activate

# Minimal run against a local API instance
python src/server.py \
  --transport stdio \
  --api-url http://localhost:8000 \
  --api-token YOUR_TOKEN_IF_REQUIRED
```

**Key flags**

| Flag | Description |
| --- | --- |
| `--api-url` | Required. Base URL of the running Cognee API server. |
| `--api-token` | Optional. Bearer token if the API is secured. |
| `--transport` | `stdio` (default), `sse`, or `http` (streamable-http). |
| `--host / --port / --path` | Network binding options for HTTP/SSE transports. |

Environment variables such as `BACKEND_API_TOKEN`, `CORS_ALLOWED_ORIGINS`, etc., still come from `config.py`/`.env`.

---

## 4. Docker & Deployment Notes

You can build or pull the `cognee/cognee-mcp` image. Key differences from the stock README:

- MCP is API-only, so always pass `API_URL` (and optionally `API_TOKEN`).
- Transport selection still uses `TRANSPORT_MODE` env inside Docker (`http`, `sse`, or `stdio`).

```bash
# Example: SSE transport pointing at backend container "cognee-api"
docker run --rm -it \
  --name cognee-mcp \
  --network your_bridge \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee-api:8000 \
  -e API_TOKEN=YOUR_TOKEN \
  -p 8001:8000 \
  cognee/cognee-mcp:main
```

Remember: Docker uses env vars (`API_URL`, `API_TOKEN`, `TRANSPORT_MODE`), while direct Python execution uses CLI flags.

---

## 5. Integrating with LibreChat

Add an entry to `librechat.yaml` under `mcpServers`:

```yaml
mcpServers:
  cognee-search:
    type: sse
    url: http://localhost:8001/mcp
    serverInstructions: true
    headers:
      Authorization: "Bearer ${COGNEE_MCP_TOKEN}"   # optional
```

Best practices:

- Call `list_datasets` once at the beginning of a conversation to discover dataset IDs and owners.
- Use `get_dataset_summary(dataset_id)` when you need a quick preview before committing to a KB.
- `search` accepts dataset IDs (recommended), combined-context flags, and node filters—set them explicitly so the LLM retrieves exactly what the user needs.
- Tune `top_k` (1–50) to balance latency vs. recall. LibreChat’s LLM can interpret the returned evidence to craft the final response.

---

## 6. Available Tools

| Tool | Description | Arguments |
| --- | --- | --- |
| `list_datasets` | Returns all datasets with IDs, owners, and timestamps so the LLM can choose the right KB. | none |
| `search` | Issues a Cognee search request with optional combined-context / context-only outputs. | `query` (required), `datasets` (names) or `dataset_ids` (UUIDs), `search_type`, `top_k` (1–50), `system_prompt`, `use_combined_context`, `only_context`, `node_name` (list of node sets). |
| `get_dataset_summary` | Fetches top `SUMMARIES` entries for a dataset to understand its scope before searching. | `dataset_id` (required), `top_k` (1–5). |

All responses are formatted as plain text blocks so calling LLMs can embed them directly into reasoning steps.

---

## 7. Health Checks & Testing

- Health endpoints (available when running SSE/HTTP transports):
  - `GET /health` – basic status.
  - `GET /health/detailed` – reports backend-access-control mode and log location.
- Smoke test via stdio:

```bash
python src/test_client.py --api-url http://localhost:8000
```

This script launches the MCP server via stdio, lists tools, lists datasets, and runs a sample search.

---

## 8. Development Workflow

1. Install dependencies with `uv sync --dev --all-extras --reinstall`.
2. Run formatters/lint:
   ```bash
   uv run ruff format
   uv run ruff check .
   ```
3. When changing transports or tool definitions, update this README and rerun the smoke test above.

---

## 9. License

Cognee is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.
