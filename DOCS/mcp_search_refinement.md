# Cognee MCP Search Improvements (Nov 12 2025)

This document summarizes the changes made to the MCP server so other developers know what to expect.

## Highlights

1. **API-only Search Proxy** – MCP server continues to talk to the Cognee FastAPI backend only (no direct mode). It now requires `--api-url` and optionally `--api-token` when launched.
2. **Tool Set** – Only three tools are exposed:
   - `list_datasets`: returns dataset id/owner/timestamps + guidance so the LLM always knows the KB catalog.
   - `search`: proxies `/api/v1/search` but now supports dataset IDs, combined context, context-only responses and node filters.
   - `get_dataset_summary`: wrapper around the SUMMARIES search type; lets the LLM preview a KB before querying it.
3. **Instructions** – Server instructions align with the SSE/stdio transports and explain the read-only pattern expected by LibreChat.

## Implementation Notes

- `cognee-mcp/src/tools.py` – reimplemented to keep logic simple and to include richer metadata in outputs. The docstrings now directly address LLM assistants (e.g. “tip: call get_dataset_summary…”).
- `cognee-mcp/src/cognee_client.py` – minimal HTTP wrapper, but now exposes `dataset_ids`, `use_combined_context`, `only_context`, and `node_name` arguments to the `/api/v1/search` endpoint.
- `cognee-mcp/src/test_client.py` – smoke test exercises all three tools: list → summary → combined-context search.
- `cognee-mcp/README.md` – rewritten accordingly (scope, commands, LibreChat config, tool table).

## How To Test

1. Install deps in `cognee-mcp/`: `uv sync --dev` (creates `.venv`).
2. Launch stdio smoke test against a running backend:
   ```bash
   cd cognee-mcp
   .venv/bin/python src/test_client.py --api-url http://<backend-host>:8000
   ```
   The script prints datasets, summaries, and a sample search response.
3. Launch SSE server for LibreChat:
   ```bash
   python src/server.py \
     --transport sse \
     --host 0.0.0.0 \
     --port 8010 \
     --api-url http://<backend-host>:8000
   ```
   Add `serverInstructions: true` in `librechat.yaml` to pull the guidance automatically.

## Next Ideas

- Surface dataset descriptions/sizes when the backend exposes them.
- Add optional arguments to let the LLM request only context (`only_context=True`) for citation-focused flows.
- Consider a `get_dataset_schema` helper if Cognee ever exposes ontology metadata.

## Frontend Alignment (Dashboard)

- The dashboard search panel now mirrors the MCP tooling. Users can select datasets, toggle combined/context-only modes, and specify node filters before issuing a query. These flags map directly to `/api/v1/search` and should be used the same way when configuring LibreChat.
- Dataset/file deletions initiated from the UI call `/api/v1/delete?mode=hard`, ensuring embeddings and graph entities are removed. The confirmation modals explain that behavior; surface the same warning in any automation (e.g., n8n).
- Notebook and Cloud UI are gated behind `NEXT_PUBLIC_ENABLE_NOTEBOOKS` / `NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR` to keep local deployments lean. Set to `"true"` to restore the upstream experience.
