# Cognee Memory — Minimal MCP Server

A dead-simple MCP server that gives any AI agent persistent memory. **2 tools, zero configuration.**

| Tool | What it does |
|------|-------------|
| `remember` | Store any information — instantly cached + background graph processing |
| `search_memory` | Retrieve relevant memories — hybrid session + knowledge graph search |

## Quick Start

### Cursor / Claude Desktop / Windsurf

Add to your MCP config (e.g. `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "cognee-memory": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/cognee-mcp-simple",
        "run", "cognee-memory"
      ]
    }
  }
}
```

### Environment

Create a `.env` file in the `cognee-mcp-simple/` directory (or set env vars):

```bash
LLM_API_KEY="your-openai-key"
```

That's it. Everything else uses sensible defaults (SQLite + LanceDB + Kuzu, all file-based).

### Optional: Enable session cache for instant recall

```bash
CACHING=true
CACHE_BACKEND=fs   # or "redis" for production
```

## How It Works

### `remember(information)`
1. **Instant** — Writes to session cache (if enabled). Available for search immediately.
2. **Background** — Runs `add → cognify → memify` pipeline. Extracts entities, builds knowledge graph, creates embeddings.

### `search_memory(query)`
1. **Session layer** — Scans recent session entries for keyword matches (fast, recent context).
2. **Graph layer** — Runs `GRAPH_COMPLETION` search (semantic, deep, LLM-powered).
3. **Merges** — Returns combined results from both layers.

## Comparison with `cognee-mcp`

| | `cognee-mcp` (full) | `cognee-mcp-simple` (this) |
|---|---|---|
| Tools | 11 | 2 |
| Target | Power users, developers | Vibe coders, agentic builders |
| Search types | Manual selection required | Automatic (GRAPH_COMPLETION) |
| Memory model | Explicit add → cognify → search | Just remember + search |
| Session cache | Via `save_interaction` | Built into `remember` |
| Background processing | Manual status checks | Fully automatic |

## Development

```bash
cd cognee-mcp-simple
uv sync
uv run cognee-memory          # stdio (default)
uv run cognee-memory --transport sse --port 8000
```
