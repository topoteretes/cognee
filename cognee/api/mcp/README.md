# Cognee Model Context Protocol (FastMCP) Module

This directory provides the internal FastMCP server integration for **Cognee**, allowing standard MCP clients (Claude Desktop, Cursor IDE, Cline, LibreChat) to communicate directly with Cognee's persistent memory and knowledge graph engine.

---

## 🏛️ Architecture Overview

The module exposes Cognee's internal context and memory engine as standardized MCP tools:

```
┌─────────────────────────────────┐
│ MCP Clients (Claude, Cursor)    │
└────────────────┬────────────────┘
                 │ JSON-RPC (stdio / SSE)
┌────────────────▼────────────────┐
│   cognee.api.mcp.server         │ (FastMCP wrapper)
├─────────────────────────────────┤
│   cognee.api.mcp.adapter        │ (Unified abstraction)
├────────────────┬────────────────┤
│ Embedded Mode  │ Remote API Mode│
│ (cognee.add,   │ (HTTP client   │
│ cognee.cognify,│  -> FastAPI)   │
│ cognee.search) │                │
└────────────────┴────────────────┘
```

---

## 🧰 Available MCP Tools

### 1. `cognee_add_memory`
- **Description**: Ingests notes, documents, or code chunks into Cognee's persistent memory.
- **Parameters**:
  - `content` (`str`, required): The text or document data to store.
  - `dataset_name` (`str`, optional, default: `"default_dataset"`): Namespace in the knowledge graph.
  - `auto_cognify` (`bool`, optional, default: `True`): Extracts entities and relationships into the knowledge graph immediately.
- **Returns**: `AddMemoryResponse` (JSON containing status, dataset name, and cognified flag).

### 2. `cognee_search_memory`
- **Description**: Searches Cognee's persistent memory using graph insights or vector similarity.
- **Parameters**:
  - `query` (`str`, required): Natural language search query.
  - `search_type` (`str`, optional, default: `"INSIGHTS"`): Options: `INSIGHTS`, `CHUNKS`, `SUMMARIES`, `COMPLETION`.
  - `dataset_name` (`str`, optional): Specific dataset filter.
- **Returns**: `SearchMemoryResponse` (JSON containing matched nodes, relationships, and context chunks).

### 3. `cognee_prune_memory`
- **Description**: Prunes memory caches or resets the knowledge graph.
- **Parameters**:
  - `prune_all` (`bool`, optional, default: `False`): If `True`, prunes the entire graph and vector system. If `False`, clears dataset data.

---

## ⚙️ Configuration & Environment

Environment variables can be defined in a `.env` file or exported into the environment:

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport protocol: `stdio` (IDE standard) or `sse` (networked) |
| `MCP_HOST` | `127.0.0.1` | Host interface for SSE server |
| `MCP_PORT` | `8000` | Port for SSE server |
| `DEFAULT_DATASET` | `default_dataset` | Default dataset name for additions |
| `AUTO_COGNIFY` | `true` | Auto-run graph extraction on new memories |
| `COGNEE_API_URL` | *(None)* | Remote Cognee REST backend URL (enables API mode) |
| `COGNEE_API_KEY` | *(None)* | Bearer token for remote Cognee API |
| `OPENAI_API_KEY` | *(None)* | LLM API key used for embeddings and entity extraction |

---

## 🚀 Execution Commands

### Stdio Transport (Default for Claude & Cursor)
```bash
python -m cognee.api.mcp.server --transport stdio
```

### SSE Transport (Web & Network Services)
```bash
python -m cognee.api.mcp.server --transport sse --host 127.0.0.1 --port 8000
```

---

## 💻 Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cognee": {
      "command": "python",
      "args": [
        "-m",
        "cognee.api.mcp.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-key-here",
        "DEFAULT_DATASET": "claude_memory",
        "AUTO_COGNIFY": "true"
      }
    }
  }
}
```

Config File Locations:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
