# Cognee Model Context Protocol (FastMCP) Server

[![Tests Passing](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)]()
[![FastMCP Spec](https://img.shields.io/badge/FastMCP-2024--11--05%20Spec-blue.svg)]()
[![Hackathon Track](https://img.shields.io/badge/Hackathon-Mergetober%20%7C%20Graph%20Hacks%202026-purple.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A production-ready [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server integration for **[Cognee](https://github.com/topoteretes/cognee)**, exposing Cognee's persistent Knowledge Graph and Vector Context Engine directly to AI clients such as **Claude Desktop**, **Cursor IDE**, **Cline**, and autonomous agent frameworks.

---

## 🏛️ Visual Architecture

```
+--------------------------------------------------------------------------+
|                       AI Clients & IDE Integrations                      |
|              (Claude Desktop, Cursor IDE, Cline, LibreChat)              |
+------------------------------------+-------------------------------------+
                                     |
                                     | JSON-RPC (stdio / SSE)
                                     v
+--------------------------------------------------------------------------+
|                         FastMCP Protocol Layer                           |
|                        (cognee.api.mcp.server)                           |
|                                                                          |
|   +-----------------------+ +-------------------------+ +------------+   |
|   |   cognee_add_memory   | |   cognee_search_memory  | |   prune    |   |
|   |  (Ingest & Cognify)   | | (Graph & Vector Search) | |  (Cleanup) |   |
|   +-----------------------+ +-------------------------+ +------------+   |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                       Unified Cognee MCP Adapter                         |
|                        (cognee.api.mcp.adapter)                          |
|                                                                          |
|         [Embedded Mode]                         [Remote API Mode]        |
|    In-process cognee execution                HTTP client via /api/v1/*   |
+--------------------+-------------------------------------+---------------+
                     |                                     |
                     v                                     v
+--------------------------------------------------------------------------+
|                     Cognee Persistent Context Engine                     |
|                                                                          |
|       +-----------------------------+   +----------------------------+   |
|       |     Vector Store Engine     |   |   Relational Graph Engine  |   |
|       |   (LanceDB, PGVector, S3)   |   |     (NetworkX, Kùzu, Neo4j)|   |
|       +-----------------------------+   +----------------------------+   |
+--------------------------------------------------------------------------+
```

---

## 🚀 3-Step Quickstart

### Step 1: Install Dependencies
```bash
pip install mcp pydantic pydantic-settings
```
*(Optional: install full Cognee if using embedded direct-engine mode)*
```bash
pip install cognee
```

### Step 2: Run the MCP Server
```bash
# Standard stdio pipe (for Claude Desktop & Cursor)
python -m cognee.api.mcp.server --transport stdio

# Or Server-Sent Events (SSE) for networked agent microservices
python -m cognee.api.mcp.server --transport sse --host 127.0.0.1 --port 8000
```

### Step 3: Run the Interactive Demo Agent
Run the self-contained demonstration showing memory ingestion, entity extraction, and semantic graph retrieval in real time:
```bash
python -m cognee.api.mcp.demo_agent
# or directly:
python cognee/api/mcp/demo_agent.py
```
> **Note**: Includes a zero-configuration mock fallback so anyone can run the demo immediately without requiring external API keys.

---

## 🧰 Available FastMCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `cognee_add_memory` | `content` (str), `dataset_name` (str), `auto_cognify` (bool) | Ingests raw text, code, or documentation into Cognee storage. When `auto_cognify=True`, automatically triggers graph node & relationship extraction. |
| `cognee_search_memory` | `query` (str), `search_type` (str), `dataset_name` (optional) | Queries memory using hybrid Graph & vector search. Modes: `INSIGHTS` (graph relations), `CHUNKS` (vector chunks), `SUMMARIES`, `COMPLETION`. |
| `cognee_prune_memory` | `prune_all` (bool) | Prunes cached memories (`prune_all=False`) or resets the entire graph and vector memory system (`prune_all=True`). |

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
        "OPENAI_API_KEY": "sk-your-openai-api-key-here",
        "DEFAULT_DATASET": "claude_memory",
        "AUTO_COGNIFY": "true"
      }
    }
  }
}
```

### Config File Locations
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

---

## 💻 Cursor IDE Configuration

In your project root, create `.cursor/mcp.json`:

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
        "OPENAI_API_KEY": "sk-your-openai-api-key-here"
      }
    }
  }
}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport protocol: `stdio` (IDE pipe) or `sse` (network stream) |
| `MCP_HOST` | `127.0.0.1` | SSE host bind address |
| `MCP_PORT` | `8000` | SSE port number |
| `COGNEE_API_URL` | *(None)* | Remote Cognee REST backend URL (enables API proxy mode) |
| `COGNEE_API_KEY` | *(None)* | Bearer authorization token for remote Cognee API |
| `DEFAULT_DATASET` | `default_dataset` | Default dataset namespace for additions |
| `AUTO_COGNIFY` | `true` | Auto-run graph extraction on ingested content |
| `OPENAI_API_KEY` | *(None)* | LLM API key for entity extraction and vector embeddings |

---

## 🧪 Unit Testing

Run the test suite using `pytest`:
```bash
python -m pytest cognee/tests/unit/test_mcp_server.py --noconftest -v
```

All 10 unit tests pass with 100% green coverage across tool validation, error branches, and protocol dispatches.
