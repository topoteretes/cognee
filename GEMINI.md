# Cognee - Knowledge Engine for AI Agents

Cognee is an open-source knowledge engine that enables AI agents to ingest data in any format and provides persistent, evolving memory through vector search and graph databases.

## Project Overview
- **Purpose:** Provide AI agents with a dynamic and personalized memory system.
- **Technologies:** Python 3.10+, LanceDB, Neo4j, PostgreSQL (optional), OpenAI, LiteLLM, FastAPI.
- **Architecture:** Cognitive science-inspired memory management with graph and vector search strategies.

## Building and Running
### Prerequisites
- Python 3.10 to 3.13
- LLM API Key (e.g., OpenAI)

### Setup
1. **Installation:** Use `uv` for fast dependency management.
   ```powershell
   uv pip install -e ".[dev]"
   ```
2. **Environment Configuration:** Create a `.env` file from the template.
   ```powershell
   cp .env.template .env
   ```
   Set `LLM_API_KEY` in your `.env` file.

### Commands
- **CLI Usage:**
  ```powershell
  cognee-cli remember "Some context"
  cognee-cli recall "What was the context?"
  cognee-cli forget --all
  ```
- **Local UI:**
  ```powershell
  cognee-cli -ui
  ```

## Development Conventions
- **Linting & Formatting:** The project uses `ruff`.
  ```powershell
  ruff check .
  ruff format .
  ```
- **Testing:** Uses `pytest`.
  ```powershell
  pytest
  ```
- **Pre-commit:** Configuration is in `.pre-commit-config.yaml`. Ensure it's installed:
  ```powershell
  pre-commit install
  ```
- **Mandates:**
  - Always use PowerShell for shell commands (Windows environment).
  - Explicitly install dependencies (no optional ones assumed).
  - Never stage or commit changes unless explicitly requested.

## Project-Specific Context
- The project is organized with modules in `cognee/modules/`.
- Frontend source is in `cognee-frontend/`.
- MCP server source is in `cognee-mcp/`.
- Distributed processing components are in `distributed/`.
