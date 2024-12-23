# cognee MCP server

### Installing via Smithery

To install Cognee for Claude Desktop automatically via [Smithery](https://smithery.ai/server/cognee):

```bash
npx -y @smithery/cli install cognee --client claude
```

### Installing Manually
A MCP server project

Create a boilerplate server:

```jsx
uvx create-mcp-server
```

1. The command will ask you to name your server, e.g. mcp_cognee


2. Answer “Y” to connect with Claude
Then run

```jsx
cd mcp_cognee
uv sync --dev --all-extras
```

Activate the venv with

```jsx
source .venv/bin/activate
```

This should already add the new server to your Claude config, but if not, add these lines manually:

```
"mcpcognee": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/your_username/mcp/mcp_cognee",
        "run",
        "mcpcognee"
      ],
      "env": {
        "ENV": "local",
        "TOKENIZERS_PARALLELISM": "false",
        "LLM_API_KEY": "add_your_api_key_here",
        "GRAPH_DATABASE_PROVIDER": "neo4j",
        "GRAPH_DATABASE_URL": "bolt://localhost:7687",
        "GRAPH_DATABASE_USERNAME": "add_username_here",
        "GRAPH_DATABASE_PASSWORD": "add_pwd_here",
        "VECTOR_DB_PROVIDER": "lancedb",
        "DB_PROVIDER": "sqlite",
        "DB_NAME": "postgres"
      }
    }
```

Then, edit the pyproject.toml in your new folder so that it includes packages from the cognee requirements. Use the pyproject.toml in your cognee library for this, but match the syntax of the automatically generated pyproject.toml so that it is compatible with uv.

Define cognify tool in server.py
Restart your Claude desktop.