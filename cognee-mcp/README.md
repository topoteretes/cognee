# cognee MCP server

1. Clone the [cognee](www.github.com/topoteretes/cognee) repo


2. Install dependencies

```jsx
cd cognee-mcp
uv sync --dev --all-extras
```

3. Activate the venv with

```jsx
source .venv/bin/activate
```

4. Add the new server to your Claude config:

```json
"cognee": {
  "command": "uv",
  "args": [
    "--directory",
    "/{Absolute path to cognee directory}/cognee-mcp",
    "run",
    "cognee"
  ],
  "env": {
    "ENV": "local",
    "TOKENIZERS_PARALLELISM": "false",
    "LLM_API_KEY": "add_your_api_key_here",
  }
}
```

Then, edit the pyproject.toml in your new folder so that it includes packages from the cognee requirements. Use the pyproject.toml in your cognee library for this, but match the syntax of the automatically generated pyproject.toml so that it is compatible with uv.

Define cognify tool in server.py
Restart your Claude desktop.
