# cognee MCP server




### Installing Manually
A MCP server project
=======
1. Clone the [cognee](www.github.com/topoteretes/cognee) repo



2. Install dependencies

```
pip install uv
```
```
brew install postgresql
```

```
brew install rust
```

```jsx
cd cognee-mcp
uv sync --dev --all-extras
```

3. Activate the venv with

```jsx
source .venv/bin/activate
```

4. Add the new server to your Claude config:

The file should be located here: ~/Library/Application\ Support/Claude/

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


Restart your Claude desktop.

### Installing via Smithery

To install Cognee for Claude Desktop automatically via [Smithery](https://smithery.ai/server/cognee):

```bash
npx -y @smithery/cli install cognee --client claude
```

Define cognify tool in server.py
Restart your Claude desktop.
