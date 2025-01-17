# cognee MCP server




### Installing Manually
A MCP server project
=======
1. Clone the [cognee](https://github.com/topoteretes/cognee) repo



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
```
cd ~/Library/Application\ Support/Claude/
```
You need to create claude_desktop_config.json in this folder if it doesn't exist
Make sure to add your paths and LLM API key to the file bellow
Use your editor of choice, for example Nano:
```
nano claude_desktop_config.json
```

```


{
	"mcpServers": {
		"cognee": {
			"command": "/Users/{user}/cognee/.venv/bin/uv",
			"args": [
        "--directory",
        "/Users/{user}/cognee/cognee-mcp",
        "run",
        "cognee"
      ],
      "env": {
        "ENV": "local",
        "TOKENIZERS_PARALLELISM": "false",
        "LLM_API_KEY": "sk-"
      }
		},
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/{user}/Desktop",
        "/Users/{user}/Projects"
      ]
    }
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


To use debugger, run:
```bash
npx @modelcontextprotocol/inspector uv --directory /Users/name/folder run cognee
```

To apply new changes while development you do:

1. Poetry lock in cognee folder
2. uv sync --dev --all-extras --reinstall 
3. npx @modelcontextprotocol/inspector uv --directory /Users/vasilije/cognee/cognee-mcp run cognee


