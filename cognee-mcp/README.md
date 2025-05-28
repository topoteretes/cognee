# cognee MCP server

### Installing Manually
A MCP server project
=======
1. Clone the [cognee](https://github.com/topoteretes/cognee) repo

2. Install dependencies

```
brew install uv
```

```jsx
cd cognee-mcp
uv sync --dev --all-extras --reinstall
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
mcp dev src/server.py
```
Open inspector with timeout passed:
```
http://localhost:5173?timeout=120000
```

To apply new changes while developing cognee you need to do:

1. `poetry lock` in cognee folder
2. `uv sync --dev --all-extras --reinstall`
3. `mcp dev src/server.py`

### Development
In order to use local cognee build, run in root of the cognee repo:
```bash
poetry build -o ./cognee-mcp/sources
```
After the build process is done, change the cognee library dependency inside the `cognee-mcp/pyproject.toml` from
```toml
cognee[postgres,codegraph,gemini,huggingface]==0.1.38
```
to
```toml
cognee[postgres,codegraph,gemini,huggingface]
```
After that add the following snippet to the same file (`cognee-mcp/pyproject.toml`).
```toml
[tool.uv.sources]
cognee = { path = "sources/cognee-0.1.38-py3-none-any.whl" }
```


## Resources 

<details>
<summary><b>Open MCP Marketplace API Support</b></summary>

![MCP Marketplace User Review Rating Badge](http://www.deepnlp.org/api/marketplace/svg?topoteretes/cognee)|[Reviews](http://www.deepnlp.org/store/ai-agent/mcp-server/pub-topoteretes/cognee)|[GitHub](https://github.com/AI-Agent-Hub/mcp-marketplace)|[Doc](http://www.deepnlp.org/doc/mcp_marketplace)|[MCP Marketplace](http://www.deepnlp.org/store/ai-agent/mcp-server)

Allow AI/Agent/LLM to find this MCP Server via common python/typescript API, search and explore relevant servers and tools

***Example: Search Server and Tools***
```python
import anthropic
import mcp_marketplace as mcpm

result_q = mcpm.search(query="cognee", mode="list", page_id=0, count_per_page=100, config_name="deepnlp") # search server by category choose various endpoint
result_id = mcpm.search(id="topoteretes/cognee", mode="list", page_id=0, count_per_page=100, config_name="deepnlp")      # search server by id choose various endpoint 
tools = mcpm.list_tools(id="topoteretes/cognee", config_name="deepnlp_tool")
# Call Claude to Choose Tools Function Calls 
# client = anthropic.Anthropic()
# response = client.messages.create(model="claude-opus-4-20250514", max_tokens=1024, tools=tools, messages=[])
```

</details>

