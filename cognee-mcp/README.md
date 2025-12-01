<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  cognee‑mcp - Run cognee’s memory engine as a Model Context Protocol server

  <p align="center">
  <a href="https://www.youtube.com/watch?v=1bezuvLwJmw&t=2s">Demo</a>
  .
  <a href="https://cognee.ai">Learn more</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Join Discord</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">Join r/AIMemory</a>
  </p>


  [![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)
  [![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/stargazers/)
  [![GitHub commits](https://badgen.net/github/commits/topoteretes/cognee)](https://GitHub.com/topoteretes/cognee/commit/)
  [![Github tag](https://badgen.net/github/tag/topoteretes/cognee)](https://github.com/topoteretes/cognee/tags/)
  [![Downloads](https://static.pepy.tech/badge/cognee)](https://pepy.tech/project/cognee)
  [![License](https://img.shields.io/github/license/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/blob/main/LICENSE)
  [![Contributors](https://img.shields.io/github/contributors/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/graphs/contributors)

<a href="https://www.producthunt.com/posts/cognee?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-cognee" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=946346&theme=light&period=daily&t=1744472480704" alt="cognee - Memory&#0032;for&#0032;AI&#0032;Agents&#0032;&#0032;in&#0032;5&#0032;lines&#0032;of&#0032;code | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<a href="https://trendshift.io/repositories/13955" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13955" alt="topoteretes%2Fcognee | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>


Build memory for Agents and query from any client that speaks MCP – in your terminal or IDE.

</div>

## ✨ Features

- Multiple transports – choose Streamable HTTP --transport http (recommended for web deployments), SSE --transport sse (real‑time streaming), or stdio (classic pipe, default)
- **API Mode** – connect to an already running Cognee FastAPI server instead of using cognee directly (see [API Mode](#-api-mode) below)
- Integrated logging – all actions written to a rotating file (see get_log_file_location()) and mirrored to console in dev
- Local file ingestion – feed .md, source files, Cursor rule‑sets, etc. straight from disk
- Background pipelines – long‑running cognify & codify jobs spawn off‑thread; check progress with status tools
- Developer rules bootstrap – one call indexes .cursorrules, .cursor/rules, AGENT.md, and friends into the developer_rules nodeset
- Prune & reset – wipe memory clean with a single prune call when you want to start fresh

Please refer to our documentation [here](https://docs.cognee.ai/how-to-guides/deployment/mcp) for further information.

## 🚀 Quick Start

1. Clone cognee repo
    ```
    git clone https://github.com/topoteretes/cognee.git
    ```
2. Navigate to cognee-mcp subdirectory
    ```
    cd cognee/cognee-mcp
    ```
3. Install uv if you don't have one
    ```
    pip install uv
    ```
4. Install all the dependencies you need for cognee mcp server with uv
    ```
    uv sync --dev --all-extras --reinstall
    ```
5. Activate the virtual environment in cognee mcp directory
    ```
    source .venv/bin/activate
    ```
6. Set up your OpenAI API key in .env for a quick setup with the default cognee configurations
    ```
    LLM_API_KEY="YOUR_OPENAI_API_KEY"
    ```
7. Run cognee mcp server with stdio (default)
    ```
    python src/server.py
    ```
    or stream responses over SSE
    ```
    python src/server.py --transport sse
    ```
    or run with Streamable HTTP transport (recommended for web deployments)
    ```
    python src/server.py --transport http --host 127.0.0.1 --port 8000 --path /mcp
    ```

You can do more advanced configurations by creating .env file using our <a href="https://github.com/topoteretes/cognee/blob/main/.env.template">template.</a>
To use different LLM providers / database configurations, and for more info check out our <a href="https://docs.cognee.ai">documentation</a>.


## 🐳 Docker Usage

If you'd rather run cognee-mcp in a container, you have two options:

1. **Build locally**
   1. Make sure you are in /cognee root directory and have a fresh `.env` containing only your `LLM_API_KEY` (and your chosen settings).
   2. Remove any old image and rebuild:
      ```bash
      docker rmi cognee/cognee-mcp:main || true
      docker build --no-cache -f cognee-mcp/Dockerfile -t cognee/cognee-mcp:main .
      ```
   3. Run it:
      ```bash
      # For HTTP transport (recommended for web deployments)
      docker run -e TRANSPORT_MODE=http --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
      # For SSE transport  
      docker run -e TRANSPORT_MODE=sse --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
      # For stdio transport (default)
      docker run -e TRANSPORT_MODE=stdio --env-file ./.env --rm -it cognee/cognee-mcp:main
      ```
      
      **Installing optional dependencies at runtime:**
      
      You can install optional dependencies when running the container by setting the `EXTRAS` environment variable:
      ```bash
      # Install a single optional dependency group at runtime
      docker run \
        -e TRANSPORT_MODE=http \
        -e EXTRAS=aws \
        --env-file ./.env \
        -p 8000:8000 \
        --rm -it cognee/cognee-mcp:main
      
      # Install multiple optional dependency groups at runtime (comma-separated)
      docker run \
        -e TRANSPORT_MODE=sse \
        -e EXTRAS=aws,postgres,neo4j \
        --env-file ./.env \
        -p 8000:8000 \
        --rm -it cognee/cognee-mcp:main
      ```
      
      **Available optional dependency groups:**
      - `aws` - S3 storage support
      - `postgres` / `postgres-binary` - PostgreSQL database support
      - `neo4j` - Neo4j graph database support
      - `neptune` - AWS Neptune support
      - `chromadb` - ChromaDB vector store support
      - `scraping` - Web scraping capabilities
      - `distributed` - Modal distributed execution
      - `langchain` - LangChain integration
      - `llama-index` - LlamaIndex integration
      - `anthropic` - Anthropic models
      - `groq` - Groq models
      - `mistral` - Mistral models
      - `ollama` / `huggingface` - Local model support
      - `docs` - Document processing
      - `codegraph` - Code analysis
      - `monitoring` - Sentry & Langfuse monitoring
      - `redis` - Redis support
      - And more (see [pyproject.toml](https://github.com/topoteretes/cognee/blob/main/pyproject.toml) for full list)
2. **Pull from Docker Hub** (no build required):
   ```bash
   # With HTTP transport (recommended for web deployments)
   docker run -e TRANSPORT_MODE=http --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
   # With SSE transport
   docker run -e TRANSPORT_MODE=sse --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
   # With stdio transport (default)
   docker run -e TRANSPORT_MODE=stdio --env-file ./.env --rm -it cognee/cognee-mcp:main
   ```
   
   **With runtime installation of optional dependencies:**
   ```bash
   # Install optional dependencies from Docker Hub image
   docker run \
     -e TRANSPORT_MODE=http \
     -e EXTRAS=aws,postgres \
     --env-file ./.env \
     -p 8000:8000 \
     --rm -it cognee/cognee-mcp:main
   ```

### **Important: Docker vs Direct Usage**
**Docker uses environment variables**, not command line arguments:
- ✅ Docker: `-e TRANSPORT_MODE=http`
- ❌ Docker: `--transport http` (won't work)

**Direct Python usage** uses command line arguments:
- ✅ Direct: `python src/server.py --transport http`
- ❌ Direct: `-e TRANSPORT_MODE=http` (won't work)

### **Docker API Mode**

To connect the MCP Docker container to a Cognee API server running on your host machine:

#### **Simple Usage (Automatic localhost handling):**
```bash
# Start your Cognee API server on the host
python -m cognee.api.client

# Run MCP container in API mode - localhost is automatically converted!
docker run \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://localhost:8000 \
  -e API_TOKEN=your_auth_token \
  -p 8001:8000 \
  --rm -it cognee/cognee-mcp:main
```
**Note:** The container will automatically convert `localhost` to `host.docker.internal` on Mac/Windows/Docker Desktop. You'll see a message in the logs showing the conversion.

#### **Explicit host.docker.internal (Mac/Windows):**
```bash
# Or explicitly use host.docker.internal
docker run \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://host.docker.internal:8000 \
  -e API_TOKEN=your_auth_token \
  -p 8001:8000 \
  --rm -it cognee/cognee-mcp:main
```

#### **On Linux (use host network or container IP):**
```bash
# Option 1: Use host network (simplest)
docker run \
  --network host \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://localhost:8000 \
  -e API_TOKEN=your_auth_token \
  --rm -it cognee/cognee-mcp:main

# Option 2: Use host IP address
# First, get your host IP: ip addr show docker0
docker run \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://172.17.0.1:8000 \
  -e API_TOKEN=your_auth_token \
  -p 8001:8000 \
  --rm -it cognee/cognee-mcp:main
```

**Environment variables for API mode:**
- `API_URL`: URL of the running Cognee API server
- `API_TOKEN`: Authentication token (optional, required if API has authentication enabled)

**Note:** When running in API mode:
- Database migrations are automatically skipped (API server handles its own DB)
- Some features are limited (see [API Mode Limitations](#-api-mode))


## 🔗 MCP Client Configuration

After starting your Cognee MCP server with Docker, you need to configure your MCP client to connect to it.

### **SSE Transport Configuration** (Recommended)

**Start the server with SSE transport:**
```bash
docker run -e TRANSPORT_MODE=sse --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
```

**Configure your MCP client:**

#### **Claude CLI (Easiest)**
```bash
claude mcp add cognee-sse -t sse http://localhost:8000/sse
```

**Verify the connection:**
```bash
claude mcp list
```

You should see your server connected:
```
Checking MCP server health...

cognee-sse: http://localhost:8000/sse (SSE) - ✓ Connected
```

#### **Manual Configuration**

**Claude (`~/.claude.json`)**
```json
{
  "mcpServers": {
    "cognee": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Cursor (`~/.cursor/mcp.json`)**
```json
{
  "mcpServers": {
    "cognee-sse": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### **HTTP Transport Configuration** (Alternative)

**Start the server with HTTP transport:**
```bash
docker run -e TRANSPORT_MODE=http --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
```

**Configure your MCP client:**

#### **Claude CLI (Easiest)**
```bash
claude mcp add cognee-http -t http http://localhost:8000/mcp
```

**Verify the connection:**
```bash
claude mcp list
```

You should see your server connected:
```
Checking MCP server health...

cognee-http: http://localhost:8000/mcp (HTTP) - ✓ Connected
```

#### **Manual Configuration**

**Claude (`~/.claude.json`)**
```json
{
  "mcpServers": {
    "cognee": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**Cursor (`~/.cursor/mcp.json`)**
```json
{
  "mcpServers": {
    "cognee-http": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### **Dual Configuration Example**
You can configure both transports simultaneously for testing:

```json
{
  "mcpServers": {
    "cognee-sse": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    },
    "cognee-http": {
      "type": "http", 
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**Note:** Only enable the server you're actually running to avoid connection errors.

## 🌐 API Mode

The MCP server can operate in two modes:

### **Direct Mode** (Default)
The MCP server directly imports and uses the cognee library. This is the default mode with full feature support.

### **API Mode**
The MCP server connects to an already running Cognee FastAPI server via HTTP requests. This is useful when:
- You have a centralized Cognee API server running
- You want to separate the MCP server from the knowledge graph backend
- You need multiple MCP servers to share the same knowledge graph

**Starting the MCP server in API mode:**
```bash
# Start your Cognee FastAPI server first (default port 8000)
cd /path/to/cognee
python -m cognee.api.client

# Then start the MCP server in API mode
cd cognee-mcp
python src/server.py --api-url http://localhost:8000 --api-token YOUR_AUTH_TOKEN
```

**API Mode with different transports:**
```bash
# With SSE transport
python src/server.py --transport sse --api-url http://localhost:8000 --api-token YOUR_TOKEN

# With HTTP transport
python src/server.py --transport http --api-url http://localhost:8000 --api-token YOUR_TOKEN
```

**API Mode with Docker:**
```bash
# On Mac/Windows (use host.docker.internal to access host)
docker run \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://host.docker.internal:8000 \
  -e API_TOKEN=YOUR_TOKEN \
  -p 8001:8000 \
  --rm -it cognee/cognee-mcp:main

# On Linux (use host network)
docker run \
  --network host \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://localhost:8000 \
  -e API_TOKEN=YOUR_TOKEN \
  --rm -it cognee/cognee-mcp:main
```

**Command-line arguments for API mode:**
- `--api-url`: Base URL of the running Cognee FastAPI server (e.g., `http://localhost:8000`)
- `--api-token`: Authentication token for the API (optional, required if API has authentication enabled)

**Docker environment variables for API mode:**
- `API_URL`: Base URL of the running Cognee FastAPI server
- `API_TOKEN`: Authentication token (optional, required if API has authentication enabled)

**API Mode limitations:**
Some features are only available in direct mode:
- `codify` (code graph pipeline)
- `cognify_status` / `codify_status` (pipeline status tracking)
- `prune` (data reset)
- `get_developer_rules` (developer rules retrieval)
- `list_data` with specific dataset_id (detailed data listing)

Basic operations like `cognify`, `search`, `delete`, and `list_data` (all datasets) work in both modes.

## 💻 Basic Usage

The MCP server exposes its functionality through tools. Call them from any MCP client (Cursor, Claude Desktop, Cline, Roo and more).


### Available Tools

- **cognify**: Turns your data into a structured knowledge graph and stores it in memory

- **cognee_add_developer_rules**: Ingest core developer rule files into memory

- **codify**: Analyse a code repository, build a code graph, stores it in memory

- **delete**: Delete specific data from a dataset (supports soft/hard deletion modes)

- **get_developer_rules**: Retrieve all developer rules that were generated based on previous interactions

- **list_data**: List all datasets and their data items with IDs for deletion operations

- **save_interaction**: Logs user-agent interactions and query-answer pairs

- **prune**: Reset cognee for a fresh start (removes all data)

- **search**: Query memory – supports GRAPH_COMPLETION, RAG_COMPLETION, CODE, CHUNKS, SUMMARIES, CYPHER, and FEELING_LUCKY

- **cognify_status / codify_status**: Track pipeline progress

**Data Management Examples:**
```bash
# List all available datasets and data items
list_data()

# List data items in a specific dataset
list_data(dataset_id="your-dataset-id-here")

# Delete specific data (soft deletion - safer, preserves shared entities)
delete(data_id="data-uuid", dataset_id="dataset-uuid", mode="soft")

# Delete specific data (hard deletion - removes orphaned entities)
delete(data_id="data-uuid", dataset_id="dataset-uuid", mode="hard")
```


## Development and Debugging

### Debugging

To use debugger, run:
    ```bash
    mcp dev src/server.py
    ```

Open inspector with timeout passed:
    ```
    http://localhost:5173?timeout=120000
    ```

To apply new changes while developing cognee you need to do:

1. Update dependencies in cognee folder if needed
2. `uv sync --dev --all-extras --reinstall`
3. `mcp dev src/server.py`

### Development

In order to use local cognee:

1. Uncomment the following line in the cognee-mcp [`pyproject.toml`](pyproject.toml) file and set the cognee root path.
    ```
    #"cognee[postgres,codegraph,gemini,huggingface,docs,neo4j] @ file:/Users/<username>/Desktop/cognee"
    ```
    Remember to replace `file:/Users/<username>/Desktop/cognee` with your actual cognee root path.

2. Install dependencies with uv in the mcp folder
    ```
    uv sync --reinstall
    ```

## Code of Conduct

We are committed to making open source an enjoyable and respectful experience for our community. See <a href="https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md"><code>CODE_OF_CONDUCT</code></a> for more information.

## 💫 Contributors

<a href="https://github.com/topoteretes/cognee/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=topoteretes/cognee"/>
</a>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=topoteretes/cognee&type=Date)](https://star-history.com/#topoteretes/cognee&Date)
