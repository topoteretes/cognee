# cognee MCP server

A Management Control Plane (MCP) server for the Cognee project that provides a standardized interface for AI tools and capabilities.

## Features

- JSON-RPC 2.0 protocol support
- Standardized initialization flow
- Tool management for AI capabilities
- Multi-stage Docker build with optimized caching
- Comprehensive testing framework

## Docker Configuration

The project uses a multi-stage Docker build for optimal efficiency:

1. First stage: Uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` to install dependencies
2. Second stage: Creates a minimal runtime image based on `python:3.12-slim-bookworm`

Key optimizations:
- Dependency caching using buildkit cache mounts
- Separate installation of dependencies and project code for better layer caching
- Bytecode compilation for improved startup performance
- Minimal runtime image with only necessary components

## Installation

### Prerequisites

- Docker
- Python 3.10+
- uv package manager

### Building the Docker Image

```bash
docker build -t cognee-mcp .
```

### Installing Manually

1. Clone the [cognee](https://github.com/topoteretes/cognee) repo

2. Install dependencies

```bash
brew install uv
```

```bash
cd cognee-mcp
uv sync --dev --all-extras --reinstall
```

3. Activate the venv with

```bash
source .venv/bin/activate
```

## Running the MCP Server

### Using Docker

```bash
docker run -it --rm cognee-mcp
```

### Manually

```bash
cd cognee-mcp
source .venv/bin/activate
cognee
```

## Testing

### Running Docker Tests

The project includes tests to verify the Docker image and MCP server initialization:

```bash
./run_docker_tests.sh
```

This will:
1. Build the Docker image
2. Test that the container starts correctly
3. Verify the MCP initialization flow with the 5-second timeout

> **Note**: Docker testing is the recommended approach as it provides a consistent environment for testing the MCP server.

### Testing MCP Initialization Directly

For debugging purposes, you can test the MCP initialization flow directly without Docker:

```bash
./test_mcp_init.py
```

This script:
1. Starts the MCP server process
2. Sends an initialization request
3. Validates the response
4. Sends an initialized notification
5. Terminates the server

> **Note**: Direct testing may require additional configuration depending on your environment. If you encounter issues, please use the Docker testing approach instead.

### MCP Server Initialization Flow

The MCP server follows a standardized initialization flow:

1. **Process Launch**: The MCP client launches the server process
2. **Transport Establishment**: The server sets up its transport layer (stdio)
3. **Initialization Request**: The client sends an initialize request:
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "method": "initialize",
     "params": {
       "protocolVersion": "2024-11-05",
       "clientInfo": {
         "name": "client-name",
         "version": "client-version"
       },
       "capabilities": {
         "resources": {},
         "tools": {},
         "prompts": {},
         "roots": {},
         "sampling": {}
       }
     }
   }
   ```
4. **Server Response**: The server responds with its capabilities:
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "protocolVersion": "2024-11-05",
       "serverInfo": {
         "name": "cognee",
         "version": "0.1.0"
       },
       "capabilities": {
         "experimental": {},
         "tools": {
           "listChanged": false
         }
       }
     }
   }
   ```
5. **Initialization Confirmation**: The client sends an initialized notification:
   ```json
   {
     "jsonrpc": "2.0",
     "method": "initialized"
   }
   ```
6. **Normal Operation**: The server and client exchange messages according to the protocol

## Claude Desktop Integration

To add the server to your Claude config:

1. Locate the Claude config directory:
```bash
cd ~/Library/Application\ Support/Claude/
```

2. Create or edit `claude_desktop_config.json`:
```json
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

3. Restart Claude desktop.

### Installing via Smithery

To install Cognee for Claude Desktop automatically via [Smithery](https://smithery.ai/server/cognee):

```bash
npx -y @smithery/cli install cognee --client claude
```

## Development

### Debugging

To use debugger, run:
```bash
mcp dev src/server.py
```

Open inspector with timeout passed:
```
http://localhost:5173?timeout=120000
```

### Applying Changes

To apply new changes while developing cognee:

1. `poetry lock` in cognee folder
2. `uv sync --dev --all-extras --reinstall`
3. `mcp dev src/server.py`
