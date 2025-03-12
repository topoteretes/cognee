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

The Cognee MCP server includes a comprehensive testing framework to ensure its functionality and reliability.

### Automated Testing

To run the automated tests for the MCP tools, use the provided script:

```bash
./run_mcp_tool_tests.sh
```

This script will:
1. Build a Docker image for the MCP server
2. Start a container with the MCP server
3. Run tests for the Docker build, cognify, search, and codify tools
4. Clean up the container and image when done

### Manual Testing

For manual testing and debugging, you can use the provided scripts:

#### Launch MCP Server

To launch the MCP server in a Docker container for manual testing:

```bash
./launch_mcp_server.sh
```

This script will:
1. Build a Docker image for the MCP server
2. Start a container with the MCP server
3. Provide example commands for testing the server

#### Test MCP Tools Manually

To test specific MCP tools with a running server:

```bash
python tests/test_mcp_tools_manual.py [options]
```

Options:
- `--host HOST`: Host where MCP server is running (default: localhost)
- `--port PORT`: Port where MCP server is running (default: 8080)
- `--container CONTAINER`: Container name if using Docker (default: cognee-mcp-dev)
- `--use-container`: Send requests to Docker container instead of host:port
- `--tool {cognify,search,codify,all}`: Which tool to test (default: all)

Examples:

```bash
# Test all tools with a server running on localhost:8080
python tests/test_mcp_tools_manual.py

# Test only the cognify tool with a server running in a Docker container
python tests/test_mcp_tools_manual.py --use-container --tool cognify

# Test all tools with a server running on a different host/port
python tests/test_mcp_tools_manual.py --host 192.168.1.100 --port 9000
```

## MCP Server Startup Flow

The MCP server follows a specific startup flow:

1. **Process Launch**: The server process is started.
2. **Environment Setup**: The server sets up its environment, including loading configuration.
3. **Server Initialization**: The server initializes its components and prepares to receive requests.
4. **Client Connection**: A client connects to the server.
5. **Initialization Request**: The client sends an initialization request to the server.
6. **Server Response**: The server responds with its capabilities and server information.
7. **Normal Operation**: The server is now ready to process tool requests.

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
