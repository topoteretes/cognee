Here's a well-structured, comprehensive documentation for setting up the Cognee MCP Server:

# Cognee MCP Server Documentation

## Overview
The Cognee MCP (Model Control Protocol) Server enables integration between Cognee and Claude Desktop, providing enhanced AI capabilities.

## Installation Methods

### Method 1: Manual Installation (Recommended)

#### Prerequisites
- Homebrew (macOS/Linux)
- Python 3.9+
- Claude Desktop installed

#### Installation Steps

1. **Install UV package manager**:
   ```bash
   brew install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/cognee/cognee.git
   cd cognee/cognee-mcp
   ```

3. **Install dependencies**:
   ```bash
   uv sync --dev --all-extras --reinstall
   ```

4. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Method 2: Smithery Installation (Automated)
```bash
npx -y @smithery/cli install cognee --client claude
```

## Configuration

### Claude Desktop Configuration

1. Navigate to Claude config directory:
   ```bash
   cd ~/Library/Application\ Support/Claude/
   ```

2. Create/edit config file:
   ```bash
   nano claude_desktop_config.json
   ```

3. Add this configuration (replace placeholders):
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
           "LLM_API_KEY": "sk-your-api-key-here"
         }
       }
     }
   }
   ```

## Running the Server

### Standard Mode
```bash
python src/server.py
```

### SSE Transport Mode
```bash
python src/server.py --transport sse
```

## Development Workflow

### Using Local Cognee Build

1. Build Cognee package:
   ```bash
   poetry build -o ./cognee-mcp/sources
   ```

2. Modify `cognee-mcp/pyproject.toml`:
   ```toml
   [tool.poetry.dependencies]
   cognee = { path = "sources/cognee-0.1.38-py3-none-any.whl" }
   
   [tool.uv.sources]
   cognee = { path = "sources/cognee-0.1.38-py3-none-any.whl" }
   ```

3. Sync dependencies:
   ```bash
   uv sync --dev --all-extras --reinstall
   ```

### Development Server
```bash
mcp dev src/server.py
```

### Debugging
Access debug inspector:
```
http://localhost:5173?timeout=120000
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all system dependencies are installed
   - Run `uv sync --dev --all-extras --reinstall` after any dependency changes

2. **Connection Issues**:
   - Verify Claude Desktop is restarted after configuration changes
   - Check server logs for errors

3. **Build Problems**:
   - Clean previous builds with `poetry lock` and rebuild
   - Verify Python version compatibility (requires 3.9+)

## Maintenance

### Updating Dependencies
```bash
poetry lock
uv sync --dev --all-extras --reinstall
```

### Updating Configuration
After any changes to:
- `claude_desktop_config.json`
- `pyproject.toml`
- Server code

Always restart both the server and Claude Desktop.
