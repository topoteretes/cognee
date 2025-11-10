# Cognee GitHub Codespaces Configuration

This directory contains the configuration for developing Cognee in GitHub Codespaces.

## Features

- **Python 3.11** development environment
- **Docker-in-Docker** support for running services
- **Pre-configured services**: PostgreSQL, Neo4j, ChromaDB, Redis
- **Delayed initialization** for faster codespace startup
- **VS Code extensions** for Python development
- **Auto-forwarded ports** for web services

## What Gets Installed

### Post-Create (Runs once after codespace creation)
The `post-create.sh` script performs delayed initialization:
- Installs Python dependencies (`cognee` with dev extras)
- Sets up pre-commit hooks
- Creates `.env` file from template
- Verifies installation

### Post-Start (Runs on each codespace start)
The `post-start.sh` script performs lightweight startup tasks:
- Checks environment configuration
- Displays helpful information
- Can optionally auto-start services (commented out by default)

## Quick Start

1. **Open in Codespace**: Click "Code" → "Codespaces" → "Create codespace on [branch]"

2. **Wait for initialization**: The post-create script will run automatically (1-3 minutes)

3. **Configure environment**: Edit `.env` file with your API keys
   ```bash
   # At minimum, add your LLM API key
   LLM_API_KEY=your-api-key-here
   ```

4. **Start services**: Choose which services you need
   ```bash
   # Start all services
   docker-compose up -d

   # Or start specific services
   docker-compose up -d postgres neo4j
   ```

5. **Start developing**:
   ```bash
   # Run CLI
   cognee-cli --help

   # Run tests
   pytest

   # Start API server
   python -m cognee.api.server
   ```

## Available Ports

The following ports are automatically forwarded:

- **8000**: Cognee API
- **3000**: Frontend (if using UI profile)
- **7474**: Neo4j Browser
- **7687**: Neo4j Bolt
- **5432**: PostgreSQL
- **6379**: Redis
- **3002**: ChromaDB

## Services

All services are available by default in codespaces (profiles removed):

- **PostgreSQL**: Vector database with pgvector
- **Neo4j**: Graph database
- **ChromaDB**: Vector store
- **Redis**: Cache and message broker

Check service status:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs -f [service-name]
```

## Customization

### Modify Initialization

Edit the initialization scripts:
- `.devcontainer/post-create.sh`: One-time setup tasks
- `.devcontainer/post-start.sh`: Startup tasks

### Auto-start Services

Uncomment the service startup lines in `post-start.sh` to automatically start services:
```bash
nohup docker-compose up postgres neo4j chromadb > /tmp/services.log 2>&1 &
```

### Add VS Code Extensions

Edit `.devcontainer/devcontainer.json` and add to the `extensions` array:
```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "your-extension-id"
      ]
    }
  }
}
```

## Troubleshooting

### Services won't start
```bash
# Check Docker status
docker ps

# Restart services
docker-compose down
docker-compose up -d
```

### Python dependencies issues
```bash
# Reinstall dependencies
pip install --no-cache-dir -e .[dev]
```

### Post-create script failed
```bash
# Manually run the script
bash .devcontainer/post-create.sh
```

### Check logs
```bash
# Service logs
docker-compose logs [service-name]

# Background services log (if auto-started)
cat /tmp/services.log
```

## Performance Tips

1. **Delayed initialization**: Services start only when needed, not automatically
2. **Resource limits**: Configured for optimal codespace performance
3. **Cached volumes**: Workspace is mounted with `cached` consistency
4. **Health checks**: All services have health checks for reliable startup

## Additional Resources

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Dev Container Specification](https://containers.dev/)
- [Cognee Documentation](https://docs.cognee.ai/)
