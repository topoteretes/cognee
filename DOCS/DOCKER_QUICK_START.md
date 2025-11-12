# Docker Quick Start Guide

This guide provides quick commands for deploying Cognee using the pre-built Docker images from Docker Hub.

## Available Images

- **MCP Server**: `lvarming/cognee-mcp:latest`
- **Frontend UI**: `lvarming/cognee-frontend:latest`

Both images support `linux/amd64` and `linux/arm64` architectures.

## Prerequisites

1. Docker installed and running
2. (Optional) Cognee API backend running
3. Network connectivity between containers

## Quick Deploy Commands

### 1. MCP Server (SSE Transport)

```bash
docker run -d \
  --name cognee-mcp \
  --restart unless-stopped \
  -p 8001:8000 \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://localhost:8000 \
  -e API_TOKEN=your_api_token \
  -e MCP_LOG_LEVEL=INFO \
  lvarming/cognee-mcp:latest
```

### 2. MCP Server (HTTP Transport)

```bash
docker run -d \
  --name cognee-mcp \
  --restart unless-stopped \
  -p 8001:8000 \
  -e TRANSPORT_MODE=http \
  -e API_URL=http://localhost:8000 \
  -e API_TOKEN=your_api_token \
  lvarming/cognee-mcp:latest
```

### 3. Frontend UI

```bash
docker run -d \
  --name cognee-frontend \
  --restart unless-stopped \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  -e NEXT_PUBLIC_ENABLE_NOTEBOOKS=false \
  lvarming/cognee-frontend:latest
```

## Docker Compose Example

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  cognee-mcp:
    image: lvarming/cognee-mcp:latest
    container_name: cognee-mcp
    restart: unless-stopped
    ports:
      - "8001:8000"
    environment:
      - TRANSPORT_MODE=sse
      - API_URL=http://cognee-api:8000
      - API_TOKEN=${COGNEE_API_TOKEN}
      - MCP_LOG_LEVEL=INFO
      - CORS_ALLOWED_ORIGINS=*
    networks:
      - cognee-network
    depends_on:
      - cognee-api

  cognee-frontend:
    image: lvarming/cognee-frontend:latest
    container_name: cognee-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_ENABLE_NOTEBOOKS=false
      - NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR=false
    networks:
      - cognee-network
    depends_on:
      - cognee-api

  # Add your cognee-api service here
  # cognee-api:
  #   image: your/cognee-api:latest
  #   ...

networks:
  cognee-network:
    driver: bridge
```

Run with:
```bash
docker-compose up -d
```

## Environment Variables

### cognee-mcp

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRANSPORT_MODE` | No | `stdio` | Transport protocol: `stdio`, `sse`, or `http` |
| `API_URL` | Yes | - | Cognee API base URL |
| `API_TOKEN` | No | - | Bearer token for API authentication |
| `MCP_LOG_LEVEL` | No | `DEBUG` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CORS_ALLOWED_ORIGINS` | No | `*` | CORS allowed origins (comma-separated) |
| `HOST` | No | `0.0.0.0` | Server host (for SSE/HTTP) |
| `PORT` | No | `8000` | Server port (for SSE/HTTP) |

### cognee-frontend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | - | Cognee API base URL |
| `NEXT_PUBLIC_ENABLE_NOTEBOOKS` | No | `false` | Enable notebook features |
| `NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR` | No | `false` | Enable cloud connector |
| `PORT` | No | `3000` | Server port |

## Health Checks

### MCP Server

```bash
# Check if MCP is running (SSE/HTTP mode only)
curl http://localhost:8001/health

# Detailed health check
curl http://localhost:8001/health/detailed
```

### Frontend

```bash
# Check if frontend is accessible
curl http://localhost:3000
```

## Logs

View container logs:

```bash
# MCP logs
docker logs -f cognee-mcp

# Frontend logs
docker logs -f cognee-frontend

# Last 100 lines
docker logs --tail 100 cognee-mcp
```

## Updating Images

Pull latest versions:

```bash
# Update MCP
docker pull lvarming/cognee-mcp:latest
docker stop cognee-mcp
docker rm cognee-mcp
# Re-run your docker run command

# Update Frontend
docker pull lvarming/cognee-frontend:latest
docker stop cognee-frontend
docker rm cognee-frontend
# Re-run your docker run command
```

Or with Docker Compose:

```bash
docker-compose pull
docker-compose up -d
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs cognee-mcp

# Check if port is already in use
lsof -i :8001  # macOS/Linux
netstat -ano | findstr :8001  # Windows

# Inspect container
docker inspect cognee-mcp
```

### Can't connect to API

1. Verify API is running: `curl http://localhost:8000/health`
2. Check network connectivity between containers
3. Verify API_URL in MCP configuration
4. Check API_TOKEN is correct

### CORS errors in frontend

1. Ensure `NEXT_PUBLIC_API_URL` points to the correct API
2. Check API CORS configuration allows your frontend origin
3. Verify network routing between frontend and API

### Image pull fails

```bash
# Verify image exists
docker pull lvarming/cognee-mcp:latest

# Try specific version
docker pull lvarming/cognee-mcp:0.4.0

# Check Docker Hub status
# https://status.docker.com
```

## Advanced Usage

### Running with custom network

```bash
# Create network
docker network create cognee-net

# Run MCP with network
docker run -d \
  --name cognee-mcp \
  --network cognee-net \
  -p 8001:8000 \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee-api:8000 \
  lvarming/cognee-mcp:latest
```

### Mounting configuration files

```bash
docker run -d \
  --name cognee-mcp \
  -v /path/to/config/.env:/app/.env:ro \
  -p 8001:8000 \
  lvarming/cognee-mcp:latest
```

### Running specific version

```bash
# Run version 0.4.0
docker run -d \
  --name cognee-mcp \
  -p 8001:8000 \
  -e API_URL=http://localhost:8000 \
  lvarming/cognee-mcp:0.4.0
```

### Building locally (for development)

```bash
# Clone repository
git clone https://github.com/Varming73/cognee.git
cd cognee

# Build MCP
docker build -t cognee-mcp:local -f cognee-mcp/Dockerfile .

# Build Frontend
docker build -t cognee-frontend:local -f cognee-frontend/Dockerfile ./cognee-frontend

# Run local builds
docker run -d --name cognee-mcp -p 8001:8000 cognee-mcp:local
```

## Unraid-Specific Tips

If deploying on Unraid:

1. Use **Bridge** network mode for container networking
2. Add containers to the same custom bridge network
3. Set up reverse proxy (Nginx Proxy Manager) for external access
4. Use persistent storage for logs: `-v /mnt/user/appdata/cognee-mcp:/app/logs`
5. Configure auto-start on boot
6. Set up container health checks in Unraid GUI

## LibreChat Integration

Add to your `librechat.yaml`:

```yaml
mcpServers:
  cognee-search:
    type: sse
    url: http://cognee-mcp:8001/mcp
    serverInstructions: true
    headers:
      Authorization: "Bearer ${COGNEE_MCP_TOKEN}"
```

Or for HTTP transport:

```yaml
mcpServers:
  cognee-search:
    type: http
    url: http://cognee-mcp:8001/mcp
    serverInstructions: true
    headers:
      Authorization: "Bearer ${COGNEE_MCP_TOKEN}"
```

## Next Steps

1. Read the full documentation:
   - [MCP README](/cognee-mcp/README.md)
   - [Frontend README](/cognee-frontend/README.md)
   - [Docker Hub CI/CD Setup](/DOCS/DOCKERHUB_CI_CD_SETUP.md)

2. Configure LibreChat or your MCP client

3. Explore the Cognee API documentation

4. Join the community for support

## Additional Resources

- Docker Hub Repositories:
  - https://hub.docker.com/r/lvarming/cognee-mcp
  - https://hub.docker.com/r/lvarming/cognee-frontend
- GitHub Repository: https://github.com/Varming73/cognee
- Cognee Documentation: https://docs.cognee.ai
