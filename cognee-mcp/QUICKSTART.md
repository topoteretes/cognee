# Cognee MCP Server - Quick Start

Fast track guide to get the MCP server running with optimized Docker configuration.

---

## 1-Minute Quick Start

```bash
# Clone and navigate
cd cognee-mcp

# Build optimized image
docker build -f Dockerfile.optimized -t cognee-mcp:latest .

# Run with SSE transport
docker run -d \
  --name cognee-mcp \
  -e API_URL=http://host.docker.internal:8000 \
  -e TRANSPORT_MODE=sse \
  -p 8001:8001 \
  cognee-mcp:latest

# Check health
curl http://localhost:8001/health
```

---

## Docker Compose Quick Start

```bash
# Start with existing compose
docker-compose --profile mcp up -d

# Or use optimized configuration
docker-compose -f docker-compose.mcp-optimized.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cognee-mcp
```

---

## Essential Commands

### Build
```bash
# Standard build
docker build -f Dockerfile.optimized -t cognee-mcp:v1 cognee-mcp/

# With debug mode
docker build --build-arg DEBUG=true -f Dockerfile.optimized -t cognee-mcp:debug cognee-mcp/
```

### Run
```bash
# SSE transport (for LibreChat, web clients)
docker run -d --name cognee-mcp \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee:8000 \
  -p 8001:8001 \
  cognee-mcp:v1

# STDIO transport (for CLI, testing)
docker run -it --name cognee-mcp \
  -e TRANSPORT_MODE=stdio \
  -e API_URL=http://cognee:8000 \
  cognee-mcp:v1

# HTTP transport (for polling clients)
docker run -d --name cognee-mcp \
  -e TRANSPORT_MODE=http \
  -e API_URL=http://cognee:8000 \
  -p 8001:8001 \
  cognee-mcp:v1
```

### Monitor
```bash
# Health check
curl http://localhost:8001/health

# Detailed health
curl http://localhost:8001/health/detailed | jq

# Logs
docker logs -f cognee-mcp

# Stats
docker stats cognee-mcp
```

### Debug
```bash
# Shell access
docker exec -it cognee-mcp /bin/bash

# Check environment
docker exec cognee-mcp env | grep API

# Test backend connectivity
docker exec cognee-mcp curl http://cognee:8000/health
```

---

## Configuration Quick Reference

### Required Environment Variables
```bash
API_URL=http://cognee:8000          # Backend API URL (REQUIRED)
```

### Common Environment Variables
```bash
TRANSPORT_MODE=sse                  # sse, http, or stdio
HTTP_PORT=8001                      # Server port (sse/http)
API_TOKEN=your_token                # API authentication token
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
ENABLE_BACKEND_ACCESS_CONTROL=true  # Multi-user KB isolation
```

### Full Configuration
See `.env.example` for complete list

---

## Transport Modes

| Mode | Use Case | Port | Command Flag |
|------|----------|------|--------------|
| **sse** | LibreChat, web clients | 8001 | `--transport sse` |
| **http** | REST API, polling | 8001 | `--transport http` |
| **stdio** | CLI, testing | N/A | `--transport stdio` |

---

## Common Scenarios

### Scenario 1: Development with Hot Reload
```yaml
# docker-compose.override.yml
services:
  cognee-mcp:
    environment:
      - DEBUG=true
      - ENVIRONMENT=dev
    volumes:
      - ./cognee-mcp/src:/app/src
    ports:
      - "8001:8001"
      - "5679:5678"  # Debugger
```

### Scenario 2: Production Deployment
```bash
docker run -d \
  --name cognee-mcp \
  --restart unless-stopped \
  --network cognee-network \
  -e ENVIRONMENT=production \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee:8000 \
  -e API_TOKEN=${COGNEE_API_TOKEN} \
  -e ENABLE_BACKEND_ACCESS_CONTROL=true \
  -p 8001:8001 \
  cognee-mcp:v1
```

### Scenario 3: Testing with Backend
```bash
# Start backend first
docker run -d --name cognee-api -p 8000:8000 cognee/cognee:latest

# Start MCP server
docker run -d \
  --name cognee-mcp \
  --link cognee-api \
  -e API_URL=http://cognee-api:8000 \
  -e TRANSPORT_MODE=sse \
  -p 8001:8001 \
  cognee-mcp:v1
```

---

## Troubleshooting Quick Fixes

### Issue: Container exits immediately
```bash
# Check logs
docker logs cognee-mcp

# Common cause: Missing API_URL
docker run -e API_URL=http://cognee:8000 ...
```

### Issue: Cannot connect to backend
```bash
# Test connectivity
docker exec cognee-mcp curl http://cognee:8000/health

# Use correct URL
# Wrong: http://localhost:8000
# Right: http://cognee:8000 (container name)
```

### Issue: Health check failing
```bash
# Check if server is listening
docker exec cognee-mcp netstat -tlnp | grep 8001

# Test manually
docker exec cognee-mcp curl localhost:8001/health
```

### Issue: Port already in use
```bash
# Find process
lsof -i :8001

# Use different port
docker run -p 8002:8001 ...
```

---

## Testing

### Automated Test
```bash
cd cognee-mcp
./test-docker.sh
```

### Manual Test
```bash
# Build
docker build -f Dockerfile.optimized -t test .

# Run
docker run -d --name test -p 8091:8001 \
  -e API_URL=http://host.docker.internal:8000 \
  -e TRANSPORT_MODE=sse test

# Test
curl http://localhost:8091/health

# Cleanup
docker rm -f test
docker rmi test
```

---

## File Locations

### Docker Files
- **Dockerfile**: `/cognee-mcp/Dockerfile.optimized`
- **Entrypoint**: `/cognee-mcp/entrypoint.optimized.sh`
- **Docker Ignore**: `/cognee-mcp/.dockerignore`
- **Compose**: `/docker-compose.mcp-optimized.yml`

### Documentation
- **This guide**: `/cognee-mcp/QUICKSTART.md`
- **Complete guide**: `/cognee-mcp/DOCKER.md`
- **Analysis**: `/DOCS/docker_optimization_analysis.md`
- **Summary**: `/DOCS/docker_optimization_summary.md`

### Configuration
- **Example env**: `/cognee-mcp/.env.example`
- **Test script**: `/cognee-mcp/test-docker.sh`

---

## Next Steps

1. **Read full documentation**: `DOCKER.md`
2. **Configure environment**: Copy `.env.example` to `.env`
3. **Run tests**: Execute `./test-docker.sh`
4. **Deploy**: Use docker-compose or docker run
5. **Monitor**: Check health and logs

---

## Getting Help

1. Check `DOCKER.md` for detailed guide
2. Review logs: `docker logs cognee-mcp`
3. Test health: `curl http://localhost:8001/health/detailed`
4. See troubleshooting in `DOCKER.md`

---

**Ready in 60 seconds!**
