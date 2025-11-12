# Cognee MCP Server - Docker Guide

Complete guide for building, deploying, and managing the Cognee MCP Server using Docker.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Files Overview](#docker-files-overview)
3. [Building Images](#building-images)
4. [Running Containers](#running-containers)
5. [Docker Compose](#docker-compose)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)
9. [Security](#security)

---

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the optimized MCP server with the main Cognee service
docker-compose --profile mcp up -d

# Or use the optimized configuration
docker-compose -f docker-compose.mcp-optimized.yml up -d

# Check logs
docker-compose logs -f cognee-mcp

# Check health
curl http://localhost:8001/health
```

### Using Docker CLI

```bash
# Build the optimized image
docker build -f cognee-mcp/Dockerfile.optimized -t cognee-mcp:latest cognee-mcp/

# Run with SSE transport
docker run -d \
  --name cognee-mcp \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://host.docker.internal:8000 \
  -e API_TOKEN=your_token_here \
  -p 8001:8001 \
  cognee-mcp:latest

# Check health
curl http://localhost:8001/health
```

---

## Docker Files Overview

### Files in cognee-mcp/

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Original Dockerfile | Active |
| `Dockerfile.optimized` | Security-hardened, production-ready | **Recommended** |
| `entrypoint.sh` | Original entrypoint script | Active |
| `entrypoint.optimized.sh` | Improved with validation | **Recommended** |
| `.dockerignore` | Excludes unnecessary files | Required |

### Key Differences: Original vs Optimized

| Feature | Original | Optimized |
|---------|----------|-----------|
| User | root | cognee (UID 1000) |
| Health Check | None | ✓ Included |
| Build Context | Root directory | cognee-mcp/ |
| System Package Versions | Unpinned | Pinned |
| Image Labels | Basic | Comprehensive |
| Bytecode Compilation | Commented | Enabled |
| Size | ~180MB | ~165MB |

---

## Building Images

### Build Optimized Image (Recommended)

```bash
# From project root
docker build \
  -f cognee-mcp/Dockerfile.optimized \
  -t cognee-mcp:optimized \
  cognee-mcp/

# With build arguments
docker build \
  -f cognee-mcp/Dockerfile.optimized \
  --build-arg DEBUG=true \
  -t cognee-mcp:debug \
  cognee-mcp/

# Tag for registry
docker tag cognee-mcp:optimized your-registry/cognee-mcp:v1.0.0
```

### Build Original Image

```bash
# From project root (requires parent context)
docker build \
  -f cognee-mcp/Dockerfile \
  -t cognee-mcp:original \
  .
```

### Multi-platform Build

```bash
# Build for multiple architectures
docker buildx build \
  -f cognee-mcp/Dockerfile.optimized \
  --platform linux/amd64,linux/arm64 \
  -t cognee-mcp:multiarch \
  --push \
  cognee-mcp/
```

---

## Running Containers

### Transport Modes

#### 1. SSE Transport (Server-Sent Events)

**Best for:** LibreChat, web clients, HTTP-based MCP clients

```bash
docker run -d \
  --name cognee-mcp-sse \
  --network cognee-network \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee:8000 \
  -e API_TOKEN=${COGNEE_API_TOKEN} \
  -p 8001:8001 \
  cognee-mcp:optimized

# Test connection
curl http://localhost:8001/mcp/sse
```

#### 2. HTTP Transport (Streamable HTTP)

**Best for:** REST API clients, polling-based integrations

```bash
docker run -d \
  --name cognee-mcp-http \
  --network cognee-network \
  -e TRANSPORT_MODE=http \
  -e API_URL=http://cognee:8000 \
  -p 8001:8001 \
  cognee-mcp:optimized
```

#### 3. STDIO Transport

**Best for:** CLI tools, direct process communication, debugging

```bash
docker run -it \
  --name cognee-mcp-stdio \
  --network cognee-network \
  -e TRANSPORT_MODE=stdio \
  -e API_URL=http://cognee:8000 \
  cognee-mcp:optimized

# Or with test client
docker run -it \
  --name cognee-mcp-test \
  --network cognee-network \
  -e API_URL=http://cognee:8000 \
  cognee-mcp:optimized \
  python src/test_client.py
```

### Development Mode

```bash
docker run -it \
  --name cognee-mcp-dev \
  --network cognee-network \
  -e ENVIRONMENT=dev \
  -e DEBUG=true \
  -e TRANSPORT_MODE=sse \
  -e API_URL=http://cognee:8000 \
  -p 8001:8001 \
  -p 5678:5678 \
  -v $(pwd)/cognee-mcp/src:/app/src \
  cognee-mcp:optimized
```

**Debug with VS Code:**
1. Set breakpoint in code
2. Attach debugger to `localhost:5678`
3. Use VS Code's Python debugger configuration

---

## Docker Compose

### Basic Usage

```yaml
# docker-compose.yml
services:
  cognee-mcp:
    image: cognee-mcp:optimized
    environment:
      - TRANSPORT_MODE=sse
      - API_URL=http://cognee:8000
      - API_TOKEN=${COGNEE_API_TOKEN}
    ports:
      - "8001:8001"
    depends_on:
      - cognee
    networks:
      - cognee-network
```

### Commands

```bash
# Start services
docker-compose --profile mcp up -d

# View logs
docker-compose logs -f cognee-mcp

# Restart MCP server
docker-compose restart cognee-mcp

# Stop services
docker-compose --profile mcp down

# Rebuild and restart
docker-compose --profile mcp up -d --build
```

### Using Optimized Configuration

```bash
# Use optimized compose file
docker-compose -f docker-compose.mcp-optimized.yml up -d

# Combine with main compose
docker-compose \
  -f docker-compose.yml \
  -f docker-compose.mcp-optimized.yml \
  --profile mcp up -d
```

---

## Configuration

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `API_URL` | Cognee API server URL | `http://cognee:8000` |

#### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSPORT_MODE` | `stdio` | Transport type: `stdio`, `sse`, `http` |
| `API_TOKEN` | (none) | Bearer token for API authentication |
| `HTTP_PORT` | `8000` | Port for HTTP/SSE transports |
| `DEBUG` | `false` | Enable debug mode |
| `ENVIRONMENT` | `production` | Environment: `dev`, `local`, `production` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MCP_LOG_LEVEL` | `INFO` | MCP-specific log level |
| `ENABLE_BACKEND_ACCESS_CONTROL` | `false` | Enable KB isolation per user |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |

#### Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRAS` | (none) | Extra dependencies to install |
| `DEBUG_PORT` | `5678` | Debugger port |

### Configuration Files

#### .env File (Recommended)

```bash
# .env
COGNEE_API_URL=http://cognee:8000
COGNEE_API_TOKEN=your_secret_token_here
MCP_TRANSPORT_MODE=sse
MCP_HTTP_PORT=8001
ENABLE_BACKEND_ACCESS_CONTROL=true
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

Load with docker-compose:

```yaml
services:
  cognee-mcp:
    env_file: .env
```

#### Config via Docker

```bash
# Pass config file
docker run -d \
  --env-file .env \
  cognee-mcp:optimized
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

**Symptom:** Container exits immediately

**Check logs:**
```bash
docker logs cognee-mcp
```

**Common causes:**
- Missing `API_URL` environment variable
- Invalid transport mode
- Backend API not accessible

**Solution:**
```bash
# Verify API_URL is set
docker run --rm cognee-mcp:optimized env | grep API_URL

# Test backend connectivity
docker run --rm --network cognee-network cognee-mcp:optimized \
  curl http://cognee:8000/health
```

#### 2. "Connection Refused" Errors

**Symptom:** Cannot connect to backend API

**Common causes:**
- Backend not running
- Wrong network configuration
- Localhost URL (should use container name)

**Solution:**
```bash
# Check if backend is running
docker ps | grep cognee

# Verify network
docker network inspect cognee-network

# Use container name instead of localhost
# Wrong: API_URL=http://localhost:8000
# Right: API_URL=http://cognee:8000
```

#### 3. Health Check Failing

**Symptom:** Health check shows "unhealthy"

**Check health:**
```bash
docker inspect cognee-mcp | grep -A 10 Health
```

**Common causes:**
- Wrong port configuration
- Server not responding
- Health endpoint not accessible

**Solution:**
```bash
# Test health endpoint manually
docker exec cognee-mcp curl http://localhost:8001/health

# Check if server is listening
docker exec cognee-mcp netstat -tlnp | grep 8001
```

#### 4. Permission Denied Errors

**Symptom:** Cannot write files or access directories

**Cause:** Running as non-root user (security feature)

**Solution:**
```bash
# Fix ownership of mounted volumes
chown -R 1000:1000 /path/to/volume

# Or run with user flag (not recommended)
docker run --user root cognee-mcp:optimized
```

#### 5. Port Already in Use

**Symptom:** "bind: address already in use"

**Solution:**
```bash
# Find process using port
lsof -i :8001

# Use different port
docker run -p 8002:8001 cognee-mcp:optimized

# Or stop conflicting service
docker stop <container-using-port>
```

### Debugging

#### Interactive Shell

```bash
# Get shell in running container
docker exec -it cognee-mcp /bin/bash

# Or start container with shell
docker run -it --entrypoint /bin/bash cognee-mcp:optimized
```

#### View Logs

```bash
# Follow logs
docker logs -f cognee-mcp

# Last 100 lines
docker logs --tail 100 cognee-mcp

# With timestamps
docker logs -t cognee-mcp
```

#### Network Debugging

```bash
# Test connectivity to backend
docker run --rm --network cognee-network curlimages/curl \
  curl -v http://cognee:8000/health

# Check DNS resolution
docker exec cognee-mcp nslookup cognee

# Check network
docker network inspect cognee-network
```

---

## Production Deployment

### Pre-deployment Checklist

- [ ] Environment variables configured
- [ ] Backend API accessible
- [ ] Network properly configured
- [ ] Resource limits set
- [ ] Health checks working
- [ ] Monitoring configured
- [ ] Backups configured (if applicable)
- [ ] Security scan completed

### Production Configuration

```yaml
# docker-compose.prod.yml
services:
  cognee-mcp:
    image: cognee-mcp:v1.0.0  # Use specific version
    restart: always
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=WARNING
      - ENABLE_BACKEND_ACCESS_CONTROL=true
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    security_opt:
      - no-new-privileges:true
```

### Monitoring

#### Health Checks

```bash
# Basic health
curl http://localhost:8001/health

# Detailed health
curl http://localhost:8001/health/detailed

# Monitor continuously
watch -n 5 curl -s http://localhost:8001/health
```

#### Metrics

```bash
# Container stats
docker stats cognee-mcp

# Resource usage
docker exec cognee-mcp ps aux

# Disk usage
docker exec cognee-mcp df -h
```

### Backup & Recovery

```bash
# Export container configuration
docker inspect cognee-mcp > cognee-mcp-config.json

# Save image
docker save cognee-mcp:optimized | gzip > cognee-mcp.tar.gz

# Load image
gunzip -c cognee-mcp.tar.gz | docker load
```

---

## Security

### Security Best Practices

1. **Run as non-root user** ✓
   - Optimized Dockerfile uses UID 1000
   - Reduces container escape risk

2. **Use specific image versions**
   ```yaml
   image: cognee-mcp:v1.0.0  # Not :latest
   ```

3. **Enable security options**
   ```yaml
   security_opt:
     - no-new-privileges:true
   ```

4. **Use secrets for sensitive data**
   ```bash
   docker secret create cognee_token token.txt
   ```

5. **Scan for vulnerabilities**
   ```bash
   docker scan cognee-mcp:optimized
   # Or
   trivy image cognee-mcp:optimized
   ```

### Security Scanning

```bash
# Using Docker scan
docker scan cognee-mcp:optimized

# Using Trivy
trivy image --severity HIGH,CRITICAL cognee-mcp:optimized

# Using Grype
grype cognee-mcp:optimized
```

### Network Security

```bash
# Use internal network
docker network create --internal cognee-internal

# Expose only necessary ports
# Don't expose debug port in production
```

### Secrets Management

```bash
# Using Docker secrets (Swarm)
echo "my_secret_token" | docker secret create api_token -

# Using in service
docker service create \
  --name cognee-mcp \
  --secret api_token \
  cognee-mcp:optimized

# Using environment file
docker run --env-file secrets.env cognee-mcp:optimized
```

---

## Additional Resources

- [Cognee MCP README](README.md)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs: `docker logs cognee-mcp`
3. Test health endpoint: `curl http://localhost:8001/health`
4. Refer to main documentation in `/DOCS/`

---

**Last Updated:** 2025-11-12
**Maintainer:** DevOps Team
