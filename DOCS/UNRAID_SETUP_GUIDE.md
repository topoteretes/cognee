# Complete Cognee Setup Guide for Unraid

This guide will walk you through setting up the complete Cognee stack on Unraid, including the backend (from upstream), MCP server and frontend (from your fork), and integrating with LibreChat.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Part 1: Cognee Backend Setup](#part-1-cognee-backend-setup)
4. [Part 2: Cognee MCP Server Setup](#part-2-cognee-mcp-server-setup)
5. [Part 3: Cognee Frontend Setup](#part-3-cognee-frontend-setup)
6. [Part 4: LibreChat MCP Integration](#part-4-librechat-mcp-integration)
7. [Verification & Testing](#verification--testing)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────┐
│   LibreChat     │ (Port 3080)
│   with MCP      │
└────────┬────────┘
         │ SSE Connection (Port 8001)
         ↓
┌─────────────────┐
│  Cognee MCP     │ (Port 8001)
│  Server         │ ← Your Fork (lvarming/cognee-mcp)
└────────┬────────┘
         │ API Calls (Port 8000)
         ↓
┌─────────────────┐
│  Cognee Backend │ (Port 8000)
│  API Server     │ ← Upstream (topoteretes/cognee)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  PostgreSQL +   │
│  Qdrant + Neo4j │
└─────────────────┘

┌─────────────────┐
│ Cognee Frontend │ (Port 3000) ← Your Fork (lvarming/cognee-frontend)
│   Web UI        │
└────────┬────────┘
         │ API Calls (Port 8000)
         └──────────────────────────────┘
```

---

## Prerequisites

### Required on Unraid

1. **Docker installed** (comes with Unraid)
2. **Custom bridge network** (we'll create this)
3. **Persistent storage paths** (we'll set these up)

### Required External Services

1. **DockerHub account** (for pulling images)
2. **OpenAI API key** or other LLM provider
3. **Jina AI API key** (free tier available)
   - Get it at: https://jina.ai/?sui=apikey
   - Used for MCP search reranking

---

## Part 1: Cognee Backend Setup

### Step 1.1: Create Docker Network

In Unraid terminal (or via SSH):

```bash
docker network create cognee-network
```

### Step 1.2: Create Storage Directories

```bash
mkdir -p /mnt/user/appdata/cognee-backend
mkdir -p /mnt/user/appdata/cognee-backend/data
mkdir -p /mnt/user/appdata/cognee-backend/.cognee_system
```

### Step 1.3: Create Backend Container

**In Unraid WebUI:**

1. Go to **Docker** tab
2. Click **Add Container**
3. Fill in the following:

**Basic Settings:**
- **Name:** `cognee-backend`
- **Repository:** `topoteretes/cognee:latest`
- **Network Type:** `Custom : cognee-network`

**Port Mappings:**
- **Container Port:** `8000`
- **Host Port:** `8000`
- **Protocol:** `TCP`

**Environment Variables:**

```bash
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Graph Database
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://neo4j:7687
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=your_secure_neo4j_password

# Vector Database
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://qdrant:6333
VECTOR_DB_KEY=

# PostgreSQL Database
DB_PROVIDER=postgres
DB_HOST=postgres
DB_PORT=5432
DB_NAME=cognee
DB_USERNAME=cognee
DB_PASSWORD=your_secure_postgres_password

# Backend Configuration
ENABLE_BACKEND_ACCESS_CONTROL=true
BACKEND_PORT=8000
LOG_LEVEL=INFO

# System Paths
SYSTEM_ROOT=/app/.cognee_system
DATA_ROOT=/app/data
```

**Path Mappings:**

| Container Path | Host Path | Access Mode |
|---------------|-----------|-------------|
| `/app/data` | `/mnt/user/appdata/cognee-backend/data` | Read/Write |
| `/app/.cognee_system` | `/mnt/user/appdata/cognee-backend/.cognee_system` | Read/Write |

**Advanced Settings:**
- **Extra Parameters:** `--restart unless-stopped`

### Step 1.4: Create PostgreSQL Container

**Add Container:**

- **Name:** `cognee-postgres`
- **Repository:** `postgres:15-alpine`
- **Network Type:** `Custom : cognee-network`

**Environment Variables:**

```bash
POSTGRES_USER=cognee
POSTGRES_PASSWORD=your_secure_postgres_password
POSTGRES_DB=cognee
```

**Path Mappings:**

| Container Path | Host Path | Access Mode |
|---------------|-----------|-------------|
| `/var/lib/postgresql/data` | `/mnt/user/appdata/cognee-backend/postgres` | Read/Write |

### Step 1.5: Create Qdrant Container

**Add Container:**

- **Name:** `cognee-qdrant`
- **Repository:** `qdrant/qdrant:latest`
- **Network Type:** `Custom : cognee-network`

**Port Mappings:**
- **Container Port:** `6333` → **Host Port:** `6333`

**Path Mappings:**

| Container Path | Host Path | Access Mode |
|---------------|-----------|-------------|
| `/qdrant/storage` | `/mnt/user/appdata/cognee-backend/qdrant` | Read/Write |

### Step 1.6: Create Neo4j Container

**Add Container:**

- **Name:** `cognee-neo4j`
- **Repository:** `neo4j:5-community`
- **Network Type:** `Custom : cognee-network`

**Port Mappings:**
- **Container Port:** `7474` → **Host Port:** `7474` (HTTP)
- **Container Port:** `7687` → **Host Port:** `7687` (Bolt)

**Environment Variables:**

```bash
NEO4J_AUTH=neo4j/your_secure_neo4j_password
NEO4J_server_memory_heap_max__size=2G
NEO4J_server_memory_pagecache_size=1G
```

**Path Mappings:**

| Container Path | Host Path | Access Mode |
|---------------|-----------|-------------|
| `/data` | `/mnt/user/appdata/cognee-backend/neo4j/data` | Read/Write |
| `/logs` | `/mnt/user/appdata/cognee-backend/neo4j/logs` | Read/Write |

### Step 1.7: Start Backend Stack

Start containers in this order:
1. `cognee-postgres`
2. `cognee-qdrant`
3. `cognee-neo4j`
4. Wait 30 seconds for databases to initialize
5. `cognee-backend`

**Verify Backend is Running:**

```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

---

## Part 2: Cognee MCP Server Setup

### Step 2.1: Create MCP Storage Directory

```bash
mkdir -p /mnt/user/appdata/cognee-mcp
```

### Step 2.2: Create MCP Container

**In Unraid WebUI:**

1. Go to **Docker** tab
2. Click **Add Container**

**Basic Settings:**
- **Name:** `cognee-mcp`
- **Repository:** `lvarming/cognee-mcp:latest`
- **Network Type:** `Custom : cognee-network`

**Port Mappings:**
- **Container Port:** `8000`
- **Host Port:** `8001` (different from backend!)
- **Protocol:** `TCP`

**Environment Variables:**

```bash
# Transport Mode
TRANSPORT_MODE=sse

# Backend API Connection
API_URL=http://cognee-backend:8000
API_TOKEN=

# MCP Configuration
MCP_SERVER_NAME=cognee-search
LOG_LEVEL=INFO

# Jina Reranker (for search quality)
JINA_API_KEY=your_jina_api_key_here
RERANK_PROVIDER=jina

# CORS (for LibreChat access)
CORS_ALLOWED_ORIGINS=http://localhost:3080,http://your-unraid-ip:3080
```

**Important Notes:**
- `API_URL` uses the Docker network name `cognee-backend` (not localhost!)
- Port `8001` on host avoids conflict with backend's `8000`
- `CORS_ALLOWED_ORIGINS` should include your LibreChat URL
- Get free Jina API key: https://jina.ai/?sui=apikey

**Advanced Settings:**
- **Extra Parameters:** `--restart unless-stopped`

### Step 2.3: Start MCP Server

Start the `cognee-mcp` container.

**Verify MCP is Running:**

```bash
# Health check
curl http://localhost:8001/health

# Should return:
# {
#   "status": "healthy",
#   "mcp_server": "cognee-search",
#   "transport": "sse",
#   "backend_url": "http://cognee-backend:8000"
# }
```

---

## Part 3: Cognee Frontend Setup

### Step 3.1: Create Frontend Container

**In Unraid WebUI:**

1. Go to **Docker** tab
2. Click **Add Container**

**Basic Settings:**
- **Name:** `cognee-frontend`
- **Repository:** `lvarming/cognee-frontend:latest`
- **Network Type:** `Custom : cognee-network`

**Port Mappings:**
- **Container Port:** `3000`
- **Host Port:** `3000`
- **Protocol:** `TCP`

**Environment Variables:**

```bash
# Backend API URL
NEXT_PUBLIC_BACKEND_API_URL=http://your-unraid-ip:8000/api

# Feature Flags (from your fork)
NEXT_PUBLIC_ENABLE_NOTEBOOKS=false
NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR=false

# Production Settings
NODE_ENV=production
PORT=3000
```

**Important:**
- Replace `your-unraid-ip` with your actual Unraid server IP
- The frontend needs to access the backend from the browser (not Docker network)

**Advanced Settings:**
- **Extra Parameters:** `--restart unless-stopped`

### Step 3.2: Start Frontend

Start the `cognee-frontend` container.

**Verify Frontend is Running:**

Open browser: `http://your-unraid-ip:3000`

You should see the Cognee web interface.

---

## Part 4: LibreChat MCP Integration

### Step 4.1: Locate LibreChat Configuration

Find your LibreChat installation's `librechat.yaml` file. Common locations:
- `/mnt/user/appdata/librechat/librechat.yaml`
- Inside LibreChat container: `/app/librechat.yaml`

### Step 4.2: Add Cognee MCP Configuration

Edit `librechat.yaml` and add the following under `mcpServers`:

```yaml
mcpServers:
  cognee-search:
    type: sse
    url: http://your-unraid-ip:8001/sse
    serverInstructions: |
      Cognee MCP provides advanced knowledge graph search capabilities:

      1. **list_datasets** - Always call this first to see available knowledge bases
         - Returns dataset IDs, names, owners, and creation dates
         - Use dataset IDs (not names) for best results

      2. **search** - Query knowledge graphs with natural language
         - Required: search_query (the user's question)
         - Recommended: dataset_ids (list of UUID strings from list_datasets)
         - Optional parameters:
           * search_type: "GRAPH_COMPLETION" (default), "RAG_COMPLETION", "CHUNKS", "SUMMARIES"
           * top_k: 1-50 results (default: 10)
           * use_combined_context: true (includes reasoning), false (raw results)
           * only_context: true (context only), false (includes answer)

      3. **get_dataset_summary** - Preview a knowledge base before searching
         - Required: dataset_id
         - Returns top summaries to understand content scope

      Best Practices:
      - Call list_datasets at conversation start to show user their knowledge bases
      - Use dataset_ids for targeted searches (more accurate than dataset names)
      - Start with top_k=10, increase if more results needed
      - Use GRAPH_COMPLETION for multi-hop reasoning across connected concepts
      - Use combined_context=true to get Cognee's reasoning with results

      Multi-User Support:
      - Each user sees only their own datasets (if backend access control enabled)
      - Dataset IDs are unique and stable across sessions

    headers:
      X-User-ID: "{{LIBRECHAT_USER_ID}}"
      X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"

    timeout: 45000
    initTimeout: 15000

    # Icon (optional - add if you have a custom icon)
    # iconPath: /path/to/cognee-icon.svg

    chatMenu: true
```

**Configuration Breakdown:**

| Setting | Value | Purpose |
|---------|-------|---------|
| `type` | `sse` | Server-Sent Events transport (best for Cognee MCP) |
| `url` | `http://your-unraid-ip:8001/sse` | MCP server endpoint |
| `serverInstructions` | Long string | Teaches the LLM how to use Cognee effectively |
| `headers.X-User-ID` | `{{LIBRECHAT_USER_ID}}` | Multi-user isolation (if enabled) |
| `headers.X-User-Email` | `{{LIBRECHAT_USER_EMAIL}}` | User tracking |
| `timeout` | `45000` | 45 seconds for search operations |
| `initTimeout` | `15000` | 15 seconds for connection |
| `chatMenu` | `true` | Show in chat dropdown for easy access |

### Step 4.3: Update LibreChat Environment Variables (Optional)

If you want to use Jina reranking via MCP headers, add to LibreChat's `.env`:

```bash
# Jina API Key (if passing through headers)
JINA_API_KEY=your_jina_api_key_here
```

Then update the MCP config:

```yaml
    headers:
      X-User-ID: "{{LIBRECHAT_USER_ID}}"
      X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"
      X-Jina-API-Key: "${JINA_API_KEY}"
```

### Step 4.4: Restart LibreChat

```bash
docker restart librechat
```

### Step 4.5: Verify MCP Integration

1. Open LibreChat: `http://your-unraid-ip:3080`
2. Start a new conversation
3. Click the **plugins/tools icon** (wrench or similar)
4. You should see **"cognee-search"** in the MCP servers list
5. Enable it and check the available tools:
   - `list_datasets`
   - `search`
   - `get_dataset_summary`

---

## Verification & Testing

### Test 1: Backend Health

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Test 2: MCP Health

```bash
curl http://localhost:8001/health
# Expected: {"status": "healthy", "mcp_server": "cognee-search", ...}
```

### Test 3: Frontend Access

Open browser: `http://your-unraid-ip:3000`

Expected: Cognee dashboard loads

### Test 4: Create Test Dataset (via Frontend)

1. Go to `http://your-unraid-ip:3000`
2. Create a new dataset: "Test Knowledge Base"
3. Add some text data: "Cognee is a knowledge graph platform for AI."
4. Click "Cognify" to process
5. Wait for processing to complete

### Test 5: LibreChat MCP Search

1. Open LibreChat
2. Enable "cognee-search" MCP server
3. Ask: "What datasets are available?"
   - LLM should call `list_datasets` tool
   - Should show your "Test Knowledge Base"
4. Ask: "Search my test knowledge base for information about Cognee"
   - LLM should call `search` tool
   - Should return: "Cognee is a knowledge graph platform for AI."

---

## Troubleshooting

### Issue 1: MCP Can't Connect to Backend

**Symptom:** MCP health check fails or shows backend unreachable

**Solution:**
1. Check backend is running: `docker ps | grep cognee-backend`
2. Verify network: `docker network inspect cognee-network`
3. Check MCP logs: `docker logs cognee-mcp`
4. Ensure `API_URL=http://cognee-backend:8000` (not localhost!)

### Issue 2: LibreChat Can't Reach MCP

**Symptom:** MCP server doesn't appear in LibreChat or shows connection error

**Solution:**
1. Check CORS settings in MCP container:
   ```bash
   CORS_ALLOWED_ORIGINS=http://localhost:3080,http://your-unraid-ip:3080
   ```
2. Verify LibreChat can reach MCP:
   ```bash
   docker exec -it librechat curl http://your-unraid-ip:8001/health
   ```
3. Check LibreChat logs: `docker logs librechat`

### Issue 3: Frontend Can't Connect to Backend

**Symptom:** Frontend loads but shows "API connection failed"

**Solution:**
1. Verify backend URL in frontend env:
   ```bash
   NEXT_PUBLIC_BACKEND_API_URL=http://your-unraid-ip:8000/api
   ```
2. Must use Unraid IP (not Docker network name) - browser needs access
3. Check backend is accessible from browser: `http://your-unraid-ip:8000/health`

### Issue 4: Search Results Are Poor Quality

**Symptom:** MCP search returns irrelevant results

**Solution:**
1. Verify Jina API key is set in MCP container
2. Check MCP logs for reranking errors
3. Try increasing `top_k` in search parameters
4. Use `search_type: "GRAPH_COMPLETION"` for better reasoning

### Issue 5: Multi-User Isolation Not Working

**Symptom:** Users see each other's datasets

**Solution:**
1. Verify backend has access control enabled:
   ```bash
   ENABLE_BACKEND_ACCESS_CONTROL=true
   ```
2. Restart backend: `docker restart cognee-backend`
3. Check LibreChat passes user ID:
   ```yaml
   headers:
     X-User-ID: "{{LIBRECHAT_USER_ID}}"
   ```

---

## Quick Reference: Container Ports

| Service | Container Port | Host Port | Purpose |
|---------|---------------|-----------|---------|
| Cognee Backend | 8000 | 8000 | Main API |
| Cognee MCP | 8000 | 8001 | MCP Server (SSE) |
| Cognee Frontend | 3000 | 3000 | Web UI |
| PostgreSQL | 5432 | (internal) | Database |
| Qdrant | 6333 | 6333 | Vector DB |
| Neo4j HTTP | 7474 | 7474 | Graph DB UI |
| Neo4j Bolt | 7687 | 7687 | Graph DB |
| LibreChat | 3000 | 3080 | Chat UI |

---

## Quick Reference: Environment Variables

### Cognee Backend
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://neo4j:7687
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://qdrant:6333
DB_HOST=postgres
ENABLE_BACKEND_ACCESS_CONTROL=true
```

### Cognee MCP
```bash
TRANSPORT_MODE=sse
API_URL=http://cognee-backend:8000
JINA_API_KEY=jina_...
CORS_ALLOWED_ORIGINS=http://localhost:3080
```

### Cognee Frontend
```bash
NEXT_PUBLIC_BACKEND_API_URL=http://your-unraid-ip:8000/api
NEXT_PUBLIC_ENABLE_NOTEBOOKS=false
NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR=false
```

---

## Advanced: Docker Compose Alternative

If you prefer Docker Compose, save this as `/mnt/user/appdata/cognee/docker-compose.yml`:

```yaml
version: '3.8'

networks:
  cognee-network:
    driver: bridge

services:
  postgres:
    image: postgres:15-alpine
    container_name: cognee-postgres
    networks:
      - cognee-network
    environment:
      POSTGRES_USER: cognee
      POSTGRES_PASSWORD: your_secure_password
      POSTGRES_DB: cognee
    volumes:
      - /mnt/user/appdata/cognee-backend/postgres:/var/lib/postgresql/data
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: cognee-qdrant
    networks:
      - cognee-network
    ports:
      - "6333:6333"
    volumes:
      - /mnt/user/appdata/cognee-backend/qdrant:/qdrant/storage
    restart: unless-stopped

  neo4j:
    image: neo4j:5-community
    container_name: cognee-neo4j
    networks:
      - cognee-network
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/your_secure_password
      NEO4J_server_memory_heap_max__size: 2G
    volumes:
      - /mnt/user/appdata/cognee-backend/neo4j/data:/data
      - /mnt/user/appdata/cognee-backend/neo4j/logs:/logs
    restart: unless-stopped

  backend:
    image: topoteretes/cognee:latest
    container_name: cognee-backend
    networks:
      - cognee-network
    ports:
      - "8000:8000"
    environment:
      LLM_PROVIDER: openai
      OPENAI_API_KEY: your_openai_key
      GRAPH_DATABASE_PROVIDER: neo4j
      GRAPH_DATABASE_URL: bolt://neo4j:7687
      GRAPH_DATABASE_USERNAME: neo4j
      GRAPH_DATABASE_PASSWORD: your_secure_password
      VECTOR_DB_PROVIDER: qdrant
      VECTOR_DB_URL: http://qdrant:6333
      DB_PROVIDER: postgres
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: cognee
      DB_USERNAME: cognee
      DB_PASSWORD: your_secure_password
      ENABLE_BACKEND_ACCESS_CONTROL: "true"
    volumes:
      - /mnt/user/appdata/cognee-backend/data:/app/data
      - /mnt/user/appdata/cognee-backend/.cognee_system:/app/.cognee_system
    depends_on:
      - postgres
      - qdrant
      - neo4j
    restart: unless-stopped

  mcp:
    image: lvarming/cognee-mcp:latest
    container_name: cognee-mcp
    networks:
      - cognee-network
    ports:
      - "8001:8000"
    environment:
      TRANSPORT_MODE: sse
      API_URL: http://backend:8000
      JINA_API_KEY: your_jina_key
      CORS_ALLOWED_ORIGINS: http://localhost:3080
    depends_on:
      - backend
    restart: unless-stopped

  frontend:
    image: lvarming/cognee-frontend:latest
    container_name: cognee-frontend
    networks:
      - cognee-network
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_BACKEND_API_URL: http://your-unraid-ip:8000/api
      NEXT_PUBLIC_ENABLE_NOTEBOOKS: "false"
      NEXT_PUBLIC_ENABLE_CLOUD_CONNECTOR: "false"
    depends_on:
      - backend
    restart: unless-stopped
```

**Run with:**
```bash
cd /mnt/user/appdata/cognee
docker-compose up -d
```

---

## Summary

You now have:
- ✅ Cognee Backend running (upstream project)
- ✅ Cognee MCP Server running (your fork)
- ✅ Cognee Frontend running (your fork)
- ✅ LibreChat integrated with Cognee MCP
- ✅ Multi-user support (if enabled)
- ✅ High-quality search with Jina reranking

**Next Steps:**
1. Create knowledge bases via Frontend (`http://your-unraid-ip:3000`)
2. Use LibreChat to search them (`http://your-unraid-ip:3080`)
3. Monitor logs: `docker logs -f cognee-mcp`

Enjoy your personal knowledge graph assistant! 🎉
