# Cognee Docker Compose Documentation

## Overview

Cognee uses Docker Compose to orchestrate multiple services that work together to provide AI memory capabilities. The setup is designed with a modular architecture using Docker Compose profiles, allowing you to run only the services you need for your specific use case.

## Architecture Overview

The Docker Compose setup consists of several key components:

- **Core Services**: Main backend API and optional MCP server
- **Database Services**: Multiple database options (PostgreSQL, Neo4j, FalkorDB, ChromaDB)
- **Frontend Service**: Next.js web interface (work in progress)
- **Network**: Shared network for inter-service communication
- **Profiles**: Optional service groups for different deployment scenarios

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Cognee API    │    │  Cognee MCP     │
│   (Next.js)     │    │   (Backend)     │    │   Server        │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  cognee-network │
                    └─────────────────┘
                                 │
         ┌───────────┬─────────────┬─────────────┬─────────────┐
         │           │             │             │             │
    ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐
    │PostgreSQL│ │ Neo4j   │ │  FalkorDB   │ │  ChromaDB   │ │   ...   │
    │Port: 5432│ │Port: 7474│ │Port: 6379   │ │Port: 3002   │ │         │
    └─────────┘ └─────────┘ └─────────────┘ └─────────────┘ └─────────┘
```

## Services Overview

### Core Services

#### 1. Cognee (Main Backend API)
- **Container Name**: `cognee`
- **Build Context**: Root directory
- **Ports**: 
  - `8000:8000` (HTTP API)
  - `5678:5678` (Debug port)
- **Purpose**: Main Cognee backend API server
- **Resources**: 4 CPUs, 8GB RAM

#### 2. Cognee MCP Server
- **Container Name**: `cognee-mcp`
- **Profile**: `mcp`
- **Build Context**: Root directory (using `cognee-mcp/Dockerfile`)
- **Ports**: 
  - `8000:8000` (MCP HTTP/SSE)
  - `5678:5678` (Debug port)
- **Purpose**: Model Context Protocol server for IDE integration (Cursor, Claude Desktop, VS Code)
- **Resources**: 2 CPUs, 4GB RAM

#### 3. Frontend
- **Container Name**: `frontend`
- **Profile**: `ui`
- **Build Context**: `./cognee-frontend`
- **Port**: `3000:3000`
- **Purpose**: Next.js web interface (work in progress)
- **Note**: Limited functionality - prefer MCP integration for full features

### Database Services (All Optional)

#### 1. PostgreSQL with pgvector
- **Container Name**: `postgres`
- **Profile**: `postgres`
- **Image**: `pgvector/pgvector:pg17`
- **Port**: `5432:5432`
- **Purpose**: Relational database with vector extensions
- **Credentials**: `cognee/cognee` (user/password)
- **Database**: `cognee_db`

#### 2. Neo4j
- **Container Name**: `neo4j`
- **Profile**: `neo4j`
- **Image**: `neo4j:latest`
- **Ports**: 
  - `7474:7474` (HTTP interface)
  - `7687:7687` (Bolt protocol)
- **Purpose**: Graph database
- **Credentials**: `neo4j/pleaseletmein`
- **Plugins**: APOC, Graph Data Science

#### 3. FalkorDB
- **Container Name**: `falkordb`
- **Profile**: `falkordb`
- **Image**: `falkordb/falkordb:edge`
- **Ports**: 
  - `6379:6379` (Redis-compatible interface)
  - `3001:3000` (Web interface)
- **Purpose**: Graph database with Redis interface

#### 4. ChromaDB
- **Container Name**: `chromadb`
- **Profile**: `chromadb`
- **Image**: `chromadb/chroma:0.6.3`
- **Port**: `3002:8000`
- **Purpose**: Vector database
- **Authentication**: Token-based (requires `VECTOR_DB_KEY`)
- **Persistence**: Enabled with local volume

## Docker Compose Profiles

Profiles allow you to selectively run services based on your needs:

### Available Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| **(default)** | `cognee` only | Basic API server with SQLite |
| `mcp` | `cognee-mcp` | IDE integration (Cursor/Claude Desktop) |
| `ui` | `frontend` | Web interface (limited functionality) |
| `postgres` | `postgres` | PostgreSQL database |
| `neo4j` | `neo4j` | Graph database |
| `falkordb` | `falkordb` | Alternative graph database |
| `chromadb` | `chromadb` | Vector database |

### Profile Usage Examples

```bash
# Basic API server only
docker compose up

# API server + PostgreSQL
docker compose --profile postgres up

# API server + Neo4j + ChromaDB
docker compose --profile neo4j --profile chromadb up

# MCP server + PostgreSQL (for IDE integration)
docker compose --profile mcp --profile postgres up

# Full stack with UI
docker compose --profile ui --profile postgres up

# All services
docker compose --profile mcp --profile ui --profile postgres --profile neo4j --profile chromadb up
```

## Environment Configuration

### Required Environment File

Create a `.env` file in the root directory with your configuration:

```bash
# Core LLM Configuration
LLM_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Database Configuration (when using external databases)
DB_PROVIDER=postgres  # or sqlite, neo4j, etc.
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cognee_db
DB_USERNAME=cognee
DB_PASSWORD=cognee

# Vector Database (when using ChromaDB)
VECTOR_DB_KEY=your_chroma_auth_token

# Optional: Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
```

### Environment Variables by Service

#### Cognee (Main API)
- `DEBUG`: Enable/disable debug mode
- `HOST`: Bind host (default: 0.0.0.0)
- `ENVIRONMENT`: Deployment environment (local/dev/prod)
- `LOG_LEVEL`: Logging level (ERROR/INFO/DEBUG)

#### Cognee MCP Server
- `TRANSPORT_MODE`: Communication protocol (stdio/sse/http)
- `MCP_LOG_LEVEL`: MCP-specific logging level
- Database configuration (inherits from main service)

#### Database Services
- **PostgreSQL**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- **Neo4j**: `NEO4J_AUTH`, `NEO4J_PLUGINS`
- **ChromaDB**: Authentication and persistence settings

## Container Build Process

### Multi-Stage Build Strategy

Both main services use multi-stage Docker builds for optimization:

#### Stage 1: Dependency Installation (UV-based)
```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv
# Install system dependencies and Python packages
# Uses UV for fast dependency resolution
```

#### Stage 2: Runtime Environment
```dockerfile
FROM python:3.12-slim-bookworm
# Lightweight runtime with only necessary components
# Copy dependencies from build stage
```

### Build Context and Caching

- **Main Service**: Uses root directory context, installs with multiple extras
- **MCP Service**: Uses root context but builds from `cognee-mcp/` subdirectory
- **Frontend**: Independent Node.js build in `cognee-frontend/`

## Networking

### Shared Network
All services communicate through the `cognee-network` Docker network:

```yaml
networks:
  cognee-network:
    name: cognee-network
```

### Inter-Service Communication
- Services can reach each other using container names as hostnames
- External access through mapped ports
- `host.docker.internal` for accessing host machine services

## Volume Management

### Application Code Volumes (Development)
```yaml
volumes:
  - ./cognee:/app/cognee          # Main API code
  - ./cognee-frontend/src:/app/src # Frontend source
  - .env:/app/.env                # Environment configuration
```

### Persistent Data Volumes
```yaml
volumes:
  - .chromadb_data/:/chroma/chroma/  # ChromaDB persistence
  - postgres_data:/var/lib/postgresql/data  # PostgreSQL data
```

## Startup Process

### 1. Database Migrations
Both main services run Alembic migrations on startup:

```bash
alembic upgrade head
```

**Error Handling**: Special handling for `UserAlreadyExists` errors during default user creation - allows safe container restarts.

### 2. Service Initialization

#### Main API Service
- Runs Gunicorn with Uvicorn workers
- Development mode: Hot reloading enabled
- Production mode: Optimized for performance
- Debug mode: Debugpy integration on port 5678

#### MCP Service
- Supports multiple transport modes (stdio/sse/http)
- Configurable via `TRANSPORT_MODE` environment variable
- Debug support with port 5678

## Resource Allocation

### CPU and Memory Limits

| Service | CPU Limit | Memory Limit | Rationale |
|---------|-----------|--------------|-----------|
| cognee | 4.0 cores | 8GB | Main processing service |
| cognee-mcp | 2.0 cores | 4GB | Lighter MCP operations |
| frontend | unlimited | unlimited | Development convenience |
| databases | unlimited | unlimited | Database-specific needs |

## Development Features

### Debug Support
- **Port 5678**: Debugpy integration for both main services
- **Environment Variable**: Set `DEBUG=true` to enable
- **Wait for Client**: Debugger waits for IDE attachment

### Hot Reloading
- **API**: Gunicorn reload mode in development
- **Frontend**: Next.js development server with file watching
- **Volume Mounts**: Live code synchronization

### Development vs Production

#### Development Mode (`ENVIRONMENT=dev/local`)
- Hot reloading enabled
- Debug logging
- Single worker processes
- Extended timeouts

#### Production Mode
- Multiple workers (configurable)
- Error-level logging only
- Optimized performance settings
- No hot reloading

## Usage Patterns

### 1. Basic Development Setup
```bash
# Start with basic API + SQLite
docker compose up

# View logs
docker compose logs -f cognee
```

### 2. Full Development Environment
```bash
# Start with PostgreSQL database
docker compose --profile postgres up -d

# Add vector database for embeddings
docker compose --profile postgres --profile chromadb up -d
```

### 3. IDE Integration Development
```bash
# Start MCP server for Cursor/Claude Desktop integration
docker compose --profile mcp --profile postgres up -d

# Check MCP server status
docker compose logs cognee-mcp
```

### 4. UI Development
```bash
# Start with frontend for web interface testing
docker compose --profile ui --profile postgres up -d

# Access frontend at http://localhost:3000
```

### 5. Graph Analysis Setup
```bash
# Start with graph database for complex relationships
docker compose --profile neo4j --profile postgres up -d

# Access Neo4j browser at http://localhost:7474
```

## Port Mapping

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| cognee | 8000 | 8000 | HTTP API |
| cognee | 5678 | 5678 | Debug |
| cognee-mcp | 8000 | 8000 | MCP HTTP/SSE |
| cognee-mcp | 5678 | 5678 | Debug |
| frontend | 3000 | 3000 | Web UI |
| postgres | 5432 | 5432 | Database |
| neo4j | 7474 | 7474 | Web interface |
| neo4j | 7687 | 7687 | Bolt protocol |
| falkordb | 6379 | 6379 | Redis interface |
| falkordb | 3000 | 3001 | Web interface |
| chromadb | 8000 | 3002 | Vector DB API |

## Security Considerations

### Database Authentication
- **PostgreSQL**: Default credentials (`cognee/cognee`)
- **Neo4j**: Default credentials (`neo4j/pleaseletmein`)
- **ChromaDB**: Token-based authentication via `VECTOR_DB_KEY`

### Network Isolation
- All services communicate through isolated `cognee-network`
- External access only through explicitly mapped ports
- `host.docker.internal` for secure host access

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check for port conflicts
docker compose ps
netstat -tulpn | grep :8000
```

#### Database Connection Issues
```bash
# Check database container status
docker compose --profile postgres ps

# View database logs
docker compose --profile postgres logs postgres
```

#### Service Dependencies
```bash
# Ensure services start in correct order
docker compose up -d postgres  # Start database first
docker compose up cognee        # Then start main service
```

### Debug Mode

#### Enable Debug Mode
1. Set `DEBUG=true` in your `.env` file or as environment variable
2. Restart the service:
```bash
docker compose down
docker compose up
```

#### Attach Debugger
1. Start service in debug mode
2. Connect your IDE debugger to `localhost:5678`
3. Set breakpoints and debug as needed

### Log Analysis
```bash
# View all service logs
docker compose logs

# Follow logs in real-time
docker compose logs -f

# Service-specific logs
docker compose logs cognee
docker compose logs postgres
```

## Maintenance

### Container Management
```bash
# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild containers after code changes
docker compose build
docker compose up --force-recreate
```

### Data Persistence
- **ChromaDB**: Data persisted in `.chromadb_data/` directory
- **PostgreSQL**: Data persisted in named volume `postgres_data`
- **Neo4j**: No explicit persistence configured (data lost on container restart)

### Updates and Rebuilds
```bash
# Pull latest images
docker compose pull

# Rebuild custom images
docker compose build --no-cache

# Update specific service
docker compose up -d --no-deps --build cognee
```

## Performance Optimization

### Resource Tuning
Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "4.0"      # Adjust based on available CPU
      memory: 8GB      # Adjust based on available RAM
```

### Database Optimization
- **PostgreSQL**: Consider shared_buffers and work_mem tuning
- **Neo4j**: Configure heap size via NEO4J_dbms_memory_heap_max_size
- **ChromaDB**: Increase memory allocation for large datasets

## Integration Examples

### Local Development
```bash
# Start minimal setup for local development
docker compose up

# Add database when needed
docker compose --profile postgres up -d
```

### IDE Integration (Recommended)
```bash
# Start MCP server for Cursor/Claude Desktop
docker compose --profile mcp --profile postgres up -d

# Configure your IDE to connect to localhost:8000
```

### Production Deployment
```bash
# Production-ready setup
docker compose --profile postgres --profile chromadb up -d

# With environment overrides
ENVIRONMENT=production LOG_LEVEL=ERROR docker compose up -d
```

## Alternative Configurations

### Helm Deployment
For Kubernetes deployment, use the Helm configuration:
- Location: `deployment/helm/`
- Simplified setup with just Cognee + PostgreSQL
- Kubernetes-native resource management

### Distributed Mode
For distributed processing:
- Uses separate `distributed/Dockerfile`
- Configured for Modal.com integration
- Environment: `COGNEE_DISTRIBUTED=true`

## Environment Templates

### Minimal Configuration
```bash
# .env (minimal setup)
LLM_API_KEY=your_openai_api_key_here
```

### Full Configuration
```bash
# .env (full setup)
# LLM Configuration
LLM_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Database Configuration
DB_PROVIDER=postgres
DB_HOST=host.docker.internal
DB_PORT=5432
DB_NAME=cognee_db
DB_USERNAME=cognee
DB_PASSWORD=cognee

# Vector Database
VECTOR_DB_KEY=your_chroma_auth_token

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

# Debug Configuration
DEBUG=false
LOG_LEVEL=INFO
```

## Best Practices

1. **Start Simple**: Begin with the basic setup and add profiles as needed
2. **Use Profiles**: Leverage profiles to avoid running unnecessary services
3. **Environment Files**: Always use `.env` files for configuration
4. **Resource Management**: Monitor resource usage and adjust limits accordingly
5. **Data Persistence**: Ensure important data is properly mounted or volumed
6. **Network Security**: Keep services on the isolated network
7. **Debug Safely**: Only enable debug mode in development environments

## Quick Start Commands

```bash
# Clone and setup
git clone <repository>
cd cognee

# Create environment file
cp .env.template .env  # Edit with your configuration

# Basic start
docker compose up

# Full development environment
docker compose --profile postgres --profile chromadb up -d

# IDE integration
docker compose --profile mcp --profile postgres up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop everything
docker compose down
```

This Docker Compose setup provides a flexible, scalable foundation for running Cognee in various configurations, from simple development setups to complex multi-database deployments.