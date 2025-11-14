# ================================
# Cognee MCP Server - Optimized Dockerfile
# ================================
# Multi-stage build for production-ready MCP server
# Features: security hardening, minimal image size, layer optimization

# ================================
# Stage 1: Builder
# ================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set build arguments
ARG DEBUG=false

# Build environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBUG=${DEBUG}

WORKDIR /app

# Install build dependencies (minimal set with version pinning)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc=4:12.2.* \
    libpq-dev=15.* \
    git=1:2.39.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency files first for better layer caching
COPY ./cognee-mcp/pyproject.toml ./cognee-mcp/uv.lock ./cognee-mcp/entrypoint.sh ./

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Copy Alembic configuration (for database migrations)
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic

# Copy application source code
COPY ./cognee-mcp /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-editable

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# ================================
# Stage 2: Runtime
# ================================
FROM python:3.12-slim-bookworm AS runtime

# Metadata labels
LABEL org.opencontainers.image.title="Cognee MCP Server" \
      org.opencontainers.image.description="Model Context Protocol server for Cognee knowledge base search" \
      org.opencontainers.image.vendor="Cognee" \
      org.opencontainers.image.source="https://github.com/topoteretes/cognee" \
      org.opencontainers.image.licenses="Apache-2.0"

# Runtime dependencies only (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5=15.* \
    curl=7.88.* \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r cognee --gid=1000 && \
    useradd -r -g cognee --uid=1000 --home-dir=/app --shell=/sbin/nologin cognee

WORKDIR /app

# Copy ONLY what's needed from builder stage (NOT /usr/local which is 3GB!)
# This is the KEY fix - only copy the virtual environment and application files
COPY --from=builder --chown=cognee:cognee /app/.venv /app/.venv
COPY --from=builder --chown=cognee:cognee /app/src /app/src
COPY --from=builder --chown=cognee:cognee /app/entrypoint.sh /app/entrypoint.sh
COPY --from=builder --chown=cognee:cognee /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=cognee:cognee /app/alembic /app/alembic

# Runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MCP_LOG_LEVEL=INFO

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user for security
USER cognee

# Expose port (documentation - actual binding controlled by args)
EXPOSE 8000

# Use entrypoint script for flexible configuration
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command arguments (can be overridden)
CMD []
