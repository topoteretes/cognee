# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
# ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Set build argument
ARG DEBUG

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}

# Needed when psycopg2 is built from source via cognee[postgres] in linux images.
# libpq-dev provides pg_config; build-essential provides libc headers and toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and lockfile first for better caching
COPY ./cognee-mcp/pyproject.toml ./cognee-mcp/uv.lock ./cognee-mcp/entrypoint.sh ./

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows layer caching
COPY ./cognee-mcp /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the uv stage
COPY --from=uv /usr/local /usr/local
COPY --from=uv /app /app

# Strip Windows carriage returns (fixes "no such file" on Windows Docker)
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Run as a non-root user. Create the user before copying ownership so the
# /app tree (which already exists from the COPY above) is readable.
RUN groupadd --system --gid 1000 cognee \
    && useradd --system --uid 1000 --gid cognee --no-create-home --shell /usr/sbin/nologin cognee \
    && chown -R cognee:cognee /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set environment variables for MCP server
ENV PYTHONUNBUFFERED=1
ENV MCP_LOG_LEVEL=DEBUG
ENV PYTHONPATH=/app

# Add labels for API mode usage
LABEL org.opencontainers.image.description="Cognee MCP Server with API mode support"

USER cognee

# Use the application name from pyproject.toml for normal operation
# For testing, we'll override this with a direct command
ENTRYPOINT ["/app/entrypoint.sh"]
