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

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    git \
    curl \
    clang \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and lockfile first for better caching
COPY README.md pyproject.toml uv.lock entrypoint.sh ./

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra debug --extra api --extra postgres --extra weaviate --extra qdrant --extra neo4j --extra kuzu --extra llama-index --extra gemini --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-install-project --no-dev --no-editable

# Copy Alembic configuration
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY ./cognee /app/cognee
RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --extra debug --extra api --extra postgres --extra weaviate --extra qdrant --extra neo4j --extra kuzu --extra llama-index --extra gemini --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=uv /app /app
# COPY --from=uv /app/.venv /app/.venv
# COPY --from=uv /root/.local /root/.local

RUN chmod +x /app/entrypoint.sh

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
