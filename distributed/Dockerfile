FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV RUN_MODE=modal
ENV SKIP_MIGRATIONS=true
ENV COGNEE_DISTRIBUTED=true
ENV UV_LINK_MODE=copy

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/

RUN uv sync --frozen --no-dev --extra neo4j --extra postgres --extra aws --extra distributed --no-install-project

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed

ENV PATH="/app/.venv/bin:$PATH"
