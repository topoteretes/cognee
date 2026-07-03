FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set environment variables
ENV PIP_NO_CACHE_DIR=true
ENV PYTHONPATH=/app
ENV SKIP_MIGRATIONS=true

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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
    --extra distributed --extra evals --extra deepeval

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed
