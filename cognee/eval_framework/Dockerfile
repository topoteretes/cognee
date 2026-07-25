FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV SKIP_MIGRATIONS=true
# uv sync installs into /app/.venv (unlike the old poetry setup, which installed
# into the system interpreter) — put it on PATH so plain `python` finds the deps.
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

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
