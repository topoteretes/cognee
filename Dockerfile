# Official Ladybug extension binaries — this image is the origin content
# behind extension.ladybugdb.com, so copying from it here means the JSON
# extension ships in the image and is never downloaded at runtime (see
# cognee_db_workers/ladybug_extensions/README.md).
FROM ghcr.io/ladybugdb/extension-repo:latest AS ladybug-extensions

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation: without it the venv ships no .pyc files, so
# every container cold start recompiles the entire dependency tree from
# source (measured on cognee-saas-pod: ~8s of a ~13s import, halving startup).
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Set build argument
ARG DEBUG

# Additional optional-dependency groups to install, separated by spaces.
# Example: docker build --build-arg COGNEE_EXTRAS="aws langchain" .
# Keep this applied to both sync steps: the second exact sync would otherwise
# remove extras installed only in the dependency-cache layer.
ARG COGNEE_EXTRAS=""

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    git \
    curl \
    cmake \
    clang \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and lockfile first for better caching
COPY README.md pyproject.toml uv.lock entrypoint.sh ./

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    set -f; \
    set --; \
    for extra in ${COGNEE_EXTRAS}; do \
        set -- "$@" --extra "$extra"; \
    done; \
    uv sync "$@" --extra fastembed --extra debug --extra api --extra postgres --extra neo4j --extra llama-index --extra aws --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY ./cognee /app/cognee
COPY ./cognee_db_workers /app/cognee_db_workers
# Bundle the JSON extension for both image arches (~1.7 MB total); the loader
# picks the one matching the runtime platform.
COPY --from=ladybug-extensions /usr/share/nginx/html/v0.18.1/linux_amd64/json/libjson.lbug_extension /app/cognee_db_workers/ladybug_extensions/v0.18.1/linux_amd64/libjson.lbug_extension
COPY --from=ladybug-extensions /usr/share/nginx/html/v0.18.1/linux_arm64/json/libjson.lbug_extension /app/cognee_db_workers/ladybug_extensions/v0.18.1/linux_arm64/libjson.lbug_extension
# Compatibility shim that re-exports ladybug under the legacy `kuzu`
# module name. Listed in [tool.hatch.build.targets.wheel] packages, and
# imported at module load by alembic/versions/b9274c27a25a_kuzu_11_migration.py.
COPY ./kuzu /app/kuzu
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    set -f; \
    set --; \
    for extra in ${COGNEE_EXTRAS}; do \
        set -- "$@" --extra "$extra"; \
    done; \
    uv sync "$@" --extra fastembed --extra debug --extra aws --extra api --extra postgres --extra neo4j --extra llama-index --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /app /app
# COPY --from=uv /app/.venv /app/.venv
# COPY --from=uv /root/.local /root/.local

# Strip Windows carriage returns (fixes "no such file" on Windows Docker)
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH=/app
# ENV LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
