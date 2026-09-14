# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS uv

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
    uv sync "$@" --extra fastembed --extra debug --extra api --extra postgres --extra neo4j --extra llama-index --extra aws --extra dlt --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY ./cognee /app/cognee
COPY ./cognee_db_workers /app/cognee_db_workers
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
    uv sync "$@" --extra fastembed --extra debug --extra aws --extra api --extra postgres --extra neo4j --extra llama-index --extra dlt --extra ollama --extra mistral --extra groq --extra anthropic --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Run as the same non-root user as the cognee-mcp image (uid/gid 1000) so both
# containers can share the storage volumes without ownership conflicts (a
# root-created database directory is unwritable for the uid-1000 MCP server).
# Created before the COPY so ownership is set in that single layer — a
# separate `chown -R /app` would copy the whole tree up into a second layer.
# /cognee-storage is baked into the image cognee-owned so a fresh named
# volume mounted there initializes with the right ownership.
# ``chown cognee /app`` (the directory inode only): WORKDIR created /app as
# root, and ``COPY --chown`` sets ownership on the copied content, not the
# pre-existing target dir — without this the non-root user cannot create
# ``$HOME/.lbdb`` and the build-time extension pre-install silently fails.
RUN groupadd --system --gid 1000 cognee \
    && useradd --system --uid 1000 --gid cognee --no-create-home --shell /usr/sbin/nologin cognee \
    && mkdir -p /cognee-storage/system /cognee-storage/data \
    && chown -R cognee:cognee /cognee-storage \
    && chown cognee:cognee /app

COPY --from=uv --chown=cognee:cognee /app /app

# Strip Windows carriage returns (fixes "no such file" on Windows Docker)
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH=/app
# ENV LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1
# Writable HOME for the non-root user (~/.cognee logs, tool caches).
ENV HOME=/app
# Default storage OUTSIDE the source tree: the ./cognee bind mount exists for
# dev reload and must not double as the persistence location (host-uid
# sensitive, pollutes the checkout, and was shared with the MCP container by
# accident rather than by design). docker-compose mounts named volumes here.
ENV SYSTEM_ROOT_DIRECTORY=/cognee-storage/system
ENV DATA_ROOT_DIRECTORY=/cognee-storage/data

USER cognee

# Pre-install Kuzu/Ladybug's JSON extension at build time (network is available
# here) so it is baked into the image — same mechanism as the cognee-mcp
# image. As root the server used to INSTALL it at runtime into /root/.lbdb on
# every boot; as the non-root user that runtime install races between graph
# workers and fails ("Directory ... cannot be created"). Best-effort: a failed
# download must not break the image build.
RUN python -c "from cognee_db_workers._kuzu_helpers import install_json_extension_local; install_json_extension_local(buffer_pool_size=268435456)" \
    || echo "WARNING: JSON extension pre-install skipped (no network at build time); it will be installed on first run if the container has network access."

ENTRYPOINT ["/app/entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
