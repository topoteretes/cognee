# Build the MCP App workspace bundle (React/Vite, served inline by the MCP
# server). Vite writes to ../src/app_bundles/ relative to apps-src, so we
# mirror the repo layout: /build/cognee-mcp/{apps-src,src}.
FROM node:22-alpine AS app-bundle
WORKDIR /build/cognee-mcp/apps-src
COPY ./cognee-mcp/apps-src/package.json ./cognee-mcp/apps-src/package-lock.json ./
RUN npm ci
COPY ./cognee-mcp/apps-src ./
RUN npm run build
# Output lands at /build/cognee-mcp/src/app_bundles/visualize-graph.html

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
# Drop in the workspace bundle built by the node stage. Order matters: the
# COPY above brings in `cognee-mcp/src/app_bundles/` (empty in CI/clean clones
# because the bundle is gitignored, or stale on a developer's machine), and
# this COPY then *shadows* it with the freshly-built bundle. uv sync below
# packages this into the wheel installed into the venv; visualize_graph_ui
# reads the bundle from its own neighbor directory inside site-packages.
COPY --from=app-bundle /build/cognee-mcp/src/app_bundles /app/src/app_bundles
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Run as a non-root user. Create it BEFORE the COPY so --chown can resolve
# the name and the tree lands owned by cognee within the COPY layer itself:
# a separate `chown -R` afterwards makes overlayfs duplicate the whole ~9GB
# /app into a second layer, doubling the image size.
RUN groupadd --system --gid 1000 cognee \
    && useradd --system --uid 1000 --gid cognee --no-create-home --shell /usr/sbin/nologin cognee \
    && chown cognee:cognee /app
# ^ non-recursive chown of the still-empty WORKDIR: /app doubles as HOME, so
# the user must be able to write dotfiles/caches (e.g. Kuzu extensions) there.

# Copy the virtual environment from the uv stage
COPY --from=uv /usr/local /usr/local
COPY --from=uv --chown=cognee:cognee /app /app

# Strip Windows carriage returns (fixes "no such file" on Windows Docker)
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set environment variables for MCP server
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Give the non-root user a stable, writable HOME so Kuzu/Ladybug caches its
# downloaded extensions in a consistent location across build and runtime.
ENV HOME=/app

# Add labels for API mode usage
LABEL org.opencontainers.image.description="Cognee MCP Server with API mode support"

USER cognee

# Pre-install Kuzu/Ladybug's JSON extension at build time (network is available
# here) so it is baked into the image. Runs the same install path the app uses,
# as the same user with the same HOME, so the cached extension lands exactly
# where the running server looks for it — avoiding the "Extension: json ... has
# not been installed" Binder error when recall runs in a network-restricted
# container. Best-effort: a failed download must not break the image build.
RUN python -c "from cognee_db_workers._kuzu_helpers import install_json_extension_local; install_json_extension_local(buffer_pool_size=268435456)" \
    || echo "WARNING: JSON extension pre-install skipped (no network at build time); it will be installed on first run if the container has network access."

# Use the application name from pyproject.toml for normal operation
# For testing, we'll override this with a direct command
ENTRYPOINT ["/app/entrypoint.sh"]
