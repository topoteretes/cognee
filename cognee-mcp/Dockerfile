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

# Create the non-root user up front so the COPY below can set ownership in a
# single layer. A separate `chown -R /app` rewrites metadata on every file,
# which forces overlayfs to copy up the entire ~9GB /app tree into a second
# layer (the venv effectively stored twice). `COPY --chown` avoids that.
# ``chown cognee /app`` (the directory inode only): WORKDIR created /app as
# root, and ``COPY --chown`` sets ownership on the copied content, not the
# pre-existing target dir — without this the JSON-extension pre-install below
# cannot create ``$HOME/.lbdb`` and has been silently falling back to its
# build warning (and the runtime install fails the same way as non-root).
RUN groupadd --system --gid 1000 cognee \
    && useradd --system --uid 1000 --gid cognee --no-create-home --shell /usr/sbin/nologin cognee \
    && mkdir -p /cognee-storage/system /cognee-storage/data \
    && chown -R cognee:cognee /cognee-storage \
    && chown cognee:cognee /app

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
# Default storage OUTSIDE the source tree, matching the main image — both
# containers share the same named volumes with the same uid (1000).
ENV SYSTEM_ROOT_DIRECTORY=/cognee-storage/system
ENV DATA_ROOT_DIRECTORY=/cognee-storage/data

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

# Only the HTTP/SSE transports serve an HTTP server; in the default stdio mode
# there is nothing to probe, so report healthy without hitting the network.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD [ "${TRANSPORT_MODE:-stdio}" = "stdio" ] || python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
