# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

# Set build argument
ARG DEBUG

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}
ENV PIP_NO_CACHE_DIR=true

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y

RUN apt-get install -y \
    gcc \
    libpq-dev

COPY . /app

RUN uv sync --reinstall

# Place executables in the environment at the front of the path
ENV PATH="/app/:/app/.venv/bin:$PATH"

ENTRYPOINT ["cognee"]
