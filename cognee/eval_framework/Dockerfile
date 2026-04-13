FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV SKIP_MIGRATIONS=true
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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra distributed --extra evals --extra deepeval --no-install-project --no-editable

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra distributed --extra evals --extra deepeval --no-editable

# Modal post-build steps and follow-up package fixes expect pip on PATH.
RUN uv pip install pip wheel

ENV PATH="/app/.venv/bin:$PATH"

# Force-upgrade packages that the locked dependency set leaves too old
# pydantic_core requires typing_extensions >= 4.12 (Sentinel)
RUN pip install --force-reinstall --no-deps "typing_extensions>=4.14" && \
    pip install datasets && \
    python -c "from typing_extensions import Sentinel; print('typing_extensions OK')" && \
    python -c "import pydantic; print(f'pydantic {pydantic.__version__} OK')"
