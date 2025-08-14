FROM python:3.11-slim

# Set environment variables
ENV PIP_NO_CACHE_DIR=true
ENV PYTHONPATH=/app
ENV RUN_MODE=modal
ENV SKIP_MIGRATIONS=true
ENV COGNEE_DISTRIBUTED=true

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

RUN pip install uv

RUN uv sync --extra neo4j --extra postgres --extra aws --extra distributed --no-dev

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed
RUN chmod +x /app/distributed/entrypoint.sh

ENTRYPOINT ["/app/distributed/entrypoint.sh"]
