FROM python:3.11-slim

# Set environment variables
ENV PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"
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

COPY pyproject.toml poetry.lock README.md /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --extras neo4j --extras postgres --extras aws --extras distributed --no-root

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed
RUN chmod +x /app/distributed/entrypoint.sh

ENTRYPOINT ["/app/distributed/entrypoint.sh"]
