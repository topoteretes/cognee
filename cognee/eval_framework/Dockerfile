FROM python:3.12-slim

# Set environment variables
ENV PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"
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

COPY pyproject.toml poetry.lock README.md /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --extras distributed --extras evals --extras deepeval --no-root

COPY cognee/ /app/cognee
COPY distributed/ /app/distributed

# Force-upgrade packages that poetry pins too old
# pydantic_core requires typing_extensions >= 4.12 (Sentinel)
RUN pip install --force-reinstall --no-deps "typing_extensions>=4.14" && \
    pip install datasets && \
    python -c "from typing_extensions import Sentinel; print('typing_extensions OK')" && \
    python -c "import pydantic; print(f'pydantic {pydantic.__version__} OK')"
