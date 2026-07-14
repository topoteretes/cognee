FROM python:3.11-slim

# Define Poetry extras to install
ARG POETRY_EXTRAS="\
# Storage & Databases \
postgres neo4j falkordb kuzu \
# Notebooks & Interactive Environments \
notebook \
# LLM & AI Frameworks \
langchain llama-index gemini huggingface ollama mistral groq \
# Evaluation & Monitoring \
deepeval evals posthog \
# Graph Processing & Code Analysis \
codegraph graphiti \
# Document Processing \
docs"

# Set build argument
ARG DEBUG
ARG POETRY_VERSION=2.1.4

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}
ENV PIP_NO_CACHE_DIR=true

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY pyproject.toml poetry.lock /app/

RUN pip install "poetry==${POETRY_VERSION}" \
  && poetry config virtualenvs.create false \
  && poetry install --extras "${POETRY_EXTRAS}" --no-root --without dev


# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH=/app

COPY cognee/ /app/cognee

# Copy Alembic configuration
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
  && sed -i 's/\r$//' /app/entrypoint.sh \
  && addgroup --system cognee \
  && mkdir -p /home/cognee \
  && adduser --system --ingroup cognee --home /home/cognee cognee \
  && chown -R cognee:cognee /app /home/cognee

USER cognee

ENTRYPOINT ["/app/entrypoint.sh"]
