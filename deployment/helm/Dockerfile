FROM python:3.11-slim

# Define Poetry extras to install
ARG POETRY_EXTRAS="\
# Storage & Databases \
postgres weaviate qdrant neo4j falkordb milvus kuzu \
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

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}
ENV PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"


RUN apt-get install -y \
  gcc \
  libpq-dev


WORKDIR /app
COPY pyproject.toml poetry.lock /app/


RUN pip install poetry

# Don't create virtualenv since docker is already isolated
RUN poetry config virtualenvs.create false

# Install the dependencies
RUN poetry install --extras "${POETRY_EXTRAS}" --no-root --without dev


# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH=/app

COPY cognee/ /app/cognee

# Copy Alembic configuration
COPY tools/alembic.ini /app/tools/alembic.ini
COPY tools/alembic/ /app/tools/alembic

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
