FROM python:3.11-slim

# Set build argument
ARG DEBUG

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}
ENV PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"

RUN apt-get update && apt-get install

RUN apt-get install -y \
  gcc \
  libpq-dev


WORKDIR /app
COPY pyproject.toml poetry.lock /app/


RUN pip install poetry

# Don't create virtualenv since docker is already isolated
RUN poetry config virtualenvs.create false

# Install the dependencies
RUN poetry install --all-extras --no-root --without dev


# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH=/app

COPY cognee/ /app/cognee

# Copy Alembic configuration
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
