FROM python:3.11-slim

# Set build argument
ARG DEBUG

# Set environment variable based on the build argument
ENV DEBUG=${DEBUG}
ENV PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

# Create virtualenv
RUN poetry config virtualenvs.create false

# Install the dependencies
RUN poetry install --no-root --no-dev
        
# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH=/app

COPY cognee/ /app/cognee

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh

# ENTRYPOINT ["/app/entrypoint.sh"]
