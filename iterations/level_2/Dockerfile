
FROM python:3.11-slim

# Set build argument
ARG API_ENABLED

# Set environment variable based on the build argument
ENV API_ENABLED=${API_ENABLED} \
    PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"
RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

# Install the dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev

RUN apt-get update -q && \
    apt-get install curl zip jq netcat-traditional -y -q
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -qq awscliv2.zip && ./aws/install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#RUN playwright install
#RUN playwright install-deps

WORKDIR /app
COPY . /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]