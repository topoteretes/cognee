"""Operator settings for a remote embedding worker (env prefix ``COGNEE_EMBEDDING_WORKER_``).

Where the worker lives is operator configuration: it is read from the
environment only, never from a request. The names and defaults match the Rust
SDK's, so one deployment environment configures both. The model, vector size,
batch size and input limit stay in the usual ``EMBEDDING_*`` settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.remote_worker.urls import redact_url

from .errors import EmbeddingWorkerConfigError

ENV_PREFIX = "COGNEE_EMBEDDING_WORKER_"
DEFAULT_TRANSPORT = "amqp"
# The worker names its RabbitMQ transport "rabbitmq"; accept that spelling too.
TRANSPORT_ALIASES = {"rabbitmq": "amqp"}
# Transports the worker serves and this client does not implement yet.
UNSUPPORTED_TRANSPORTS = ("http", "grpc")
DEFAULT_AMQP_QUEUE = "embedding_worker.embed"


class EmbeddingWorkerSettings(BaseSettings):
    transport: Literal["amqp", "http", "grpc"] = DEFAULT_TRANSPORT
    # amqp(s)://user:pass@host:5672/vhost
    endpoint: str | None = None
    amqp_queue: str = Field(default=DEFAULT_AMQP_QUEUE, min_length=1)
    # The model the worker must serve. Unset, any model is accepted and logged.
    expected_model: str | None = None
    # Process-wide cap on requests in flight to one worker endpoint (per event
    # loop). Size it to the fleet.
    max_concurrent_requests: int = Field(default=16, ge=1)
    connect_timeout_secs: float = Field(default=5.0, gt=0)
    # One request, including the time it waits in the worker's queue.
    request_timeout_secs: float = Field(default=300.0, gt=0)

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    @field_validator("transport", mode="before")
    @classmethod
    def _normalize_transport(cls, value):
        if isinstance(value, str):
            value = value.strip().lower() or DEFAULT_TRANSPORT
            return TRANSPORT_ALIASES.get(value, value)
        return value

    @field_validator("endpoint", "expected_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value

    def require_amqp_endpoint(self) -> str:
        """The broker URL, or a configuration error saying what to set."""
        if self.transport in UNSUPPORTED_TRANSPORTS:
            raise EmbeddingWorkerConfigError(
                f"{ENV_PREFIX}TRANSPORT={self.transport} is not supported yet by the embedding "
                "worker client; only amqp (RabbitMQ) is implemented.",
                remediation=(
                    f"Set {ENV_PREFIX}TRANSPORT=amqp and run the worker with "
                    "EMBEDDING_WORKER_TRANSPORT=rabbitmq, or use EMBEDDING_PROVIDER="
                    "openai_compatible against the worker's HTTP /v1/embeddings."
                ),
            )
        if not self.endpoint:
            raise EmbeddingWorkerConfigError(
                "EMBEDDING_PROVIDER=embedding_worker needs the worker's broker URL.",
                remediation=f"Set {ENV_PREFIX}ENDPOINT to amqp(s)://user:pass@host:5672/%2f.",
            )
        if not self.endpoint.startswith(("amqp://", "amqps://")):
            raise EmbeddingWorkerConfigError(
                f"{ENV_PREFIX}ENDPOINT must be an amqp(s):// broker URL for the RabbitMQ "
                f"transport, got {redact_url(self.endpoint)!r}."
            )
        return self.endpoint


def get_embedding_worker_settings() -> EmbeddingWorkerSettings:
    """Read the settings from the environment; malformed values are a configuration error."""
    try:
        return EmbeddingWorkerSettings()
    except ValidationError as error:
        fields = ", ".join(ENV_PREFIX + str(issue["loc"][0]).upper() for issue in error.errors())
        raise EmbeddingWorkerConfigError(
            f"Invalid embedding worker setting(s): {fields}.",
            remediation=(
                "Fix the value(s); see the COGNEE_EMBEDDING_WORKER_* block in .env.template."
            ),
        ) from error
