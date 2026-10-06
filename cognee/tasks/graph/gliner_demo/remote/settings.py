"""Operator settings for a remote GLiNER worker (env prefix ``COGNEE_GLINER_``).

Where the worker lives is operator configuration: it is read from the
environment only, never from a request, so an API caller cannot point the
server at an address of their choosing. The names match the Rust SDK's, so one
deployment environment configures both.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import GlinerRemoteConfigError

LOCAL_TRANSPORT = "local"
REMOTE_TRANSPORTS = ("http", "grpc", "amqp")
# The worker names its RabbitMQ transport "rabbitmq"; accept that spelling too.
TRANSPORT_ALIASES = {"rabbitmq": "amqp"}
DEFAULT_AMQP_QUEUE = "gliner_worker.extract"


class RemoteGlinerSettings(BaseSettings):
    # "local" runs gliner2 in this process; "http", "grpc" or "amqp" sends
    # extraction to a gliner_worker at ``endpoint``.
    transport: Literal["local", "http", "grpc", "amqp"] = LOCAL_TRANSPORT
    # http(s)://host:8080 (HTTP); http(s)://host:50051, comma-separated for
    # several replicas (gRPC); amqp(s)://user:pass@host:5672 (RabbitMQ).
    endpoint: str | None = None
    amqp_queue: str = Field(default=DEFAULT_AMQP_QUEUE, min_length=1)
    # Bearer token for HTTP and gRPC, matching the worker's GLINER_WORKER_API_KEY.
    api_key: SecretStr | None = None
    # The model the worker must serve. Unset, any model is accepted and recorded.
    expected_model: str | None = None
    # Process-wide cap on requests in flight to one worker endpoint. Size it to
    # the fleet: each worker runs one inference at a time.
    max_concurrent_requests: int = Field(default=16, ge=1)
    # Requests one extraction call keeps in flight; the process-wide cap above
    # still applies across calls.
    max_in_flight_requests: int = Field(default=4, ge=1)
    # Chunks per request. The worker windows each chunk itself, so one chunk is
    # many model inputs; keep requests well inside the request timeout.
    inputs_per_request: int = Field(default=8, ge=1)
    connect_timeout_secs: float = Field(default=5.0, gt=0)
    # One request, including the time it waits in the worker's queue.
    request_timeout_secs: float = Field(default=300.0, gt=0)
    max_response_bytes: int = Field(default=64 * 1024 * 1024, ge=1)

    model_config = SettingsConfigDict(env_prefix="COGNEE_GLINER_", extra="ignore")

    @field_validator("transport", mode="before")
    @classmethod
    def _normalize_transport(cls, value):
        if isinstance(value, str):
            value = value.strip().lower() or LOCAL_TRANSPORT
            return TRANSPORT_ALIASES.get(value, value)
        return value

    @field_validator("endpoint", "expected_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_unset(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_remote(self) -> bool:
        return self.transport != LOCAL_TRANSPORT

    def require_endpoint(self) -> str:
        if not self.endpoint:
            raise GlinerRemoteConfigError(
                f"COGNEE_GLINER_TRANSPORT={self.transport} needs a worker endpoint.",
                remediation=(
                    "Set COGNEE_GLINER_ENDPOINT to the gliner_worker address, or unset "
                    "COGNEE_GLINER_TRANSPORT to extract in this process."
                ),
            )
        return self.endpoint


def get_remote_gliner_settings() -> RemoteGlinerSettings:
    """Read the settings from the environment; malformed values are a configuration error.

    Not cached: they are read when a pipeline is built, so a changed environment
    takes effect on the next cognify.
    """
    try:
        return RemoteGlinerSettings()
    except ValidationError as error:
        fields = ", ".join(
            "COGNEE_GLINER_" + str(issue["loc"][0]).upper() for issue in error.errors()
        )
        raise GlinerRemoteConfigError(
            f"Invalid remote GLiNER setting(s): {fields}.",
            remediation="Fix the value(s); see the COGNEE_GLINER_* block in .env.template.",
        ) from error


def remote_gliner_configured() -> bool:
    """Whether GLiNER extraction is sent to a worker instead of running in this process."""
    return get_remote_gliner_settings().is_remote
