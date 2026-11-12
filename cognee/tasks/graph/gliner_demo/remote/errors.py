"""Typed failures of a remote GLiNER worker.

Each class decides two things: whether the adapter retries it, and how far it
reaches. ``fatal`` errors (an outage that outlived its retries, bad
credentials, the wrong model, an incompatible worker, bad settings) fail every
later extraction of the run at once, because every document would meet the same
problem. The rest (a rejected, failed or timed-out request) fail only the
document whose request caused them.

None of them log on construction: retried attempts would flood the log, so the
adapter logs what it decides instead.
"""

from __future__ import annotations

from fastapi import status

from cognee.exceptions import CogneeApiError, CogneeConfigurationError


class GlinerWorkerError(CogneeApiError):
    """Base class for failures reported by, or about, a remote GLiNER worker."""

    retryable = False
    fatal = False

    def __init__(
        self,
        message: str,
        name: str = "GlinerWorkerError",
        status_code=status.HTTP_502_BAD_GATEWAY,
        remediation: str | None = None,
    ):
        super().__init__(message, name, status_code, log=False, remediation=remediation)


class GlinerWorkerUnavailableError(GlinerWorkerError):
    """Unreachable, overloaded or restarting (HTTP 429/502/503/504, connect
    timeouts, gRPC UNAVAILABLE and friends, RabbitMQ ``unavailable`` or a request
    the broker did not accept).

    Retried. Once the retries are spent it is fatal for the run.
    """

    retryable = True
    fatal = True

    def __init__(self, message: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            message,
            "GlinerWorkerUnavailableError",
            status.HTTP_503_SERVICE_UNAVAILABLE,
            remediation="Check that the GLiNER worker at COGNEE_GLINER_ENDPOINT is running.",
        )


class GlinerWorkerUnauthorizedError(GlinerWorkerError):
    """The worker refused the credentials (HTTP 401/403, gRPC UNAUTHENTICATED)."""

    fatal = True

    def __init__(self, message: str):
        super().__init__(
            message,
            "GlinerWorkerUnauthorizedError",
            remediation="Set COGNEE_GLINER_API_KEY to the worker's GLINER_WORKER_API_KEY.",
        )


class GlinerWorkerRejectedError(GlinerWorkerError):
    """The worker rejected the request as invalid (HTTP 400/413/422, gRPC
    INVALID_ARGUMENT, RabbitMQ ``invalid_request``). Fails that document only."""

    def __init__(self, message: str):
        super().__init__(
            message, "GlinerWorkerRejectedError", status.HTTP_422_UNPROCESSABLE_CONTENT
        )


class GlinerWorkerTimeoutError(GlinerWorkerError):
    """A request reached the worker but got no answer within
    COGNEE_GLINER_REQUEST_TIMEOUT_SECS (HTTP read timeout, gRPC DEADLINE_EXCEEDED,
    no RabbitMQ reply to a published request).

    Not retried: the worker may still be running it, and resending a request that
    is slow on its own would only multiply that work. Fails that document only;
    the adapter first re-checks readiness, so a worker that went away still fails
    the run as unavailable.
    """

    def __init__(self, message: str):
        super().__init__(
            message,
            "GlinerWorkerTimeoutError",
            status.HTTP_504_GATEWAY_TIMEOUT,
            remediation=(
                "Raise COGNEE_GLINER_REQUEST_TIMEOUT_SECS, or lower "
                "COGNEE_GLINER_INPUTS_PER_REQUEST or the chunk size so one request is smaller."
            ),
        )


class GlinerWorkerRuntimeError(GlinerWorkerError):
    """Inference failed on the worker (HTTP 500, gRPC INTERNAL, RabbitMQ
    ``internal``), or its reply was malformed. Deterministic, so not retried;
    fails that document only."""

    def __init__(self, message: str):
        super().__init__(message, "GlinerWorkerRuntimeError")


class GlinerWorkerModelMismatchError(GlinerWorkerError):
    """The worker serves a different model than COGNEE_GLINER_EXPECTED_MODEL, or
    than the ``model_name`` the caller passed."""

    fatal = True

    def __init__(self, expected: str, served: str):
        self.expected = expected
        self.served = served
        super().__init__(
            f"The GLiNER worker serves {served!r}, but {expected!r} is required.",
            "GlinerWorkerModelMismatchError",
            remediation=(
                "Point COGNEE_GLINER_ENDPOINT at a worker serving the expected model, "
                "or change COGNEE_GLINER_EXPECTED_MODEL / the model_name argument."
            ),
        )


class GlinerWorkerIncompatibleError(GlinerWorkerError):
    """The endpoint is not a worker this client can use: it predates a contract
    feature (windowing), or its readiness probe does not answer like a worker."""

    fatal = True

    def __init__(self, message: str, remediation: str | None = None):
        super().__init__(
            message,
            "GlinerWorkerIncompatibleError",
            remediation=remediation
            or (
                "Upgrade gliner_worker to a version that supports the window_words option "
                "(main at or after 92aba3c, topoteretes/gliner_worker#4)."
            ),
        )


class GlinerRemoteConfigError(CogneeConfigurationError):
    """Remote GLiNER settings are missing, malformed, beyond the worker's limits,
    or need an extra that is not installed."""

    retryable = False
    fatal = True

    def __init__(self, message: str, remediation: str | None = None):
        super().__init__(message, "GlinerRemoteConfigError", log=False, remediation=remediation)
