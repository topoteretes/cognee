"""Typed failures of a remote embedding worker.

Each class decides two things: whether the engine retries it, and how far it
reaches. ``fatal`` errors (an outage that outlived its retries, bad broker
credentials, the wrong model or vector size, bad settings) are latched by the
engine and fail every later call at once, because every call would meet the same
problem; an outage is released again by the first readiness check that finds a
worker. The rest (a rejected, failed or timed-out request) fail only the call
that caused them.

All are :class:`EmbeddingException` subclasses, so the API answers them like any
other embedding failure. None of them log on construction: retried attempts
would flood the log, so the engine logs what it decides instead.
"""

from __future__ import annotations

from fastapi import status

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.exceptions import EmbeddingException


class EmbeddingWorkerError(EmbeddingException):
    """Base class for failures reported by, or about, a remote embedding worker."""

    retryable = False
    fatal = False

    def __init__(
        self,
        message: str,
        name: str = "EmbeddingWorkerError",
        status_code=status.HTTP_502_BAD_GATEWAY,
        remediation: str | None = None,
    ):
        # EmbeddingException's own __init__ always logs; these must not.
        CogneeConfigurationError.__init__(
            self, message, name, status_code, log=False, remediation=remediation
        )


class EmbeddingWorkerUnavailableError(EmbeddingWorkerError):
    """The broker is unreachable, does not accept the request (no queue, nacked,
    under an alarm), no worker consumes the queue, or the worker answered
    ``unavailable``.

    Retried. Once the retries are spent it is fatal until a readiness check finds
    a worker again.
    """

    retryable = True
    fatal = True

    def __init__(self, message: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            message,
            "EmbeddingWorkerUnavailableError",
            status.HTTP_503_SERVICE_UNAVAILABLE,
            remediation=(
                "Check that an embedding worker consumes COGNEE_EMBEDDING_WORKER_AMQP_QUEUE "
                "on the broker at COGNEE_EMBEDDING_WORKER_ENDPOINT."
            ),
        )


class EmbeddingWorkerUnauthorizedError(EmbeddingWorkerError):
    """The broker refused the credentials in COGNEE_EMBEDDING_WORKER_ENDPOINT."""

    fatal = True

    def __init__(self, message: str):
        super().__init__(
            message,
            "EmbeddingWorkerUnauthorizedError",
            remediation="Fix the user and password in COGNEE_EMBEDDING_WORKER_ENDPOINT.",
        )


class EmbeddingWorkerRejectedError(EmbeddingWorkerError):
    """The worker rejected the request as invalid (``invalid_request``: too many
    inputs, an input over its character limit, ...). Fails that call only."""

    def __init__(self, message: str):
        super().__init__(
            message, "EmbeddingWorkerRejectedError", status.HTTP_422_UNPROCESSABLE_CONTENT
        )


class EmbeddingWorkerInputTooLongError(EmbeddingWorkerRejectedError):
    """Inputs have more tokens (or characters) than the worker accepts (``input_too_long``).

    ``indices`` lists every offending input of the request, when the worker says
    (its ``inputs``: ``[{"index", "chars", "tokens"}]``); None from an older
    worker. The engine splits exactly those inputs and sends the request again,
    pooling each one's parts; it only escapes when an input cannot be split
    further."""

    def __init__(
        self,
        message: str,
        *,
        indices: list[int] | None = None,
        max_input_tokens: int | None = None,
        max_text_chars: int | None = None,
    ):
        self.indices = indices
        self.max_input_tokens = max_input_tokens
        self.max_text_chars = max_text_chars
        super().__init__(message)


class EmbeddingWorkerTimeoutError(EmbeddingWorkerError):
    """The broker accepted a request but no reply came within
    COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS.

    Not retried: the worker may still be running it, and resending a request that
    is slow on its own would only multiply that work. Fails that call only; the
    engine first re-checks readiness, so a worker that went away fails as
    unavailable instead.
    """

    def __init__(self, message: str):
        super().__init__(
            message,
            "EmbeddingWorkerTimeoutError",
            status.HTTP_504_GATEWAY_TIMEOUT,
            remediation=(
                "Raise COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS, or lower "
                "EMBEDDING_BATCH_SIZE so one request is smaller."
            ),
        )


class EmbeddingWorkerRuntimeError(EmbeddingWorkerError):
    """Inference failed on the worker (``internal``, or an unknown error code), or
    its reply was malformed. Deterministic, so not retried; fails that call only."""

    def __init__(self, message: str):
        super().__init__(message, "EmbeddingWorkerRuntimeError")


class EmbeddingWorkerModelMismatchError(EmbeddingWorkerError):
    """The worker serves a different model than COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL
    (from its model info, or from an embedding reply)."""

    fatal = True

    def __init__(self, expected: str, served: str):
        self.expected = expected
        self.served = served
        super().__init__(
            f"The embedding worker serves {served!r}, but {expected!r} is required "
            "(COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL). Vectors from different models are "
            "not comparable.",
            "EmbeddingWorkerModelMismatchError",
            remediation=(
                "Point COGNEE_EMBEDDING_WORKER_ENDPOINT / _AMQP_QUEUE at a worker serving the "
                "expected model, or change COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL."
            ),
        )


class EmbeddingWorkerDimensionsMismatchError(EmbeddingWorkerError):
    """The worker's vectors do not have EMBEDDING_DIMENSIONS components (or, with it
    unset, the size the engine first learned from the worker), so they would not
    fit the vector index."""

    fatal = True

    def __init__(
        self, expected: int, served: int, model: str, source: str = "EMBEDDING_DIMENSIONS"
    ):
        self.expected = expected
        self.served = served
        super().__init__(
            f"The embedding worker's model {model!r} returns {served}-dimensional vectors, "
            f"but {source} is {expected}; they would not fit the vector index.",
            "EmbeddingWorkerDimensionsMismatchError",
            remediation=(
                f"Set EMBEDDING_DIMENSIONS={served} (and rebuild existing vector collections), "
                "or point the client at a worker serving a model of the configured size."
            ),
        )


class EmbeddingWorkerConfigError(CogneeConfigurationError):
    """Embedding worker settings are missing or malformed, name a transport that is
    not implemented, or need an extra that is not installed."""

    retryable = False
    fatal = True

    def __init__(self, message: str, remediation: str | None = None):
        super().__init__(message, "EmbeddingWorkerConfigError", log=False, remediation=remediation)
