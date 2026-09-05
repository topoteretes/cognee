"""Typed errors raised by the remote CloudClient.

Every error subclasses ``RuntimeError`` so existing callers that catch
``RuntimeError`` keep working. The hierarchy exists so integration clients
(circuit breakers, "UNREACHABLE" sentinels, retry policies) can distinguish
the cases that must be handled differently:

- ``CogneeTransportError`` — the server was never reached (connect refused,
  DNS failure, timeout). The only case that should count as "unreachable".
- ``CogneeAuthError`` — reached, but the credentials were rejected (401/403).
  Retrying without new credentials is pointless.
- ``CogneeClientRequestError`` — reached, request was invalid (other 4xx).
- ``CogneeServerError`` — reached, server failed (5xx). Retryable.
"""

from typing import Any, Optional


class CogneeAPIError(RuntimeError):
    """Base class for all remote-operation failures."""

    def __init__(self, message: str, *, operation: Optional[str] = None):
        super().__init__(message)
        self.operation = operation


class CogneeTransportError(CogneeAPIError):
    """The Cognee instance could not be reached at the transport level.

    Wraps the underlying aiohttp/asyncio error; the server is not known to
    have received the request. This is deliberately narrower than "any
    error": an HTTP 4xx/5xx means the server *is* reachable and raises one
    of the HTTP error classes below instead.
    """

    def __init__(self, message: str, *, operation: Optional[str] = None, cause: BaseException):
        super().__init__(message, operation=operation)
        self.__cause__ = cause


class CogneeHTTPError(CogneeAPIError):
    """The server responded with an HTTP error status."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: Any = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, operation=operation)
        self.status = status
        self.body = body


class CogneeAuthError(CogneeHTTPError):
    """401/403 — the request reached the server but authentication failed."""


class CogneeClientRequestError(CogneeHTTPError):
    """Any other 4xx — the request was invalid or the resource is missing."""


class CogneeServerError(CogneeHTTPError):
    """5xx — the server accepted the request but failed to process it."""


def http_error_for_status(
    status: int,
    body: Any,
    *,
    operation: str,
) -> CogneeHTTPError:
    """Build the matching typed error for an HTTP error response."""
    message = f"Remote {operation} failed ({status}): {body}"
    if status in (401, 403):
        return CogneeAuthError(message, status=status, body=body, operation=operation)
    if status < 500:
        return CogneeClientRequestError(message, status=status, body=body, operation=operation)
    return CogneeServerError(message, status=status, body=body, operation=operation)
