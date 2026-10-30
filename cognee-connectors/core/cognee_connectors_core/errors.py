"""Errors raised by the connector contract.

Host-agnostic on purpose: a connector package never imports FastAPI, so the
one place the contract has to say "this request is bad" raises a plain
exception carrying the HTTP status a host *would* answer with, and the host
shim (cognee's ``POST /{provider}/events`` route, cloud-backend's) maps it to
its own response type.
"""


class ConnectorError(Exception):
    """Base class for every error this package raises on purpose."""


class WebhookVerificationError(ConnectorError):
    """An inbound webhook delivery failed verification and must be refused.

    ``status_code`` is the HTTP status a host should answer with — ``401`` for
    a bad or missing signature, ``413`` for a body over the cap, ``400`` for a
    request the verifier could not read — and ``detail`` the response text.
    The detail is deliberately uniform across every rejection reason (see
    ``signatures._reject`` for why), so a host can return it verbatim.
    """

    def __init__(self, detail: str = "Invalid signature", *, status_code: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class RateLimitedError(ConnectorError):
    """The provider rate-limited a request.

    Raised by provider clients so a sync can back off rather than fail.
    ``retry_after`` is the provider's own hint in seconds when it sent one
    (``Retry-After``), otherwise ``None``.
    """

    def __init__(
        self,
        message: str = "rate limited by the provider",
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


__all__ = ["ConnectorError", "RateLimitedError", "WebhookVerificationError"]
