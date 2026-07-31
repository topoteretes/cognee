from typing import Any

from cognee.exceptions.error_codes import ErrorCode
from cognee.exceptions.exceptions import CogneeApiError, CogneeValidationError

# Legacy exceptions that cannot be re-parented in Pillar B without broader refactors.
SPECIAL_CASE_OVERRIDES: dict[type[CogneeApiError], dict[str, Any]] = {}


def _register_special_case_overrides() -> None:
    from cognee.infrastructure.databases.exceptions.exceptions import DatabaseNotCreatedError

    SPECIAL_CASE_OVERRIDES[DatabaseNotCreatedError] = {
        "code": ErrorCode.DATA_NOT_READY,
        "retryable": False,
        "remediation": "Initialize cognee with await cognee.setup() or run cognify first.",
    }


_register_special_case_overrides()


def coerce_to_cognee_error(exc: Exception) -> CogneeApiError:
    """Wrap unknown exceptions for MCP/CLI surfaces."""
    if isinstance(exc, CogneeApiError):
        return exc
    if isinstance(exc, ValueError):
        return CogneeValidationError(str(exc))
    return CogneeApiError(
        message=str(exc),
        name=type(exc).__name__,
        details={"type": type(exc).__name__},
    )


def serialize_cognee_error(exc: CogneeApiError) -> dict[str, Any]:
    """Build a stable JSON envelope for SDK, HTTP, CLI, and MCP surfaces."""
    envelope = exc.to_envelope()
    payload = envelope.model_dump(exclude_none=True)
    payload["code"] = envelope.code.value
    return payload


def http_error_content(exc: CogneeApiError) -> dict[str, Any]:
    """HTTP response body with structured error plus legacy detail string."""
    return {
        "error": serialize_cognee_error(exc),
        "detail": exc.message,
    }


def mcp_error_payload(exc: CogneeApiError) -> dict[str, Any]:
    """Structured MCP error payload without MCP runtime dependencies."""
    return {"error": serialize_cognee_error(exc)}
