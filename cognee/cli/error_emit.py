"""Structured error emission for CLI human and machine modes."""

from __future__ import annotations

import json
import sys
from typing import Any

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.exceptions import CogneeApiError, serialize_cognee_error


def _exit_code_for_error(exc: CogneeApiError) -> int:
    if exc.status_code < 256:
        return exc.status_code
    return 1


def emit_cognee_error_json(exc: CogneeApiError) -> int:
    """Emit machine-readable JSON on stdout only."""
    payload = {"error": serialize_cognee_error(exc)}
    print(json.dumps(payload))
    return _exit_code_for_error(exc)


def emit_cognee_error(
    exc: CogneeApiError,
    json_mode: bool = False,
    docs_url: str | None = None,
) -> int:
    """Emit structured error for CLI consumers."""
    if json_mode:
        return emit_cognee_error_json(exc)

    envelope = serialize_cognee_error(exc)
    fmt.error(f"{envelope['code']}: {envelope['message']}")
    fmt.note(envelope["remediation"])
    fmt.note(f"Please refer to our docs at '{docs_url or DEFAULT_DOCS_URL}' for further assistance.")
    return _exit_code_for_error(exc)


def parse_http_error_payload(detail: Any, status_code: int) -> CogneeApiError:
    """Reconstruct CogneeApiError from structured HTTP error response."""
    if isinstance(detail, dict) and "error" in detail:
        err = detail["error"]
        return CogneeApiError(
            message=err.get("message", "API error"),
            name=err.get("name", "CogneeApiError"),
            status_code=status_code,
            code=err.get("code"),
            remediation=err.get("remediation"),
            retryable=err.get("retryable"),
            details=err.get("details"),
            docs_url=err.get("docs_url"),
            log=False,
        )
    message = detail if isinstance(detail, str) else str(detail)
    return CogneeApiError(message=message, status_code=status_code, log=False)
