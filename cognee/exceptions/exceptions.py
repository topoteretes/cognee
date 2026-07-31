from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from fastapi import status

from cognee.exceptions.error_codes import ErrorCode
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.exceptions.envelope import CogneeErrorEnvelope

logger = get_logger()

SEMANTIC_ERROR_BASES: tuple[type["CogneeApiError"], ...] = ()


class CogneeApiError(Exception):
    """Base exception class with structured fields for agent consumers."""

    default_code: ErrorCode = ErrorCode.SYSTEM
    default_remediation: str = "Check the error message and cognee documentation."
    default_retryable: bool = False

    def __init__(
        self,
        message: str = "Service is unavailable.",
        name: str = "Cognee",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
        self.code = code
        self.remediation = remediation
        self.retryable = retryable
        self.details = details
        self.docs_url = docs_url

        if log and log_level == "ERROR":
            logger.error(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and log_level == "WARNING":
            logger.warning(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and log_level == "INFO":
            logger.info(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and log_level == "DEBUG":
            logger.debug(f"{self.name}: {self.message} (Status code: {self.status_code})")

        super().__init__(self.message, self.name)

    def __str__(self) -> str:
        return f"{self.name}: {self.message} (Status code: {self.status_code})"

    def _resolved_code(self) -> ErrorCode:
        if self.code is not None:
            if isinstance(self.code, ErrorCode):
                return self.code
            return ErrorCode(str(self.code))
        return self.default_code

    def _resolved_remediation(self) -> str:
        if self.remediation is not None:
            return self.remediation
        return self.default_remediation

    def _resolved_retryable(self) -> bool:
        if self.retryable is not None:
            return self.retryable
        return self.default_retryable

    def to_envelope(self) -> "CogneeErrorEnvelope":
        from cognee.exceptions.envelope import CogneeErrorEnvelope
        from cognee.exceptions.serialize import SPECIAL_CASE_OVERRIDES

        code = self._resolved_code()
        remediation = self._resolved_remediation()
        retryable = self._resolved_retryable()

        for exc_type, overrides in SPECIAL_CASE_OVERRIDES.items():
            if isinstance(self, exc_type):
                code = overrides.get("code", code)
                remediation = overrides.get("remediation", remediation)
                retryable = overrides.get("retryable", retryable)
                break

        return CogneeErrorEnvelope(
            code=code,
            message=self.message,
            name=self.name,
            remediation=remediation,
            retryable=retryable,
            status_code=self.status_code,
            docs_url=self.docs_url,
            details=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        from cognee.exceptions.serialize import serialize_cognee_error

        return serialize_cognee_error(self)


class CogneeSystemError(CogneeApiError):
    """System error"""

    default_code = ErrorCode.SYSTEM
    default_remediation = "Check server logs or retry after verifying your setup."

    def __init__(
        self,
        message: str = "A system error occurred.",
        name: str = "CogneeSystemError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


class CogneeValidationError(CogneeApiError):
    """Validation error"""

    default_code = ErrorCode.INVALID_INPUT
    default_remediation = "Check your input parameters and try again."

    def __init__(
        self,
        message: str = "A validation error occurred.",
        name: str = "CogneeValidationError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


class CogneeDataNotReadyError(CogneeValidationError):
    """Raised when data is missing or not processed for recall/search."""

    default_code = ErrorCode.DATA_NOT_READY
    default_remediation = (
        "Add data with cognee.add(), then run cognee.cognify() before searching."
    )

    def __init__(
        self,
        message: str = "Data is not ready for recall or search.",
        name: str = "CogneeDataNotReadyError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


class CogneePermissionError(CogneeValidationError):
    """Raised when ACL denies access."""

    default_code = ErrorCode.PERMISSION_DENIED
    default_remediation = (
        "Grant read access on the dataset or disable ENABLE_BACKEND_ACCESS_CONTROL."
    )

    def __init__(
        self,
        message: str = "Permission denied.",
        name: str = "CogneePermissionError",
        status_code: int = status.HTTP_403_FORBIDDEN,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


class CogneeConfigurationError(CogneeApiError):
    """Configuration error"""

    default_code = ErrorCode.MISSING_CONFIG
    default_remediation = "Review your environment variables and cognee configuration."

    def __init__(
        self,
        message: str = "A system configuration error occurred.",
        name: str = "CogneeConfigurationError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


class CogneeTransientError(CogneeApiError):
    """Transient error"""

    default_code = ErrorCode.TRANSIENT
    default_remediation = "Retry the request after a short backoff."
    default_retryable = True

    def __init__(
        self,
        message: str = "A transient error occurred.",
        name: str = "CogneeTransientError",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        log: bool = True,
        log_level: str = "ERROR",
        code: ErrorCode | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(
            message,
            name,
            status_code,
            log,
            log_level,
            code=code,
            remediation=remediation,
            retryable=retryable,
            details=details,
            docs_url=docs_url,
        )


SEMANTIC_ERROR_BASES = (
    CogneeDataNotReadyError,
    CogneePermissionError,
    CogneeConfigurationError,
    CogneeTransientError,
    CogneeSystemError,
    CogneeValidationError,
)
