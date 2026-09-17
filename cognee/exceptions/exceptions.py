from fastapi import status

from cognee.shared.logging_utils import get_logger

from .remediation import REMEDIATION_MARKER

logger = get_logger()


class CogneeApiError(Exception):
    """Root of the cognee exception hierarchy.

    Every error cognee raises on purpose derives from this class and carries:

    * ``message`` -- what went wrong, for humans and agents;
    * ``name`` -- the error class name, appended to REST ``detail`` as ``[Name]``;
    * ``status_code`` -- the HTTP status the REST layer answers with;
    * ``remediation`` -- optional, what to change to make the error go away (an env var
      to set, an extra to install, a call to make first). ``str(exc)`` appends it as
      ``Fix: ...`` so it survives every transport (CLI, REST ``remediation`` key, MCP
      tool text). Errors without one fall back to the substring table in
      ``cognee.exceptions.remediation.find_remediation``.

    Subclasses must call ``super().__init__`` (enforced by
    ``cognee/tests/unit/exceptions/test_cognee_error_contract.py``).
    """

    def __init__(
        self,
        message: str = "Service is unavailable.",
        name: str = "Cognee",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log=True,
        log_level="ERROR",
        remediation: str | None = None,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
        self.remediation = remediation

        # Automatically log the exception details
        if log and (log_level == "ERROR"):
            logger.error("%s raised (Status code: %s)", self.name, self.status_code)
        elif log and (log_level == "WARNING"):
            logger.warning("%s raised (Status code: %s)", self.name, self.status_code)
        elif log and (log_level == "INFO"):
            logger.info("%s raised (Status code: %s)", self.name, self.status_code)
        elif log and (log_level == "DEBUG"):
            logger.debug("%s raised (Status code: %s)", self.name, self.status_code)

        super().__init__(self.message, self.name)

    def __str__(self):
        text = f"{self.name}: {self.message} (Status code: {self.status_code})"
        if self.remediation:
            text = f"{text}{REMEDIATION_MARKER}{self.remediation}"
        return text


class CogneeSystemError(CogneeApiError):
    """System error"""

    def __init__(
        self,
        message: str = "A system error occurred.",
        name: str = "CogneeSystemError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log=True,
        log_level="ERROR",
        remediation: str | None = None,
    ):
        super().__init__(message, name, status_code, log, log_level, remediation=remediation)


class CogneeValidationError(CogneeApiError):
    """Validation error"""

    def __init__(
        self,
        message: str = "A validation error occurred.",
        name: str = "CogneeValidationError",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        log=True,
        log_level="ERROR",
        remediation: str | None = None,
    ):
        super().__init__(message, name, status_code, log, log_level, remediation=remediation)


class CogneeConfigurationError(CogneeApiError):
    """SystemConfigError"""

    def __init__(
        self,
        message: str = "A system configuration error occurred.",
        name: str = "CogneeConfigurationError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log=True,
        log_level="ERROR",
        remediation: str | None = None,
    ):
        super().__init__(message, name, status_code, log, log_level, remediation=remediation)


class CogneeTransientError(CogneeApiError):
    """TransientError"""

    def __init__(
        self,
        message: str = "A transient error occurred.",
        name: str = "CogneeTransientError",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        log=True,
        log_level="ERROR",
        remediation: str | None = None,
    ):
        super().__init__(message, name, status_code, log, log_level, remediation=remediation)
