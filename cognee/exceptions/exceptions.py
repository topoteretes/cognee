from fastapi import status
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class CogneeApiError(Exception):
    """Base exception class"""

    def __init__(
        self,
        message: str = "Service is unavailable.",
        name: str = "Cognee",
        status_code=status.HTTP_418_IM_A_TEAPOT,
        log=True,
        log_level="ERROR",
    ):
        self.message = message
        self.name = name
        self.status_code = status_code

        # Automatically log the exception details
        if log and (log_level == "ERROR"):
            logger.error(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and (log_level == "WARNING"):
            logger.warning(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and (log_level == "INFO"):
            logger.info(f"{self.name}: {self.message} (Status code: {self.status_code})")
        elif log and (log_level == "DEBUG"):
            logger.debug(f"{self.name}: {self.message} (Status code: {self.status_code})")

        super().__init__(self.message, self.name)

    def __str__(self):
        return f"{self.name}: {self.message} (Status code: {self.status_code})"


class CogneeSystemError(CogneeApiError):
    """System error"""

    def __init__(
        self,
        message: str = "A system error occurred.",
        name: str = "CogneeSystemError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log=True,
        log_level="ERROR",
    ):
        super().__init__(message, name, status_code, log, log_level)


class CogneeValidationError(CogneeApiError):
    """Validation error"""

    def __init__(
        self,
        message: str = "A validation error occurred.",
        name: str = "CogneeValidationError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        log=True,
        log_level="ERROR",
    ):
        super().__init__(message, name, status_code, log, log_level)


class CogneeConfigurationError(CogneeApiError):
    """SystemConfigError"""

    def __init__(
        self,
        message: str = "A system configuration error occurred.",
        name: str = "CogneeConfigurationError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        log=True,
        log_level="ERROR",
    ):
        super().__init__(message, name, status_code, log, log_level)


class CogneeTransientError(CogneeApiError):
    """TransientError"""

    def __init__(
        self,
        message: str = "A transient error occurred.",
        name: str = "CogneeTransientError",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        log=True,
        log_level="ERROR",
    ):
        super().__init__(message, name, status_code, log, log_level)
