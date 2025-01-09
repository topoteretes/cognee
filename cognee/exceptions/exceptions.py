from fastapi import status
import logging

logger = logging.getLogger(__name__)


class CogneeApiError(Exception):
    """Base exception class"""

    def __init__(
        self,
        message: str = "Service is unavailable.",
        name: str = "Cognee",
        status_code=status.HTTP_418_IM_A_TEAPOT,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code

        # Automatically log the exception details
        logger.error(f"{self.name}: {self.message} (Status code: {self.status_code})")

        super().__init__(self.message, self.name)


class ServiceError(CogneeApiError):
    """Failures in external services or APIs, like a database or a third-party service"""

    def __init__(
        self,
        message: str = "Service is unavailable.",
        name: str = "ServiceError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)


class InvalidValueError(CogneeApiError):
    def __init__(
        self,
        message: str = "Invalid Value.",
        name: str = "InvalidValueError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)


class InvalidAttributeError(CogneeApiError):
    def __init__(
        self,
        message: str = "Invalid attribute.",
        name: str = "InvalidAttributeError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
