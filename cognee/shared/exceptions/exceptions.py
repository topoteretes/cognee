from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeValidationError


class IngestionError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Failed to load data.",
        name: str = "IngestionError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ) -> None:
        super().__init__(message, name, status_code)


class UsageLoggerError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = "Usage logging configuration is invalid.",
        name: str = "UsageLoggerError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(message, name, status_code)
