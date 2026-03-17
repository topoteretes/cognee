from cognee.exceptions import CogneeConfigurationError, CogneeValidationError
from fastapi import status


class IngestionError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Failed to load data.",
        name: str = "IngestionError",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class UsageLoggerError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = "Usage logging configuration is invalid.",
        name: str = "UsageLoggerError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)
