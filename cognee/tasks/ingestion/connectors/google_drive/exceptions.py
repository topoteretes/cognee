from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError


class GoogleDriveConfigError(CogneeConfigurationError):
    def __init__(
        self,
        name: str = "GoogleDriveConfigError",
        message: str = "Invalid or missing Google Drive connector configuration.",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class GoogleDriveAPIError(CogneeSystemError):
    def __init__(
        self,
        name: str = "GoogleDriveAPIError",
        message: str = "Error communicating with the Google Drive API.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)
