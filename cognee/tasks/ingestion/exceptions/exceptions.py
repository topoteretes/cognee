from cognee.exceptions import CogneeSystemError, CogneeValidationError, CogneeConfigurationError
from fastapi import status


class S3FileSystemNotFoundError(CogneeSystemError):
    def __init__(
        self,
        name: str = "S3FileSystemNotFoundError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        message = "Could not find S3FileSystem."
        super().__init__(message, name, status_code)


class InvalidDLTArgumentError(CogneeValidationError):
    def __init__(
        self,
        name: str = "InvalidDLTArgumentError",
        message: str = "Invalid argument for dlt ingestion.",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class UnsupportedDBProviderError(CogneeConfigurationError):
    def __init__(
        self,
        name: str = "UnsupportedDBProviderError",
        message: str = "Unsupported database provider.",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class DLTIngestionError(CogneeSystemError):
    def __init__(
        self,
        name: str = "DLTIngestionError",
        message: str = "Error in the execution of a DLT pipeline, and the extraction of its schema",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)
