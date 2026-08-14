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


class LabelCountMismatchError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Label count does not match data item count.",
        name: str = "LabelCountMismatchError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class InvalidLabelsError(CogneeValidationError):
    def __init__(
        self,
        message: str = 'labels must be a JSON array of strings, e.g. ["finance", ""].',
        name: str = "InvalidLabelsError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class InvalidExternalMetadataError(CogneeValidationError):
    def __init__(
        self,
        message: str = (
            'external_metadata must be a JSON array of objects, e.g. [{"source": "crm"}, null].'
        ),
        name: str = "InvalidExternalMetadataError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class ExternalMetadataCountMismatchError(CogneeValidationError):
    def __init__(
        self,
        message: str = "external_metadata count does not match data item count.",
        name: str = "ExternalMetadataCountMismatchError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
