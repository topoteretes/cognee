from cognee.exceptions import CogneeValidationError
from fastapi import status


class IngestionError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Type of data sent to classify not supported.",
        name: str = "IngestionError",
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ):
        super().__init__(message, name, status_code)


class DataContentConflictError(CogneeValidationError):
    """A pinned data_id already holds different content in the dataset.

    Raised by the incremental add pre-check instead of silently keeping the
    stale record: the caller must choose between updating the existing
    document, ingesting under a different identity, or forcing re-ingestion.
    """

    def __init__(
        self,
        message: str = "Data id already exists with different content.",
        name: str = "DataContentConflictError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)
