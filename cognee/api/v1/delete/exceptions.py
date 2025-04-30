from cognee.exceptions import CogneeApiError
from fastapi import status


class DocumentNotFoundError(CogneeApiError):
    """Raised when a document cannot be found in the database."""

    def __init__(
        self,
        message: str = "Document not found in database.",
        name: str = "DocumentNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DatasetNotFoundError(CogneeApiError):
    """Raised when a dataset cannot be found."""

    def __init__(
        self,
        message: str = "Dataset not found.",
        name: str = "DatasetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DocumentSubgraphNotFoundError(CogneeApiError):
    """Raised when a document's subgraph cannot be found in the graph database."""

    def __init__(
        self,
        message: str = "Document subgraph not found in graph database.",
        name: str = "DocumentSubgraphNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
