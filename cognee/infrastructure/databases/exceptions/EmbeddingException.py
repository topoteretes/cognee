from cognee.exceptions import CogneeApiError
from fastapi import status


class EmbeddingException(CogneeApiError):
    """Custom exception for handling embedding-related errors."""

    def __init__(
        self,
        message: str = "Embedding Exception.",
        name: str = "EmbeddingException",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
