from cognee.exceptions import CogneeApiError
from fastapi import status


class EmbeddingException(CogneeApiError):
    """
    Custom exception for handling embedding-related errors.

    This exception class is designed to indicate issues specifically related to embeddings
    within the application. It extends the base exception class CogneeApiError and allows
    for customization of the error message, name, and status code.
    """

    def __init__(
        self,
        message: str = "Embedding Exception.",
        name: str = "EmbeddingException",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
