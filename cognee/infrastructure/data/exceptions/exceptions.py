from cognee.exceptions import (
    CogneeValidationError,
)
from fastapi import status


class KeywordExtractionError(CogneeValidationError):
    """
    Raised when a provided value is syntactically valid but semantically unacceptable
    for the given operation.

    Example:
        - Passing an empty string to a keyword extraction function.
    """

    def __init__(
        self,
        message: str = "Extract_keywords cannot extract keywords from empty text.",
        name: str = "KeywordExtractionError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
