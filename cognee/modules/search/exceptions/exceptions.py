from cognee.exceptions import (
    CogneeValidationError,
)
from fastapi import status


class UnsupportedSearchTypeError(CogneeValidationError):
    def __init__(
        self,
        search_type: str,
        name: str = "UnsupportedSearchTypeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"Unsupported search type: {search_type}"
        super().__init__(message, name, status_code)
