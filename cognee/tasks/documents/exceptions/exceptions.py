from cognee.exceptions import (
    CogneeValidationError,
    CogneeConfigurationError,
)
from fastapi import status


class WrongDataDocumentInputError(CogneeValidationError):
    """Raised when a wrong data document is provided."""
    def __init__(
        self,
        field: str,
        name: str = "WrongDataDocumentInputError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        message = f"Missing of invalid parameter: '{field}'."
        super().__init__(message, name, status_code)