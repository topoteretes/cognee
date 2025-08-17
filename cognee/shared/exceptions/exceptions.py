from cognee.exceptions import CogneeValidationError
from fastapi import status


class IngestionError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Failed to load data.",
        name: str = "IngestionError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
