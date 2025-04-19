from fastapi import status
from cognee.exceptions import CriticalError


class CollectionNotFoundError(CriticalError):
    def __init__(
        self,
        message,
        name: str = "DatabaseNotCreatedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
