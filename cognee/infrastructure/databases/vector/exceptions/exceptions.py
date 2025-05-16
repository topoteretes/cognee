from fastapi import status
from cognee.exceptions import CriticalError


class CollectionNotFoundError(CriticalError):
    def __init__(
        self,
        message,
        name: str = "CollectionNotFoundError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        log=True,
        log_level="DEBUG",
    ):
        super().__init__(message, name, status_code, log, log_level)
