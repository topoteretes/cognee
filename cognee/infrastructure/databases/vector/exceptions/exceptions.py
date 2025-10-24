from fastapi import status
from cognee.exceptions import CogneeValidationError


class CollectionNotFoundError(CogneeValidationError):
    """
    Represents an error that occurs when a requested collection cannot be found.

    This class extends the CogneeValidationError to handle specific cases where a requested
    collection is unavailable. It can be initialized with a custom message and allows for
    logging options including log level and whether to log the error.
    """

    def __init__(
        self,
        message,
        name: str = "CollectionNotFoundError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log=True,
        log_level="DEBUG",
    ):
        super().__init__(message, name, status_code, log, log_level)
