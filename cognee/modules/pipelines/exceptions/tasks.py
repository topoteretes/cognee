from fastapi import status
from cognee.exceptions import CogneeValidationError


class WrongTaskTypeError(CogneeValidationError):
    """
    Raised when the tasks argument is not a list of Task class instances.
    """

    def __init__(
        self,
        message: str = "tasks argument must be a list, containing Task class instances.",
        name: str = "WrongTaskTypeError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
