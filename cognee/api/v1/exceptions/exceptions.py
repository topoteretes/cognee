from cognee.exceptions import (
    CogneeConfigurationError,
)
from fastapi import status


class InvalidConfigAttributeError(CogneeConfigurationError):
    def __init__(
        self,
        attribute: str,
        name: str = "InvalidConfigAttributeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"'{attribute}' is not a valid attribute of the configuration."
        super().__init__(message, name, status_code)
