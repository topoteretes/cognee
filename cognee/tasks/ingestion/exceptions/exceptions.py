from fastapi import status
from cognee.exceptions import CogneeSystemError


class CloudFileSystemNotFoundError(CogneeSystemError):
    def __init__(
        self,
        name: str = "CloudFileSystemNotFoundError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        message = "Could not find CloudFileSystem."
        super().__init__(message, name, status_code)
