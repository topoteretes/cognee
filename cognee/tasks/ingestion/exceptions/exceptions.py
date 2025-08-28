from cognee.exceptions import CogneeSystemError
from fastapi import status


class S3FileSystemNotFoundError(CogneeSystemError):
    def __init__(
        self,
        name: str = "S3FileSystemNotFoundError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        message = "Could not find S3FileSystem."
        super().__init__(message, name, status_code)
