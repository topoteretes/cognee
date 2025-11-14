from fastapi import status


class FileContentHashingError(Exception):
    """Raised when the file content cannot be hashed."""

    def __init__(
        self,
        message: str = "Failed to hash content of the file.",
        name: str = "FileContentHashingError",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)
