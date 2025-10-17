from fastapi import status


class FileContentHashingError(Exception):
    """Raised when the file content cannot be hashed."""

    def __init__(
        self,
        message: str = "Failed to hash content of the file.",
        name: str = "FileContentHashingError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)


class UnsupportedPathSchemeError(Exception):
    """Raised when a non-filesystem path scheme (like http://, https://) is passed to a function expecting filesystem paths."""

    def __init__(
        self,
        message: str = "This function only supports filesystem paths (file:// or local paths), not HTTP/HTTPS URLs.",
        name: str = "UnsupportedPathSchemeError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
