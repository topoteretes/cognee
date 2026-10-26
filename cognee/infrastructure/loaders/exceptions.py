"""Errors the loader layer raises for files it cannot read."""

from fastapi import status

from cognee.exceptions import CogneeValidationError


class UnreadableFileContentError(CogneeValidationError):
    """A loader claimed a file it then could not decode.

    Raised instead of letting a bare ``UnicodeDecodeError`` escape: that is a
    Python builtin with no status code, so it reaches an API caller as a 500,
    which reads as "server fault, retry" rather than "this file will never
    work". 415 says the media type is the problem.
    """

    def __init__(
        self,
        message: str,
        name: str = "UnreadableFileContentError",
        status_code: int = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ):
        super().__init__(message, name, status_code)
