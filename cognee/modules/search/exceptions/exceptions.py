from fastapi import status

from cognee.exceptions import (
    CogneeValidationError,
)


class UnsupportedSearchTypeError(CogneeValidationError):
    def __init__(
        self,
        search_type: str,
        name: str = "UnsupportedSearchTypeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        message: str | None = None,
        log: bool = True,
        remediation: str | None = None,
    ):
        if message is None:
            message = f"Unsupported search type: {search_type}"
        super().__init__(message, name, status_code, log=log, remediation=remediation)


class SearchTypeDisabledError(UnsupportedSearchTypeError):
    """A supported search type that this deployment has switched off.

    Subclasses UnsupportedSearchTypeError so callers that already catch that error
    keep working.
    """

    def __init__(
        self,
        search_type: str,
        message: str,
        name: str = "SearchTypeDisabledError",
        status_code: int = status.HTTP_403_FORBIDDEN,
        log: bool = True,
        remediation: str | None = None,
    ):
        super().__init__(
            search_type, name, status_code, message=message, log=log, remediation=remediation
        )
