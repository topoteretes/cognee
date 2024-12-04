from cognee.exceptions import CogneeApiError
from fastapi import status


class GroupNotFoundError(CogneeApiError):
    """User group not found"""

    def __init__(
        self,
        message: str = "User group not found.",
        name: str = "GroupNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class UserNotFoundError(CogneeApiError):
    """User not found"""

    def __init__(
        self,
        message: str = "No user found in the system. Please create a user.",
        name: str = "UserNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class PermissionDeniedError(CogneeApiError):
    def __init__(
            self,
            message: str = "User does not have permission on documents.",
            name: str = "PermissionDeniedError",
            status_code=status.HTTP_403_FORBIDDEN,
    ):
        super().__init__(message, name, status_code)
