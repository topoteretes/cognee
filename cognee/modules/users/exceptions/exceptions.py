from fastapi import status

from cognee.exceptions import CogneeValidationError


class RoleNotFoundError(CogneeValidationError):
    """User group not found"""

    def __init__(
        self,
        message: str = "User role not found.",
        name: str = "RoleNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class TenantNotFoundError(CogneeValidationError):
    """User group not found"""

    def __init__(
        self,
        message: str = "Tenant not found.",
        name: str = "TenantNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class UserNotFoundError(CogneeValidationError):
    """User not found"""

    def __init__(
        self,
        message: str = "No user found in the system. Please create a user.",
        name: str = "UserNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class PermissionDeniedError(CogneeValidationError):
    def __init__(
        self,
        message: str = "User does not have permission on documents.",
        name: str = "PermissionDeniedError",
        status_code=status.HTTP_403_FORBIDDEN,
        log: bool = True,
        log_level: str = "ERROR",
    ):
        super().__init__(message, name, status_code, log, log_level)


class CapabilityDeniedError(PermissionDeniedError):
    """Requester does not hold the capability the operation needs.

    The message is built from the capability name so every check for the same
    capability fails with identical text. That matters where a missing principal
    has to read exactly like a refused request, or the response tells a caller
    which ids exist. The name stays PermissionDeniedError so API responses are
    the same as before this class existed.
    """

    def __init__(self, capability: str):
        super().__init__(
            message=f"User is not authorized to {capability.replace('_', ' ')} for this tenant"
        )


class PermissionNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Permission type does not exist.",
        name: str = "PermissionNotFoundError",
        status_code=status.HTTP_403_FORBIDDEN,
    ):
        super().__init__(message, name, status_code)


class CapabilityNotFoundError(CogneeValidationError):
    """Capability name is not in the CAPABILITY_TYPES catalog"""

    def __init__(
        self,
        message: str = "Capability does not exist.",
        name: str = "CapabilityNotFoundError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
