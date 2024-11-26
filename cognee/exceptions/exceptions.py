from fastapi import status


class CogneeApiError(Exception):
    """Base exception class"""

    def __init__(
        self,
        message: str = "Service is unavailable",
        name: str = "Cognee",
        status_code=status.HTTP_418_IM_A_TEAPOT,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
        super().__init__(self.message, self.name)


class ServiceError(CogneeApiError):
    """Failures in external services or APIs, like a database or a third-party service"""

    def __init__(
        self,
        message: str = "Service is unavailable",
        name: str = "ServiceError",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
        super().__init__(self.message, self.name)


class EntityDoesNotExistError(CogneeApiError):
    """Database returns nothing"""

    pass


class GroupNotFound(CogneeApiError):
    """User group not found"""

    pass


class EntityAlreadyExistsError(CogneeApiError):
    """Conflict detected, like trying to create a resource that already exists"""

    pass


class InvalidOperationError(CogneeApiError):
    """Invalid operations like trying to delete a non-existing entity, etc."""

    pass


class AuthenticationFailed(CogneeApiError):
    """Invalid authentication credentials"""

    pass


class InvalidTokenError(CogneeApiError):
    """Invalid token"""

    pass
