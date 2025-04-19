from fastapi import status
from cognee.exceptions import CogneeApiError, CriticalError


class DatabaseNotCreatedError(CriticalError):
    def __init__(
        self,
        message: str = "The database has not been created yet. Please call `await setup()` first.",
        name: str = "DatabaseNotCreatedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)


class EntityNotFoundError(CogneeApiError):
    """Database returns nothing"""

    def __init__(
        self,
        message: str = "The requested entity does not exist.",
        name: str = "EntityNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
        # super().__init__(message, name, status_code) :TODO: This is not an error anymore with the dynamic exception handling therefore we shouldn't log error


class EntityAlreadyExistsError(CogneeApiError):
    """Conflict detected, like trying to create a resource that already exists"""

    def __init__(
        self,
        message: str = "The entity already exists.",
        name: str = "EntityAlreadyExistsError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)
