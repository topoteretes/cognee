from cognee.exceptions import CogneeApiError
from fastapi import status


class EntityNotFoundError(CogneeApiError):
    """Database returns nothing"""

    def __init__(
        self,
        message: str = "The requested entity does not exist.",
        name: str = "EntityNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class EntityAlreadyExistsError(CogneeApiError):
    """Conflict detected, like trying to create a resource that already exists"""

    def __init__(
        self,
        message: str = "The entity already exists.",
        name: str = "EntityAlreadyExistsError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)
