from cognee.exceptions import CogneeValidationError
from fastapi import status


class EntityNotFoundError(CogneeValidationError):
    """Database returns nothing"""

    def __init__(
        self,
        message: str = "The requested entity does not exist.",
        name: str = "EntityNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class EntityAlreadyExistsError(CogneeValidationError):
    """Conflict detected, like trying to create a resource that already exists"""

    def __init__(
        self,
        message: str = "The entity already exists.",
        name: str = "EntityAlreadyExistsError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)


class InvalidDimensionsError(CogneeValidationError):
    def __init__(
        self,
        name: str = "InvalidDimensionsError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = "Dimensions must be positive integers."
        super().__init__(message, name, status_code)


class DimensionOutOfRangeError(CogneeValidationError):
    def __init__(
        self,
        dimension: int,
        max_index: int,
        name: str = "DimensionOutOfRangeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"Dimension {dimension} is out of range. Valid range is 0 to {max_index}."
        super().__init__(message, name, status_code)
