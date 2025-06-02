from fastapi import status
from cognee.exceptions import CogneeApiError, CriticalError


class DatabaseNotCreatedError(CriticalError):
    """
    Represents an error indicating that the database has not been created. This error should
    be raised when an attempt is made to access the database before it has been initialized.

    Inherits from CriticalError. Overrides the constructor to include a default message and
    status code.
    """

    def __init__(
        self,
        message: str = "The database has not been created yet. Please call `await setup()` first.",
        name: str = "DatabaseNotCreatedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)


class EntityNotFoundError(CogneeApiError):
    """
    Represents an error when a requested entity is not found in the database. This class
    inherits from CogneeApiError.

    Public methods:

    - __init__ : Initializes the EntityNotFoundError with a specific message, name, and
    status code.

    Instance variables:

    - message: A string containing the error message.
    - name: A string representing the name of the error type.
    - status_code: An integer indicating the HTTP status code associated with the error.
    """

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
    """
    Represents an error when an entity creation is attempted but the entity already exists.

    This class is derived from CogneeApiError and is used to signal a conflict in operations
    involving resource creation.
    """

    def __init__(
        self,
        message: str = "The entity already exists.",
        name: str = "EntityAlreadyExistsError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)


class NodesetFilterNotSupportedError(CogneeApiError):
    """
    Raise an exception when a nodeset filter is not supported by the current database.

    This exception inherits from `CogneeApiError` and is designed to provide information
    about the specific issue of unsupported nodeset filters in the context of graph
    databases.
    """

    def __init__(
        self,
        message: str = "The nodeset filter is not supported in the current graph database.",
        name: str = "NodeSetFilterNotSupportedError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        self.message = message
        self.name = name
        self.status_code = status_code
