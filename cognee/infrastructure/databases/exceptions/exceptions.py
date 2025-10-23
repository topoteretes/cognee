from fastapi import status
from cognee.exceptions import CogneeSystemError, CogneeValidationError, CogneeConfigurationError


class DatabaseNotCreatedError(CogneeSystemError):
    """
    Represents an error indicating that the database has not been created. This error should
    be raised when an attempt is made to access the database before it has been initialized.

    Inherits from CogneeSystemError. Overrides the constructor to include a default message and
    status code.
    """

    def __init__(
        self,
        message: str = "The database has not been created yet. Please call `await setup()` first.",
        name: str = "DatabaseNotCreatedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class EntityNotFoundError(CogneeValidationError):
    """
    Represents an error when a requested entity is not found in the database. This class
    inherits from CogneeValidationError.

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


class EntityAlreadyExistsError(CogneeValidationError):
    """
    Represents an error when an entity creation is attempted but the entity already exists.

    This class is derived from CogneeValidationError and is used to signal a conflict in operations
    involving resource creation.
    """

    def __init__(
        self,
        message: str = "The entity already exists.",
        name: str = "EntityAlreadyExistsError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)


class NodesetFilterNotSupportedError(CogneeConfigurationError):
    """
    Raise an exception when a nodeset filter is not supported by the current database.

    This exception inherits from `CogneeConfigurationError` and is designed to provide information
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


class EmbeddingException(CogneeConfigurationError):
    """
    Custom exception for handling embedding-related errors.

    This exception class is designed to indicate issues specifically related to embeddings
    within the application. It extends the base exception class CogneeConfigurationError allows
    for customization of the error message, name, and status code.
    """

    def __init__(
        self,
        message: str = "Embedding Exception.",
        name: str = "EmbeddingException",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class MissingQueryParameterError(CogneeValidationError):
    """
    Raised when neither 'query_text' nor 'query_vector' is provided,
    and at least one is required to perform the operation.
    """

    def __init__(
        self,
        name: str = "MissingQueryParameterError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = "One of query_text or query_vector must be provided!"
        super().__init__(message, name, status_code)


class MutuallyExclusiveQueryParametersError(CogneeValidationError):
    """
    Raised when both 'text' and 'embedding' are provided to the search function,
    but only one type of input is allowed at a time.
    """

    def __init__(
        self,
        name: str = "MutuallyExclusiveQueryParametersError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = "The search function accepts either text or embedding as input, but not both."
        super().__init__(message, name, status_code)


class CacheConnectionError(CogneeConfigurationError):
    """
    Raised when connection to the cache database (e.g., Redis) fails.

    This error indicates that the cache service is unavailable or misconfigured.
    """

    def __init__(
        self,
        message: str = "Failed to connect to cache database. Please check your cache configuration.",
        name: str = "CacheConnectionError",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
    ):
        super().__init__(message, name, status_code)
