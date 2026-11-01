from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeValidationError


class IngestionError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Failed to load data.",
        name: str = "IngestionError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ) -> None:
        super().__init__(message, name, status_code)


class UsageLoggerError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = "Usage logging configuration is invalid.",
        name: str = "UsageLoggerError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(message, name, status_code)


class ExternalSchemaReferenceError(CogneeValidationError):
    """A graph model schema pointed a ``$ref`` outside the document it arrived in."""

    def __init__(
        self,
        message: str = "graph_model contains an external $ref. Only in-document references "
        "starting with '#' are allowed; URLs and file paths are not fetched.",
        name: str = "ExternalSchemaReferenceError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message, name, status_code)


class ReservedGraphModelTitleError(CogneeValidationError):
    """A graph model schema named a type that collides with a cognee DataPoint subclass.

    A schema title becomes the generated class name, which cognee persists as the
    graph node ``type``. Several components resolve nodes by type (tools, skills),
    so a user-supplied title matching a first-party model would let the schema mint
    nodes indistinguishable from cognee's own.
    """

    def __init__(
        self,
        message: str = "graph_model title collides with a registered cognee DataPoint type.",
        name: str = "ReservedGraphModelTitleError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message, name, status_code)
