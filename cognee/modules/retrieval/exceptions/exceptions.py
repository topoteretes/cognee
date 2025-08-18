from fastapi import status
from cognee.exceptions import CogneeValidationError, CogneeSystemError


class SearchTypeNotSupported(CogneeValidationError):
    def __init__(
        self,
        message: str = "CYPHER search type not supported by the adapter.",
        name: str = "SearchTypeNotSupported",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class CypherSearchError(CogneeSystemError):
    def __init__(
        self,
        message: str = "An error occurred during the execution of the Cypher query.",
        name: str = "CypherSearchError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class NoDataError(CogneeValidationError):
    def __init__(
        self,
        message: str = "No data found in the system, please add data first.",
        name: str = "NoDataError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class CollectionDistancesNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "No collection distances found for the given query.",
        name: str = "CollectionDistancesNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
