from cognee.exceptions import CogneeApiError
from fastapi import status


class CollectionDistancesNotFoundError(CogneeApiError):
    def __init__(
        self,
        message: str = "No distances found between the query and collections. It is possible that the given collection names don't exist.",
        name: str = "CollectionDistancesNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)

      
class SearchTypeNotSupported(CogneeApiError):
    def __init__(
        self,
        message: str = "CYPHER search type not supported by the adapter.",
        name: str = "SearchTypeNotSupported",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class CypherSearchError(CogneeApiError):
    def __init__(
        self,
        message: str = "An error occurred during the execution of the Cypher query.",
        name: str = "CypherSearchError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
