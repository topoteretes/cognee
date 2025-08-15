from cognee.exceptions import CogneeSystemError
from fastapi import status


class OntologyInitializationError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Ontology initialization failed",
        name: str = "OntologyInitializationError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class FindClosestMatchError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Error in find_closest_match",
        name: str = "FindClosestMatchError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class GetSubgraphError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Failed to retrieve subgraph",
        name: str = "GetSubgraphError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)
