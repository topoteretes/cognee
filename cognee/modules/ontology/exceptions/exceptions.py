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


class EmptyOntologyInStrictModeError(CogneeSystemError):
    def __init__(
        self,
        message: str = (
            "ONTOLOGY_MODE=strict requires an ontology with at least one class or "
            "individual, but the configured ontology is empty (often a mistyped "
            "ONTOLOGY_FILE_PATH). Strict mode would drop every extracted entity, "
            "so the run is refused instead."
        ),
        name: str = "EmptyOntologyInStrictModeError",
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
