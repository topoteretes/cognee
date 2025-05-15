from fastapi import status
from cognee.exceptions import CogneeApiError, CriticalError


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


class NoDataError(CriticalError):
    message: str = "No data found in the system, please add data first."
