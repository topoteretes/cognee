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
