from cognee.exceptions import CogneeApiError
from fastapi import status


class NoRelevantDataError(CogneeApiError):
    def __init__(
        self,
        message: str = "Search did not find any data.",
        name: str = "NoRelevantDataError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
