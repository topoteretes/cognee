from cognee.exceptions import CogneeApiError
from fastapi import status


class NoRelevantDataFound(CogneeApiError):
    def __init__(
        self,
        message: str = "Search did not find any data.",
        name: str = "NoRelevantDataFound",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
