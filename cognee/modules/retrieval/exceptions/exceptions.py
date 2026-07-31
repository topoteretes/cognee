from typing import Literal

from fastapi import status

from cognee.exceptions import (
    CogneeDataNotReadyError,
    CogneeSystemError,
    CogneeValidationError,
    ErrorCode,
)


class SearchTypeNotSupported(CogneeValidationError):
    def __init__(
        self,
        message: str = "CYPHER search type not supported by the adapter.",
        name: str = "SearchTypeNotSupported",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        remediation: str | None = "Pick a search type supported by your graph/vector backend.",
    ):
        super().__init__(message, name, status_code, remediation=remediation)


class CypherSearchError(CogneeSystemError):
    def __init__(
        self,
        message: str = "An error occurred during the execution of the Cypher query.",
        name: str = "CypherSearchError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class NoDataError(CogneeDataNotReadyError):
    _STAGE_REMEDIATION = {
        "add": "Add data with cognee.add() first.",
        "cognify": "Run cognee.cognify() after adding data.",
    }
    _DEFAULT_REMEDIATION = (
        "Add data with cognee.add(), then run cognee.cognify() before searching."
    )

    def __init__(
        self,
        message: str = "No data found in the system, please add data first.",
        name: str = "NoDataError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        stage: Literal["add", "cognify"] | None = None,
        log: bool = True,
        log_level: str = "ERROR",
    ):
        details = {"stage": stage} if stage else None
        remediation = (
            self._STAGE_REMEDIATION[stage]
            if stage in self._STAGE_REMEDIATION
            else self._DEFAULT_REMEDIATION
        )
        super().__init__(
            message,
            name,
            status_code,
            log=log,
            log_level=log_level,
            code=ErrorCode.DATA_NOT_READY,
            remediation=remediation,
            details=details,
        )


class CollectionDistancesNotFoundError(CogneeDataNotReadyError):
    def __init__(
        self,
        message: str = "No collection distances found for the given query.",
        name: str = "CollectionDistancesNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class QueryValidationError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Queries not supplied in the correct format.",
        name: str = "QueryValidationError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)
