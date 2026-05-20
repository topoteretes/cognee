from typing import Any, Optional

from cognee.exceptions import CogneeSystemError
from fastapi import status


class PipelineItemFailure:
    """Per-item failure record: which data item failed and the original
    exception that caused it. Carried inside PipelineRunFailedError so
    callers can introspect which items failed and why."""

    def __init__(self, data_id: Any, exception: BaseException) -> None:
        self.data_id = data_id
        self.exception = exception

    def __repr__(self) -> str:
        return f"PipelineItemFailure(data_id={self.data_id!r}, exception={self.exception!r})"


class PipelineRunFailedError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Pipeline run failed.",
        name: str = "PipelineRunFailedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        *,
        item_failures: Optional[list[PipelineItemFailure]] = None,
    ):
        super().__init__(message, name, status_code)
        self.item_failures: list[PipelineItemFailure] = item_failures or []
