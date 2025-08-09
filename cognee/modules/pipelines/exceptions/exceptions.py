from cognee.exceptions import CogneeApiError
from fastapi import status


class PipelineRunFailedError(CogneeApiError):
    def __init__(
        self,
        message: str = "Pipeline run failed.",
        name: str = "PipelineRunFailedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)
