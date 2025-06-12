from typing import Any, Optional
from uuid import UUID
from pydantic import BaseModel


class PipelineRunInfo(BaseModel):
    status: str
    pipeline_run_id: UUID
    payload: Optional[Any] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }


class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"
    pass


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"
    pass


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"
    pass


class PipelineRunErrored(PipelineRunInfo):
    status: str = "PipelineRunErrored"
    pass
