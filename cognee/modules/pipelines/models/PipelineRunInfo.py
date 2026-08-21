from typing import Any, Optional, List, Union
from uuid import UUID
from pydantic import BaseModel, field_serializer
from cognee.modules.data.models.Data import Data


class PipelineRunInfo(BaseModel):
    status: str
    pipeline_run_id: UUID
    dataset_id: UUID
    dataset_name: str
    # Data must be mentioned in typing to allow custom encoders for Data to be activated
    payload: Optional[Union[Any, List[Data]]] = None
    data_ingestion_info: Optional[list] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
     
    }

    @field_serializer("payload")
    def serialize_payload(self, payload, _info):
        if isinstance(payload, list):
            return [item.to_json() if isinstance(item, Data) else item for item in payload]
        if isinstance(payload, Data):
            return payload.to_json()
        return payload

class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"
    pass


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"
    pass


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"
    pass


class PipelineRunAlreadyCompleted(PipelineRunInfo):
    status: str = "PipelineRunAlreadyCompleted"
    pass


class PipelineRunErrored(PipelineRunInfo):
    status: str = "PipelineRunErrored"
    pass
