from pydantic import BaseModel
from pydantic import Field


class TruthCentroidPayload(BaseModel):
    dataset_id: str
    slot: int
    count: int
    truth_epoch: int
    updated_at: int
    centroid: list[float]
    learning_ids: list[str] = Field(default_factory=list)
