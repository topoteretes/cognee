from pydantic import BaseModel


class TruthCentroidPayload(BaseModel):
    dataset_id: str
    slot: int
    count: int
    truth_epoch: int
    updated_at: int
    centroid: list[float]
