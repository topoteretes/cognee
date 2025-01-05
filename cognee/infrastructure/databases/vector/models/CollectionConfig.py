from pydantic import BaseModel
from .VectorConfig import VectorConfig


class CollectionConfig(BaseModel):
    vector_config: VectorConfig
