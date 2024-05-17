from typing import Generic, TypeVar
from pydantic import BaseModel

PayloadSchema = TypeVar("PayloadSchema", bound = BaseModel)

class DataPoint(BaseModel, Generic[PayloadSchema]):
    id: str
    payload: PayloadSchema
    embed_field: str = "value"

    def get_embeddable_data(self):
        if hasattr(self.payload, self.embed_field):
            return getattr(self.payload, self.embed_field)
