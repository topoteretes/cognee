from typing import Dict
from pydantic import BaseModel

class DataPoint(BaseModel):
    id: str
    payload: Dict[str, str]
    embed_field: str

    def get_embeddable_data(self):
        return self.payload[self.embed_field]
