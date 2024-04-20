from typing import Union
from pydantic import BaseModel

class DataPoint(BaseModel):
    id: str
    payload: dict[str, Union[str, dict[str, str]]]
    embed_field: str = "value"

    def get_embeddable_data(self):
        return self.payload[self.embed_field]
