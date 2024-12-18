from typing import Literal
from pydantic import BaseModel


class VectorConfig(BaseModel):
    distance: Literal["Cosine", "Dot"]
    size: int
