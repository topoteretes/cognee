from typing import Literal
from pydantic import BaseModel


class VectorConfig(BaseModel):
    """
    Represent a configuration for vector operations, including distance metric and size.

    Public methods include: None
    Instance variables: distance, size
    """

    distance: Literal["Cosine", "Dot"]
    size: int
