from pydantic import BaseModel
from .VectorConfig import VectorConfig


class CollectionConfig(BaseModel):
    """
    Represent a configuration for a collection of vector embeddings.

    This class is a subclass of BaseModel and encapsulates the configuration details for a
    vector collection.
    Public methods include:

    - __init__() : Initialize a new CollectionConfig instance with a vector configuration.

    Instance variables:

    - vector_config : An instance of VectorConfig representing the configuration for the
    vector embeddings.
    """

    vector_config: VectorConfig
