from enum import Flag, auto


class EngineCapability(Flag):
    """Capability flags for unified store engines.

    Describes what operations a store engine supports:
    - GRAPH: Can perform graph operations (implements GraphDBInterface)
    - VECTOR: Can perform vector operations (implements VectorDBInterface)
    - HYBRID_WRITE: Supports atomic graph+vector writes in a single backend
    - HYBRID_SEARCH: Supports combined graph+vector queries in a single backend
    """

    NONE = 0
    GRAPH = auto()
    VECTOR = auto()
    HYBRID_WRITE = auto()
    HYBRID_SEARCH = auto()
