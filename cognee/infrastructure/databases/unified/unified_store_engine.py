from __future__ import annotations

from typing import Optional, cast

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface

from .capabilities import EngineCapability


class UnifiedStoreEngine:
    """Facade that wraps graph and vector engines with capability flags.

    For separate backends (e.g. Ladybug + LanceDB), holds two independent engine
    instances.  For hybrid backends (e.g. Neptune Analytics), both properties
    point to the same adapter object.

    The pipeline can check ``has_capability()`` to decide whether to optimise
    writes or searches for hybrid backends.
    """

    def __init__(
        self,
        graph_engine: Optional[GraphDBInterface] = None,
        vector_engine: Optional[VectorDBInterface] = None,
        capabilities: EngineCapability = EngineCapability.NONE,
    ):
        self._graph = graph_engine
        self._vector = vector_engine
        self._capabilities = capabilities

    @property
    def capabilities(self) -> EngineCapability:
        return self._capabilities

    def has_capability(self, cap: EngineCapability) -> bool:
        return bool(self._capabilities & cap)

    @property
    def graph(self) -> GraphDBInterface:
        if not self.has_capability(EngineCapability.GRAPH) or self._graph is None:
            raise RuntimeError(
                "This UnifiedStoreEngine has no GRAPH capability. "
                "Check has_capability(EngineCapability.GRAPH) before accessing .graph"
            )
        return self._graph

    @property
    def vector(self) -> VectorDBInterface:
        if not self.has_capability(EngineCapability.VECTOR) or self._vector is None:
            raise RuntimeError(
                "This UnifiedStoreEngine has no VECTOR capability. "
                "Check has_capability(EngineCapability.VECTOR) before accessing .vector"
            )
        return cast(VectorDBInterface, self._vector)

    @property
    def is_hybrid(self) -> bool:
        return self.has_capability(EngineCapability.HYBRID_WRITE) or self.has_capability(
            EngineCapability.HYBRID_SEARCH
        )

    @property
    def is_same_backend(self) -> bool:
        return self._graph is not None and self._graph is self._vector
