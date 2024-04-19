from typing import Protocol, Optional, Dict, Any
from abc import abstractmethod

class GraphDBInterface(Protocol):
    @abstractmethod
    async def graph(self):
        raise NotImplementedError

    @abstractmethod
    async def add_node(
        self,
        node_id: str,
        node_properties: dict
    ): raise NotImplementedError

    @abstractmethod
    async def add_nodes(
        self,
        nodes: list[tuple[str, dict]]
    ): raise NotImplementedError

    @abstractmethod
    async def delete_node(
        self,
        node_id: str
    ): raise NotImplementedError

    @abstractmethod
    async def extract_node(
        self,
        node_id: str
    ): raise NotImplementedError

    @abstractmethod
    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = None
    ): raise NotImplementedError

    @abstractmethod
    async def add_edges(
        self,
        edges: tuple[str, str, str, dict]
    ): raise NotImplementedError

    @abstractmethod
    async def delete_graph(
        self,
    ): raise NotImplementedError
