from typing import Protocol, Optional, Dict, Any
from abc import abstractmethod


class GraphDBInterface(Protocol):
    @abstractmethod
    async def query(self, query: str, params: dict):
        raise NotImplementedError

    @abstractmethod
    async def add_node(self, node_id: str, node_properties: dict):
        raise NotImplementedError

    @abstractmethod
    async def add_nodes(self, nodes: list[tuple[str, dict]]):
        raise NotImplementedError

    @abstractmethod
    async def delete_node(self, node_id: str):
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, node_ids: list[str]):
        raise NotImplementedError

    @abstractmethod
    async def extract_node(self, node_id: str):
        raise NotImplementedError

    @abstractmethod
    async def extract_nodes(self, node_ids: list[str]):
        raise NotImplementedError

    @abstractmethod
    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def add_edges(self, edges: tuple[str, str, str, dict]):
        raise NotImplementedError

    @abstractmethod
    async def delete_graph(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    async def get_graph_data(self):
        raise NotImplementedError

    @abstractmethod
    async def get_graph_metrics(self, include_optional):
        """ "https://docs.cognee.ai/core_concepts/graph_generation/descriptive_metrics"""
        raise NotImplementedError
