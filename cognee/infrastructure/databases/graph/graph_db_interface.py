from typing import Protocol, Optional, Dict, Any, List
from abc import abstractmethod, ABC
from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.relationship_manager import create_relationship
from functools import wraps
import inspect
from cognee.modules.users.models import User


def record_graph_changes(func):
    """Decorator to record graph changes in the relationship database."""

    @wraps(func)
    async def wrapper(self, *args, user: User = None, **kwargs):
        # Get caller information
        frame = inspect.currentframe()
        while frame:
            if frame.f_back and frame.f_back.f_code.co_name != "wrapper":
                caller_frame = frame.f_back
                break
            frame = frame.f_back

        caller_name = caller_frame.f_code.co_name
        caller_class = (
            caller_frame.f_locals.get("self", None).__class__.__name__
            if caller_frame.f_locals.get("self", None)
            else None
        )
        creator = f"{caller_class}.{caller_name}" if caller_class else caller_name

        # Execute original function
        result = await func(self, *args, **kwargs)

        if user:
            db_engine = get_relational_engine()
            async with db_engine.get_async_session() as session:
                # For add_nodes
                if func.__name__ == "add_nodes":
                    nodes = args[0]
                    for node_id, node_props in nodes:
                        await create_relationship(
                            session=session,
                            parent_id=node_id,
                            child_id=node_id,
                            creator_function=f"{creator}.node",
                            user_id=user.id,
                        )

                # For add_edges
                elif func.__name__ == "add_edges":
                    edges = args[0]
                    for source_id, target_id, relationship_type, _ in edges:
                        await create_relationship(
                            session=session,
                            parent_id=source_id,
                            child_id=target_id,
                            creator_function=f"{creator}.{relationship_type}",
                            user_id=user.id,
                        )

                await session.commit()

        return result

    return wrapper


class GraphDBInterface(ABC):
    """Interface for graph database operations."""

    @abstractmethod
    async def query(self, query: str, params: dict):
        raise NotImplementedError

    @abstractmethod
    async def add_node(self, node_id: str, node_properties: dict):
        raise NotImplementedError

    @abstractmethod
    @record_graph_changes
    async def add_nodes(self, nodes: list, user: User = None) -> None:
        """Add nodes to the graph database."""
        pass

    @abstractmethod
    async def delete_node(self, node_id: str):
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, node_ids: list[str]):
        raise NotImplementedError

    @abstractmethod
    async def extract_node(self, node_id: str) -> Optional[dict]:
        """Extract a node from the graph database."""
        pass

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
    @record_graph_changes
    async def add_edges(self, edges: list, user: User = None) -> None:
        """Add edges to the graph database."""
        pass

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

    @abstractmethod
    async def has_edges(self, edges: list) -> list:
        """Check if edges exist in the graph database."""
        pass
