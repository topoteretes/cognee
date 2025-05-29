import inspect
from functools import wraps
from abc import abstractmethod, ABC
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from uuid import NAMESPACE_OID, UUID, uuid5
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models.graph_relationship_ledger import GraphRelationshipLedger
from cognee.infrastructure.databases.relational.get_relational_engine import get_relational_engine

logger = get_logger()

# Type aliases for better readability
NodeData = Dict[str, Any]
EdgeData = Tuple[
    str, str, str, Dict[str, Any]
]  # (source_id, target_id, relationship_name, properties)
Node = Tuple[str, NodeData]  # (node_id, properties)


def record_graph_changes(func):
    """Decorator to record graph changes in the relationship database."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        db_engine = get_relational_engine()
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

        result = await func(self, *args, **kwargs)

        async with db_engine.get_async_session() as session:
            if func.__name__ == "add_nodes":
                nodes: List[DataPoint] = args[0]
                for node in nodes:
                    try:
                        node_id = UUID(str(node.id))
                        relationship = GraphRelationshipLedger(
                            id=uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
                            source_node_id=node_id,
                            destination_node_id=node_id,
                            creator_function=f"{creator}.node",
                            node_label=getattr(node, "name", None) or str(node.id),
                        )
                        session.add(relationship)
                        await session.flush()
                    except Exception as e:
                        logger.debug(f"Error adding relationship: {e}")
                        await session.rollback()
                        continue

            elif func.__name__ == "add_edges":
                edges = args[0]
                for edge in edges:
                    try:
                        source_id = UUID(str(edge[0]))
                        target_id = UUID(str(edge[1]))
                        rel_type = str(edge[2])
                        relationship = GraphRelationshipLedger(
                            id=uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
                            source_node_id=source_id,
                            destination_node_id=target_id,
                            creator_function=f"{creator}.{rel_type}",
                        )
                        session.add(relationship)
                        await session.flush()
                    except Exception as e:
                        logger.debug(f"Error adding relationship: {e}")
                        await session.rollback()
                        continue

            try:
                await session.commit()
            except Exception as e:
                logger.debug(f"Error committing session: {e}")

        return result

    return wrapper


class GraphDBInterface(ABC):
    """Interface for graph database operations."""

    @abstractmethod
    async def query(self, query: str, params: dict) -> List[Any]:
        """Execute a raw query against the database."""
        raise NotImplementedError

    @abstractmethod
    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Add a single node to the graph."""
        raise NotImplementedError

    @abstractmethod
    @record_graph_changes
    async def add_nodes(self, nodes: List[Node]) -> None:
        """Add multiple nodes to the graph."""
        raise NotImplementedError

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes from the graph."""
        raise NotImplementedError

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get a single node by ID."""
        raise NotImplementedError

    @abstractmethod
    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        """Get multiple nodes by their IDs."""
        raise NotImplementedError

    @abstractmethod
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single edge to the graph."""
        raise NotImplementedError

    @abstractmethod
    @record_graph_changes
    async def add_edges(self, edges: List[EdgeData]) -> None:
        """Add multiple edges to the graph."""
        raise NotImplementedError

    @abstractmethod
    async def delete_graph(self) -> None:
        """Delete the entire graph."""
        raise NotImplementedError

    @abstractmethod
    async def get_graph_data(self) -> Tuple[List[Node], List[EdgeData]]:
        """Get all nodes and edges in the graph."""
        raise NotImplementedError

    @abstractmethod
    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Get graph metrics and statistics."""
        raise NotImplementedError

    @abstractmethod
    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check if an edge exists."""
        raise NotImplementedError

    @abstractmethod
    async def has_edges(self, edges: List[EdgeData]) -> List[EdgeData]:
        """Check if multiple edges exist."""
        raise NotImplementedError

    @abstractmethod
    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """Get all edges connected to a node."""
        raise NotImplementedError

    @abstractmethod
    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """Get all neighboring nodes."""
        raise NotImplementedError

    @abstractmethod
    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[NodeData, Dict[str, Any], NodeData]]:
        """Get all nodes connected to a given node with their relationships."""
        raise NotImplementedError
