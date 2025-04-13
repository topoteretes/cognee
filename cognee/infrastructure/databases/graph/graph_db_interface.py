from typing import Protocol, Optional, Dict, Any, List
from abc import abstractmethod, ABC
from uuid import UUID, uuid5
from cognee.modules.graph.relationship_manager import create_relationship
from functools import wraps
import inspect
from cognee.modules.data.models.graph_relationship_ledger import GraphRelationshipLedger
from cognee.infrastructure.databases.relational.get_relational_engine import get_relational_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def record_graph_changes(func):
    """Decorator to record graph changes in the relationship database."""
    # Get the engine once when the decorator is defined
    db_engine = get_relational_engine()

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Get caller information for logging
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

        async with db_engine.get_async_session() as session:
            # For add_nodes
            if func.__name__ == "add_nodes":
                nodes = args[0]
                if isinstance(nodes, list):
                    for node in nodes:
                        try:
                            # Handle DataPoint objects (original input)
                            if hasattr(node, "id"):
                                node_id = node.id  # Already a UUID object
                                node_label = type(node).__name__
                            # Handle Neo4j dictionary format
                            elif isinstance(node, dict) and "node_id" in node:
                                node_id = UUID(str(node["node_id"]))
                                node_label = node.get("label")
                            # Handle tuple format
                            elif isinstance(node, tuple) and len(node) >= 1:
                                node_id = UUID(str(node[0]))
                                if len(node) > 1 and isinstance(node[1], dict):
                                    node_label = node[1].get("type") or node[1].get("label")
                                else:
                                    node_label = "Unknown"
                            else:
                                logger.error(f"DEBUG: Unhandled node format: {type(node)}")
                                continue

                            relationship = GraphRelationshipLedger(
                                id=uuid5(),
                                source_node_id=node_id,  # Now a UUID object
                                destination_node_id=node_id,  # Now a UUID object
                                creator_function=f"{creator}.node",
                                node_label=node_label,
                            )
                            session.add(relationship)
                            await session.flush()
                        except Exception as e:
                            logger.error(f"DEBUG: Error adding relationship: {e}")
                            await session.rollback()  # Explicitly rollback on error
                            continue  # Continue with next node

            # For add_edges
            elif func.__name__ == "add_edges":
                edges = args[0]
                if isinstance(edges, list):
                    for edge in edges:
                        try:
                            # Handle Neo4j format
                            if isinstance(edge, dict):
                                source_id = UUID(str(edge.get("from_node")))
                                target_id = UUID(str(edge.get("to_node")))
                                rel_type = str(edge.get("relationship_name"))
                            # Handle tuple format
                            elif isinstance(edge, tuple):
                                source_id = UUID(str(edge[0]))
                                target_id = UUID(str(edge[1]))
                                rel_type = str(edge[2]) if len(edge) > 2 else "UNKNOWN"
                            else:
                                logger.error(f"DEBUG: Unhandled edge format: {type(edge)}")
                                continue

                            relationship = GraphRelationshipLedger(
                                id=uuid5(),
                                source_node_id=source_id,
                                destination_node_id=target_id,
                                creator_function=f"{creator}.{rel_type}",
                            )
                            session.add(relationship)
                            await session.flush()
                        except Exception as e:
                            logger.error(f"DEBUG: Error adding relationship: {e}")

            try:
                await session.commit()
            except Exception as e:
                logger.error(f"DEBUG: Error committing session: {e}")

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
    async def add_nodes(self, nodes: list) -> None:
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
    async def add_edges(self, edges: list) -> None:
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

    @abstractmethod
    async def get_document_subgraph(self, content_hash: str) -> Dict[str, list]:
        """Get all nodes connected to a document that should be deleted with it.

        Returns:
            Dict with keys: 'document', 'chunks', 'orphan_entities',
            'made_from_nodes', 'orphan_types'
        """
        pass
