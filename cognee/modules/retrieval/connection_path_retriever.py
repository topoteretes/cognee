from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion

logger = get_logger("ConnectionPathRetriever")

NO_PATH_MESSAGE = "No connecting path was found between the two entities within {max_depth} hops."

_ANCHOR_SYSTEM_PROMPT = (
    "You are given a question that asks how two entities are connected or related. "
    "Identify exactly the two entities the user wants to connect. Return the first one "
    "as `source` and the second as `target`, using the entity names as they appear in "
    "the question, without extra words."
)

# In cognee's graph every entity hangs off the same DocumentChunk via "contains" edges,
# so a path search must skip these structural nodes to follow real relationships instead
# of hopping through the shared chunk.
_STRUCTURAL_NODE_TYPES = [
    "DocumentChunk",
    "TextDocument",
    "TextSummary",
    "EntityType",
    "NodeSet",
]


class _PathAnchors(BaseModel):
    """Two entity anchors extracted from a connection-path question."""

    source: str = Field(description="The first entity to connect from.")
    target: str = Field(description="The second entity to connect to.")


class ConnectionPathRetriever(BaseRetriever):
    """
    Retriever that explains how two entities are connected.

    Given a question like "How is A connected to B?", it extracts the two anchor
    entities, resolves each to a graph node via vector search, finds the connecting
    path with the backend-agnostic ``find_paths`` primitive, and asks the LLM to explain
    the resulting relationship chain. When no path exists within ``max_depth`` hops it
    returns a clear "no path" answer instead of a hallucinated one.
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        entity_collection: str = "Entity_name",
        max_depth: int = 5,
        excluded_node_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        """Initialize retriever with prompt paths and pathfinding parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.entity_collection = entity_collection
        self.max_depth = max_depth
        self.excluded_node_types = (
            excluded_node_types if excluded_node_types is not None else _STRUCTURAL_NODE_TYPES
        )
        self.session_id = session_id
        self.response_model = response_model

    async def _extract_anchors(self, query: Optional[str]) -> Optional[Tuple[str, str]]:
        """Use the LLM to pull the two entity anchors out of the question."""
        if not query:
            return None
        try:
            anchors = await LLMGateway.acreate_structured_output(
                text_input=query,
                system_prompt=_ANCHOR_SYSTEM_PROMPT,
                response_model=_PathAnchors,
            )
        except Exception as error:
            logger.error(f"Failed to extract path anchors: {error}")
            return None

        source = (anchors.source or "").strip()
        target = (anchors.target or "").strip()
        if not source or not target:
            return None
        return source, target

    async def _resolve_anchor(self, vector_engine, anchor: str) -> Optional[str]:
        """Resolve an anchor string to the id of the closest entity node."""
        try:
            results = await vector_engine.search(self.entity_collection, query_text=anchor, limit=1)
        except Exception as error:
            logger.warning(f"Could not resolve anchor '{anchor}': {error}")
            return None
        if not results:
            return None
        return str(results[0].id)

    @staticmethod
    def _node_name(node: Any) -> str:
        if isinstance(node, dict):
            return str(node.get("name") or node.get("id") or "unknown")
        return str(node)

    def _path_to_edges(self, path) -> List[Edge]:
        """Convert ``(source, edge, target)`` triplets into CogneeGraph Edge objects."""
        edges: List[Edge] = []
        for source_node, edge_data, target_node in path:
            node1 = Node(str(source_node.get("id")), {"name": self._node_name(source_node)})
            node2 = Node(str(target_node.get("id")), {"name": self._node_name(target_node)})
            relationship_name = ""
            if isinstance(edge_data, dict):
                relationship_name = (
                    edge_data.get("relationship_name") or edge_data.get("relationship_type") or ""
                )
            edges.append(Edge(node1, node2, {"relationship_name": relationship_name}))
        return edges

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Edge]:
        """Resolve the two anchors and return the connecting path as a list of edges."""
        anchors = await self._extract_anchors(query)
        if not anchors:
            return []
        source_anchor, target_anchor = anchors

        graph_engine = await get_graph_engine()
        if await graph_engine.is_empty():
            logger.warning("Search attempt on an empty knowledge graph")
            return []

        vector_engine = get_vector_engine()
        source_id = await self._resolve_anchor(vector_engine, source_anchor)
        target_id = await self._resolve_anchor(vector_engine, target_anchor)
        if not source_id or not target_id:
            return []

        paths = await graph_engine.find_paths(
            source_id,
            target_id,
            max_depth=self.max_depth,
            excluded_node_types=self.excluded_node_types,
        )
        if not paths or not paths[0]:
            return []
        return self._path_to_edges(paths[0])

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
    ) -> str:
        """Render the ordered path into a readable chain, or a clear no-path message."""
        edges = retrieved_objects or []
        if not edges:
            return NO_PATH_MESSAGE.format(max_depth=self.max_depth)

        steps = []
        for edge in edges:
            source = edge.node1.attributes.get("name")
            target = edge.node2.attributes.get("name")
            relationship = edge.attributes.get("relationship_name") or "related to"
            steps.append(f"{source} --[{relationship}]--> {target}")
        return "Connection path:\n" + "\n".join(steps)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> List[Any]:
        """Explain the connecting path, or return the no-path message as-is."""
        if not retrieved_objects:
            # No path found: return the message directly, no LLM call and no guessing.
            return [context]

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
        )
        return [completion]
