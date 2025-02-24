from typing import Any, Optional

from cognee.infrastructure.engine import ExtendableDataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.tasks.completion.exceptions import NoRelevantDataFound


class GraphCompletionRetriever(BaseRetriever):
    """Retriever for handling graph-based completion searches."""

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: int = 5,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """Converts retrieved graph edges into a human-readable string format."""
        edge_strings = []
        for edge in retrieved_edges:
            node1_string = edge.node1.attributes.get("text") or edge.node1.attributes.get("name")
            node2_string = edge.node2.attributes.get("text") or edge.node2.attributes.get("name")
            edge_string = edge.attributes["relationship_type"]
            edge_str = f"{node1_string} -- {edge_string} -- {node2_string}"
            edge_strings.append(edge_str)
        return "\n---\n".join(edge_strings)

    async def get_triplets(self, query: str) -> list:
        """Retrieves relevant graph triplets."""
        subclasses = get_all_subclasses(ExtendableDataPoint)
        vector_index_collections = []

        for subclass in subclasses:
            index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
            for field_name in index_fields:
                vector_index_collections.append(f"{subclass.__name__}_{field_name}")

        found_triplets = await brute_force_triplet_search(
            query, top_k=self.top_k, collections=vector_index_collections or None
        )

        if len(found_triplets) == 0:
            raise NoRelevantDataFound

        return found_triplets

    async def get_context(self, query: str) -> Any:
        """Retrieves and resolves graph triplets into context."""
        triplets = await self.get_triplets(query)
        return await self.resolve_edges_to_text(triplets)

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using graph connections context."""
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]
