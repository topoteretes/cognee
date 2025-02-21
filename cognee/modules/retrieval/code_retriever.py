from typing import Any, Optional

from cognee.low_level import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search


class CodeRetriever(BaseRetriever):
    """Retriever for handling code-based searches."""

    def __init__(self, top_k: int = 5):
        """Initialize retriever with search parameters."""
        self.top_k = top_k

    async def get_context(self, query: str) -> Any:
        """Find relevant code files based on the query."""
        subclasses = get_all_subclasses(DataPoint)
        vector_index_collections = []

        for subclass in subclasses:
            index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
            for field_name in index_fields:
                vector_index_collections.append(f"{subclass.__name__}_{field_name}")

        found_triplets = await brute_force_triplet_search(
            query,
            top_k=self.top_k,
            collections=vector_index_collections or None,
            properties_to_project=["id", "file_path", "source_code"],
        )

        retrieved_files = {}
        for triplet in found_triplets:
            if triplet.node1.attributes["source_code"]:
                retrieved_files[triplet.node1.attributes["file_path"]] = triplet.node1.attributes[
                    "source_code"
                ]
            if triplet.node2.attributes["source_code"]:
                retrieved_files[triplet.node2.attributes["file_path"]] = triplet.node2.attributes[
                    "source_code"
                ]

        return [
            {
                "name": file_path,
                "description": file_path,
                "content": source_code,
            }
            for file_path, source_code in retrieved_files.items()
        ]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Returns the code files context."""
        if context is None:
            context = await self.get_context(query)
        return context
