from typing import List
import json
from dataclasses import dataclass

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult


@dataclass
class TripletNode:
    """Graph node representation from triplet data."""

    id: str
    attributes: dict


@dataclass
class TripletEdge:
    """Graph edge representation from triplet data."""

    node1: TripletNode
    node2: TripletNode
    attributes: dict


class EmbeddedTripletCompletionRetriever(GraphCompletionRetriever):
    """Retriever that uses embedded triplets from vector collection instead of brute force search.

    Data Conversion Rationale:
    GraphCompletionRetriever expects graph objects with .node1/.node2 having .id and .attributes,
    but our triplets are stored as JSON: {"start_node": {"id", "content"}, "relationship", "end_node": {"id", "content"}}.
    We convert triplet format → TripletNode/TripletEdge objects to match the expected interface.
    """

    def __init__(self, collection_name: str = "Triplets", **kwargs):
        """Initialize with configurable collection name."""
        super().__init__(**kwargs)
        self.collection_name = collection_name

    async def get_triplets(self, query: str) -> List[TripletEdge]:
        """Override parent method to use vector search on triplet collection."""
        vector_engine = get_vector_engine()
        search_results = await vector_engine.search(
            collection_name=self.collection_name, query_text=query, limit=self.top_k
        )

        triplet_edges = []
        for search_result in search_results:
            triplet_edge = self._convert_search_result_to_edge(search_result)
            triplet_edges.append(triplet_edge)

        return triplet_edges

    def _parse_triplet_payload(self, search_result: ScoredResult) -> dict:
        """Extract triplet data from search result payload."""
        # ScoredResult.payload contains entire DataPoint structure
        # Actual triplet JSON is nested at payload['payload']
        triplet_json = search_result.payload["payload"]
        return json.loads(triplet_json)

    def _create_triplet_node(self, node_data: dict) -> TripletNode:
        """Convert triplet node data to graph node format."""
        return TripletNode(
            id=node_data["id"],
            attributes={
                "text": node_data.get("content", ""),
                "name": node_data.get("content", "Unnamed Node"),
                "description": node_data.get("content", ""),
            },
        )

    def _create_triplet_edge(
        self, node1: TripletNode, node2: TripletNode, relationship: str
    ) -> TripletEdge:
        """Create graph edge from two nodes and relationship."""
        return TripletEdge(node1=node1, node2=node2, attributes={"relationship_type": relationship})

    def _convert_search_result_to_edge(self, search_result) -> TripletEdge:
        """Main conversion method: ScoredResult → TripletEdge."""
        # 1. Extract triplet data from search result
        triplet_data = self._parse_triplet_payload(search_result)

        # 2. Extract triplet components
        start_node_data = triplet_data["start_node"]
        relationship = triplet_data["relationship"]
        end_node_data = triplet_data["end_node"]

        # 3. Create triplet nodes
        start_node = self._create_triplet_node(start_node_data)
        end_node = self._create_triplet_node(end_node_data)

        # 4. Create triplet edge
        triplet_edge = self._create_triplet_edge(start_node, end_node, relationship)

        return triplet_edge


if __name__ == "__main__":
    import asyncio

    async def main():
        retriever = EmbeddedTripletCompletionRetriever(collection_name="Triplets", top_k=5)

        # Test triplet retrieval with CV/resume data
        query = "machine learning experience Python data scientist"
        print(f"Searching for: {query}")

        triplets = await retriever.get_triplets(query)
        print(f"Found {len(triplets)} triplets")

        # Test full context generation
        context = await retriever.get_context(query)
        print(f"Generated context:\n{context}")

        # Test another query
        query2 = "Stanford University education background"
        print(f"\nSecond search: {query2}")
        triplets2 = await retriever.get_triplets(query2)
        print(f"Found {len(triplets2)} triplets")

    asyncio.run(main())
