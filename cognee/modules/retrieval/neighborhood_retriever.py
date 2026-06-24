"""First-class neighborhood search retriever.

This exposes the existing neighborhood projection path as a dedicated
SearchType.NEIGHBORHOOD and forwards edge_types to the existing graph flow.
"""

from typing import List, Optional, Union

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search


class NeighborhoodRetriever(GraphCompletionRetriever):
    def __init__(
        self,
        depth: int = 2,
        seed_top_k: int = 10,
        edge_types: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            neighborhood_depth=depth,
            neighborhood_seed_top_k=seed_top_k,
            **kwargs,
        )
        self.edge_types = edge_types

    async def get_triplets(
        self, query=None, query_batch=None
    ) -> Union[List[Edge], List[List[Edge]]]:
        collections = self._get_vector_index_collections()
        unified_engine = getattr(self, "_unified_engine", None)
        return await brute_force_triplet_search(
            query,
            query_batch,
            top_k=self.top_k,
            collections=collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
            wide_search_top_k=self.wide_search_top_k,
            triplet_distance_penalty=self.triplet_distance_penalty,
            feedback_influence=self.feedback_influence,
            unified_engine=unified_engine,
            neighborhood_depth=self.neighborhood_depth,
            neighborhood_seed_top_k=self.neighborhood_seed_top_k,
            edge_types=self.edge_types,
        )
