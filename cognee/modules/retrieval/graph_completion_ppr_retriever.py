from typing import List, Optional, Type, Union

from cognee.base_config import get_base_config
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.ppr_triplet_search import (
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_PPR_ALPHA,
    DEFAULT_PPR_WEIGHT,
    ppr_triplet_search,
)


class GraphCompletionPPRRetriever(GraphCompletionRetriever):
    """Graph-completion retriever that ranks triplets with Personalized PageRank.

    Identical to ``GraphCompletionRetriever`` except for the retrieval core: instead
    of ranking each triplet by the vector distance of its own elements, it seeds
    Personalized PageRank on the vector hits and lets relevance spread through the
    graph, so facts that bridge several query-relevant entities surface even when
    their own text is not similar to the query (the multi-hop case a vector-only
    ranking misses). All context-building and completion behavior is inherited.
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 6.5,
        feedback_influence: float = get_base_config().default_feedback_influence,
        session_id: Optional[str] = None,
        response_model: Type = str,
        neighborhood_depth: Optional[int] = DEFAULT_NEIGHBORHOOD_DEPTH,
        neighborhood_seed_top_k: Optional[int] = 10,
        include_global_context_index: bool = False,
        global_context_index_top_k: int = 3,
        include_references: bool = False,
        ppr_alpha: Optional[float] = DEFAULT_PPR_ALPHA,
        ppr_weight: Optional[float] = DEFAULT_PPR_WEIGHT,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            feedback_influence=feedback_influence,
            session_id=session_id,
            response_model=response_model,
            neighborhood_depth=neighborhood_depth,
            neighborhood_seed_top_k=neighborhood_seed_top_k,
            include_global_context_index=include_global_context_index,
            global_context_index_top_k=global_context_index_top_k,
            include_references=include_references,
        )
        # Coalesce None to defaults so the search-type dispatch can forward raw
        # retriever-specific config (which is None when unset) without overriding
        # the defaults.
        self.ppr_alpha = ppr_alpha if ppr_alpha is not None else DEFAULT_PPR_ALPHA
        self.ppr_weight = ppr_weight if ppr_weight is not None else DEFAULT_PPR_WEIGHT

    async def get_triplets(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
    ) -> Union[List[Edge], List[List[Edge]]]:
        """Retrieve triplets via Personalized PageRank.

        Single query -> a flat list of edges. Batch -> one list of edges per query
        (each query is scored independently, since PPR is seeded per query).
        """
        collections = self._get_vector_index_collections()
        unified_engine = getattr(self, "_unified_engine", None)

        async def _search(single_query: str) -> List[Edge]:
            return await ppr_triplet_search(
                single_query,
                top_k=self.top_k,
                collections=collections or None,
                node_type=self.node_type,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
                wide_search_top_k=self.wide_search_top_k,
                triplet_distance_penalty=self.triplet_distance_penalty,
                feedback_influence=self.feedback_influence,
                ppr_alpha=self.ppr_alpha,
                ppr_weight=self.ppr_weight,
                neighborhood_depth=self.neighborhood_depth,
                neighborhood_seed_top_k=self.neighborhood_seed_top_k,
                unified_engine=unified_engine,
            )

        if query_batch:
            return [await _search(single_query) for single_query in query_batch]
        return await _search(query)
