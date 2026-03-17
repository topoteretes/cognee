import asyncio
from typing import Optional, List, Type, Union

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.query_state import QueryState
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion_batch

logger = get_logger()


class GraphCompletionContextExtensionRetriever(GraphCompletionRetriever):
    """
    Handles graph context completion for question answering tasks, extending context based
    on retrieved triplets.
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        context_extension_rounds: int = 4,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            system_prompt=system_prompt,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            session_id=session_id,
            response_model=response_model,
        )
        self.context_extension_rounds = context_extension_rounds

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Union[List[Edge], List[List[Edge]]]:
        """
        Extends the context for a given query by retrieving related triplets and generating new
        completions based on them.

        The method runs for a specified number of rounds to enhance results until no new
        triplets are found or the maximum rounds are reached. It retrieves triplet suggestions
        based on a generated completion from previous iterations, logging the process of context
        extension.

        Parameters:
        -----------
            - query (str): The input query for which the completion is generated.

        Returns:
        --------
            - List[Edge]: A list of retrieved triplet edges relevant to the query.
        """
        validate_retriever_input(query, query_batch, self._use_session_cache())

        # Normalize single query to batch for uniform processing
        effective_batch = [query] if query else query_batch

        triplets_batch = await self.get_triplets(query_batch=effective_batch)
        if not triplets_batch:
            return []

        context_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(t) for t in triplets_batch]
        )
        states = {
            q: QueryState(t, c) for q, t, c in zip(effective_batch, triplets_batch, context_batch)
        }

        for _ in range(self.context_extension_rounds):
            if all(s.done for s in states.values()):
                logger.info("Context extension: all queries converged; stopping early.")
                break
            await self._run_extension_round(states)

        return self._collect_triplets(states, query, effective_batch)

    # -- Extension round logic --

    async def _run_extension_round(self, states: dict):
        """Run one extension round: generate completions, fetch new triplets, check convergence."""
        active_queries = [q for q, s in states.items() if not s.done]
        active_contexts = [states[q].context_text for q in active_queries]
        prev_sizes = [len(states[q].triplets) for q in active_queries]

        # Use current completions as new search queries
        completions = await generate_completion_batch(
            query_batch=active_queries,
            context=active_contexts,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
        )

        new_triplets_batch = await self.get_triplets(query_batch=list(completions))
        for q, new_triplets in zip(active_queries, new_triplets_batch):
            states[q].merge_triplets(new_triplets)

        context_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(states[q].triplets) for q in active_queries]
        )
        for q, context, prev_size in zip(active_queries, context_batch, prev_sizes):
            states[q].context_text = context
            states[q].check_convergence(prev_size)

        sizes = [len(states[q].triplets) for q in active_queries]
        logger.info(f"Context extension: unique triplets per query: {sizes}")

    @staticmethod
    def _collect_triplets(
        states: dict, query: Optional[str], query_batch: List[str]
    ) -> Union[List[Edge], List[List[Edge]]]:
        """Extract final triplet lists from states."""
        if query:
            return states[query].triplets
        return [states[q].triplets for q in query_batch]
