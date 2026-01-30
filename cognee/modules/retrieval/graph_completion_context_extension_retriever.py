import asyncio
from typing import Optional, List, Type, Any
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.utils.query_state import QueryState
from cognee.modules.retrieval.utils.validate_queries import validate_queries
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion, summarize_text
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig

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
        save_interaction: bool = False,
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
            save_interaction=save_interaction,
            system_prompt=system_prompt,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            session_id=session_id,
            response_model=response_model,
        )

        # context_extension_rounds: The maximum number of rounds to extend the context with
        # new triplets before halting. (default 4)
        self.context_extension_rounds = context_extension_rounds

    async def get_retrieved_objects(
        self, query: Optional[str], query_batch: Optional[List[str]]
    ) -> List[Edge]:
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

        # Check if we need to generate context summary for caching
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        if query_batch and session_save:
            raise QueryValidationError(
                message="You cannot use batch queries with session saving currently."
            )
        if query_batch and self.save_interaction:
            raise QueryValidationError(
                message="Cannot use batch queries with interaction saving currently."
            )

        is_query_valid, msg = validate_queries(query, query_batch)
        if not is_query_valid:
            raise QueryValidationError(message=msg)

        if query:
            # This is done mostly to avoid duplicating a lot of code unnecessarily
            query_batch = [query]

        triplets_batch = await self.get_triplets(query_batch=query_batch)
        if not triplets_batch:
            return []

        context_text_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(triplets) for triplets in triplets_batch]
        )

        round_idx = 1

        # We store queries as keys and their associated states in this dict.
        # The state is a 3-item object QueryState, which holds triplets, context text,
        # and a boolean marking whether we should continue extending the context for that query.
        finished_queries_states = {}

        for batched_query, batched_triplets, batched_context_text in zip(
            query_batch, triplets_batch, context_text_batch
        ):
            # Populating the dict at the start with initial information.
            finished_queries_states[batched_query] = QueryState(
                batched_triplets, batched_context_text, False
            )

        while round_idx <= self.context_extension_rounds:
            logger.info(
                f"Context extension: round {round_idx} - generating next graph locational query."
            )

            if all(
                batched_query_state.finished_extending_context
                for batched_query_state in finished_queries_states.values()
            ):
                # We stop early only if all queries in the batch have reached their final state
                logger.info(
                    f"Context extension: round {round_idx} – no new triplets found; stopping early."
                )
                break

            relevant_queries = [
                rel_query
                for rel_query in finished_queries_states.keys()
                if not finished_queries_states[rel_query].finished_extending_context
            ]

            prev_sizes = [
                len(finished_queries_states[rel_query].triplets) for rel_query in relevant_queries
            ]

            completions = await asyncio.gather(
                *[
                    generate_completion(
                        query=rel_query,
                        context=finished_queries_states[rel_query].context_text,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                    )
                    for rel_query in relevant_queries
                ],
            )

            # Get new triplets, and merge them with existing ones, filtering out duplicates
            new_triplets_batch = await self.get_triplets(query_batch=completions)
            for rel_query, batched_new_triplets in zip(relevant_queries, new_triplets_batch):
                finished_queries_states[rel_query].triplets = list(
                    dict.fromkeys(
                        finished_queries_states[rel_query].triplets + batched_new_triplets
                    )
                )

            # Resolve new triplets to text
            context_text_batch = await asyncio.gather(
                *[
                    self.resolve_edges_to_text(finished_queries_states[rel_query].triplets)
                    for rel_query in relevant_queries
                ]
            )

            # Update context_texts in query states
            for rel_query, batched_context_text in zip(relevant_queries, context_text_batch):
                finished_queries_states[rel_query].context_text = batched_context_text

            new_sizes = [
                len(finished_queries_states[rel_query].triplets) for rel_query in relevant_queries
            ]

            for rel_query, prev_size, new_size in zip(relevant_queries, prev_sizes, new_sizes):
                # Mark done queries accordingly
                if prev_size == new_size:
                    finished_queries_states[rel_query].finished_extending_context = True

            logger.info(
                f"Context extension: round {round_idx} - "
                f"number of unique retrieved triplets for each query : {new_sizes}"
            )

            round_idx += 1

        return [query_state.triplets for query_state in finished_queries_states.values()]

    async def get_completion_from_context(
        self,
        query: Optional[str],
        query_batch: Optional[List[str]],
        retrieved_objects: List[Edge] | List[List[Edge]],
        context: str | List[str],
    ) -> List[Any]:
        """
        Returns a human readable answer based on the provided query and extended context derived from the retrieved objects.

        Returns:
        --------

            - List[str]: A list containing the generated answer based on the query and the
              extended context.
        """

        # Check if we need to generate context summary for caching
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        if session_save:
            conversation_history = await get_conversation_history(session_id=self.session_id)

            context_summary, completion = await asyncio.gather(
                summarize_text(context),
                generate_completion(
                    query=query,
                    context=context,
                    user_prompt_path=self.user_prompt_path,
                    system_prompt_path=self.system_prompt_path,
                    system_prompt=self.system_prompt,
                    conversation_history=conversation_history,
                    response_model=self.response_model,
                ),
            )
        else:
            if query_batch:
                completion = await asyncio.gather(
                    *[
                        generate_completion(
                            query=batched_query,
                            context=batched_context,
                            user_prompt_path=self.user_prompt_path,
                            system_prompt_path=self.system_prompt_path,
                            system_prompt=self.system_prompt,
                            response_model=self.response_model,
                        )
                        for batched_query, batched_context in zip(query_batch, context)
                    ]
                )
            else:
                completion = await generate_completion(
                    query=query,
                    context=context,
                    user_prompt_path=self.user_prompt_path,
                    system_prompt_path=self.system_prompt_path,
                    system_prompt=self.system_prompt,
                    response_model=self.response_model,
                )

        if self.save_interaction and context and retrieved_objects and completion:
            await self.save_qa(
                question=query, answer=completion, context=context, triplets=retrieved_objects
            )

        if session_save:
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=self.session_id,
            )

        return completion if query_batch else [completion]
