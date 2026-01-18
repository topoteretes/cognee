import asyncio
from typing import Optional, List, Type, Any
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
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

    Public methods:
    - get_completion

    Instance variables:
    - user_prompt_path
    - system_prompt_path
    - top_k
    - node_type
    - node_name
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
        )

    async def get_completion(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        context: Optional[List[Edge] | List[List[Edge]]] = None,
        session_id: Optional[str] = None,
        context_extension_rounds=4,
        response_model: Type = str,
    ) -> List[Any]:
        """
        Extends the context for a given query by retrieving related triplets and generating new
        completions based on them.

        The method runs for a specified number of rounds to enhance context until no new
        triplets are found or the maximum rounds are reached. It retrieves triplet suggestions
        based on a generated completion from previous iterations, logging the process of context
        extension.

        Parameters:
        -----------

            - query (str): The input query for which the completion is generated.
            - context (Optional[Any]): The existing context to use for enhancing the query; if
              None, it will be initialized from triplets generated for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)
            - context_extension_rounds: The maximum number of rounds to extend the context with
              new triplets before halting. (default 4)
            - response_model (Type): The Pydantic model type for structured output. (default str)

        Returns:
        --------

            - List[str]: A list containing the generated answer based on the query and the
              extended context.
        """
        # TODO: This may be unnecessary in this retriever, will check later
        query_validation = validate_queries(query, query_batch)
        if not query_validation[0]:
            raise ValueError(query_validation[1])

        triplets_batch = context

        if query:
            # This is done mostly to avoid duplicating a lot of code unnecessarily
            query_batch = [query]
            if triplets_batch:
                triplets_batch = [triplets_batch]

        if triplets_batch is None:
            triplets_batch = await self.get_context(query_batch=query_batch)

        context_text = ""
        context_text_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(triplets) for triplets in triplets_batch]
        )

        round_idx = 1

        # We will be removing queries, and their associated triplets and context, as we go
        # through iterations, so we need to save their final states for the final generation.
        # Final state is stored in the finished_queries_data dict, and we populate it at the start as well.
        original_query_batch = query_batch
        finished_queries_data = {}
        for i, query in enumerate(query_batch):
            finished_queries_data[query] = (triplets_batch[i], context_text_batch[i])

        while round_idx <= context_extension_rounds:
            logger.info(
                f"Context extension: round {round_idx} - generating next graph locational query."
            )

            # Filter out the queries that cannot be extended further, and their associated contexts
            query_batch = [query for query in query_batch if query]
            triplets_batch = [triplets for triplets in triplets_batch if triplets]
            context_text_batch = [
                context_text for context_text in context_text_batch if context_text
            ]
            if len(query_batch) == 0:
                logger.info(
                    f"Context extension: round {round_idx} – no new triplets found; stopping early."
                )
                break

            prev_sizes = [len(triplets) for triplets in triplets_batch]

            completions = await asyncio.gather(
                *[
                    generate_completion(
                        query=query,
                        context=context,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                    )
                    for query, context in zip(query_batch, context_text_batch)
                ],
            )

            # Get new triplets, and merge them with existing ones, filtering out duplicates
            new_triplets_batch = await self.get_context(query_batch=completions)
            for i, (triplets, new_triplets) in enumerate(zip(triplets_batch, new_triplets_batch)):
                triplets += new_triplets
                triplets_batch[i] = list(dict.fromkeys(triplets))

            context_text_batch = await asyncio.gather(
                *[self.resolve_edges_to_text(triplets) for triplets in triplets_batch]
            )

            new_sizes = [len(triplets) for triplets in triplets_batch]

            for i, (batched_query, prev_size, new_size, triplets, context_text) in enumerate(
                zip(query_batch, prev_sizes, new_sizes, triplets_batch, context_text_batch)
            ):
                finished_queries_data[query] = (triplets, context_text)
                if prev_size == new_size:
                    # In this case, we can stop trying to extend the context of this query
                    query_batch[i] = ""
                    triplets_batch[i] = []
                    context_text_batch[i] = ""

            logger.info(
                f"Context extension: round {round_idx} - "
                f"number of unique retrieved triplets for each query : {new_sizes}"
            )

            round_idx += 1

        # Reset variables for the final generations. They contain the final state
        # of triplets and contexts for each query, after all extension iterations.
        query_batch = original_query_batch
        triplets_batch = []
        context_text_batch = []
        for batched_query in query_batch:
            triplets_batch.append(finished_queries_data[batched_query][0])
            context_text_batch.append(finished_queries_data[batched_query][1])

        # Check if we need to generate context summary for caching
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        completion_batch = []

        if session_save:
            conversation_history = await get_conversation_history(session_id=session_id)

            context_summary, completion = await asyncio.gather(
                summarize_text(context_text),
                generate_completion(
                    query=query,
                    context=context_text,
                    user_prompt_path=self.user_prompt_path,
                    system_prompt_path=self.system_prompt_path,
                    system_prompt=self.system_prompt,
                    conversation_history=conversation_history,
                    response_model=response_model,
                ),
            )
        else:
            completion_batch = await asyncio.gather(
                *[
                    generate_completion(
                        query=batched_query,
                        context=batched_context_text,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                        response_model=response_model,
                    )
                    for batched_query, batched_context_text in zip(query_batch, context_text_batch)
                ],
            )

        # TODO: Do batch queries for save interaction
        if self.save_interaction and context_text_batch and triplets_batch and completion_batch:
            await self.save_qa(
                question=query,
                answer=completion_batch[0],
                context=context_text_batch[0],
                triplets=triplets_batch[0],
            )

        if session_save:
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=session_id,
            )

        return completion_batch if completion_batch else [completion]
