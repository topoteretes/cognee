import asyncio
from typing import Optional, List, Type, Any
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
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
        triplets = context

        if query:
            # This is done mostly to avoid duplicating a lot of code unnecessarily
            query_batch = [query]
            if triplets:
                triplets = [triplets]

        if triplets is None:
            triplets = await self.get_context(query_batch=query_batch)

        context_text = ""
        context_texts = await asyncio.gather(
            *[self.resolve_edges_to_text(triplets_element) for triplets_element in triplets]
        )

        round_idx = 1

        # We will be removing queries, and their associated triplets and context, as we go
        # through iterations, so we need to save their final states for the final generation.
        # Final state is stored in the finished_queries_data dict, and we populate it at the start as well.
        original_query_batch = query_batch
        finished_queries_data = {}
        for i, query in enumerate(query_batch):
            finished_queries_data[query] = (triplets[i], context_texts[i])

        while round_idx <= context_extension_rounds:
            logger.info(
                f"Context extension: round {round_idx} - generating next graph locational query."
            )

            # Filter out the queries that cannot be extended further, and their associated contexts
            query_batch = [query for query in query_batch if query]
            triplets = [triplet_element for triplet_element in triplets if triplet_element]
            context_texts = [context_text for context_text in context_texts if context_text]
            if len(query_batch) == 0:
                logger.info(
                    f"Context extension: round {round_idx} – no new triplets found; stopping early."
                )
                break

            prev_sizes = [len(triplets_element) for triplets_element in triplets]

            completions = await asyncio.gather(
                *[
                    generate_completion(
                        query=query,
                        context=context,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                    )
                    for query, context in zip(query_batch, context_texts)
                ],
            )

            # Get new triplets, and merge them with existing ones, filtering out duplicates
            new_triplets = await self.get_context(query_batch=completions)
            for i, (triplets_element, new_triplets_element) in enumerate(
                zip(triplets, new_triplets)
            ):
                triplets_element += new_triplets_element
                triplets[i] = list(dict.fromkeys(triplets_element))

            context_texts = await asyncio.gather(
                *[self.resolve_edges_to_text(triplets_element) for triplets_element in triplets]
            )

            new_sizes = [len(triplets_element) for triplets_element in triplets]

            for i, (query, prev_size, new_size, triplets_element, context_text) in enumerate(
                zip(query_batch, prev_sizes, new_sizes, triplets, context_texts)
            ):
                finished_queries_data[query] = (triplets_element, context_text)
                if prev_size == new_size:
                    # In this case, we can stop trying to extend the context of this query
                    query_batch[i] = ""
                    triplets[i] = []
                    context_texts[i] = ""

            logger.info(
                f"Context extension: round {round_idx} - "
                f"number of unique retrieved triplets for each query : {new_sizes}"
            )

            round_idx += 1

        # Reset variables for the final generations. They contain the final state
        # of triplets and contexts for each query, after all extension iterations.
        query_batch = original_query_batch
        triplets = []
        context_texts = []
        for query in query_batch:
            triplets.append(finished_queries_data[query][0])
            context_texts.append(finished_queries_data[query][1])

        # Check if we need to generate context summary for caching
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

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
            completion = await asyncio.gather(
                *[
                    generate_completion(
                        query=query,
                        context=context_text,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                        response_model=response_model,
                    )
                    for query, context_text in zip(query_batch, context_texts)
                ],
            )

        # TODO: Do batch queries for save interaction
        if self.save_interaction and context_texts and triplets and completion:
            await self.save_qa(
                question=query, answer=completion[0], context=context_texts[0], triplets=triplets[0]
            )

        if session_save:
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=session_id,
            )

        return completion if isinstance(completion, list) else [completion]
