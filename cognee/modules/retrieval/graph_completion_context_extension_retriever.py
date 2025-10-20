import asyncio
from typing import Optional, List, Type
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
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            system_prompt=system_prompt,
        )

    async def get_completion(
        self,
        query: str,
        context: Optional[List[Edge]] = None,
        session_id: Optional[str] = None,
        context_extension_rounds=4,
    ) -> List[str]:
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

        Returns:
        --------

            - List[str]: A list containing the generated answer based on the query and the
              extended context.
        """
        triplets = context

        if triplets is None:
            triplets = await self.get_context(query)

        context_text = await self.resolve_edges_to_text(triplets)

        round_idx = 1

        while round_idx <= context_extension_rounds:
            prev_size = len(triplets)

            logger.info(
                f"Context extension: round {round_idx} - generating next graph locational query."
            )
            completion = await generate_completion(
                query=query,
                context=context_text,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
            )

            triplets += await self.get_context(completion)
            triplets = list(set(triplets))
            context_text = await self.resolve_edges_to_text(triplets)

            num_triplets = len(triplets)

            if num_triplets == prev_size:
                logger.info(
                    f"Context extension: round {round_idx} – no new triplets found; stopping early."
                )
                break

            logger.info(
                f"Context extension: round {round_idx} - "
                f"number of unique retrieved triplets: {num_triplets}"
            )

            round_idx += 1

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
                ),
            )
        else:
            completion = await generate_completion(
                query=query,
                context=context_text,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
            )

        if self.save_interaction and context_text and triplets and completion:
            await self.save_qa(
                question=query, answer=completion, context=context_text, triplets=triplets
            )

        if session_save:
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=session_id,
            )

        return [completion]
