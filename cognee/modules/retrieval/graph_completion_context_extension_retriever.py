from typing import Any, Optional, List, Type
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

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
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        )

    async def get_completion(
        self, query: str, context: Optional[Any] = None, context_extension_rounds=4
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
            - context_extension_rounds: The maximum number of rounds to extend the context with
              new triplets before halting. (default 4)

        Returns:
        --------

            - List[str]: A list containing the generated answer based on the query and the
              extended context.
        """
        triplets = []

        if context is None:
            triplets += await self.get_triplets(query)
            context = await self.resolve_edges_to_text(triplets)

        round_idx = 1

        while round_idx <= context_extension_rounds:
            prev_size = len(triplets)

            logger.info(
                f"Context extension: round {round_idx} - generating next graph locational query."
            )
            completion = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )

            triplets += await self.get_triplets(completion)
            triplets = list(set(triplets))
            context = await self.resolve_edges_to_text(triplets)

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

        answer = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )

        return [answer]
