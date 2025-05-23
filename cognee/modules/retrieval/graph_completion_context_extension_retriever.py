from typing import Any, Optional, List, Type
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

logger = get_logger()


class GraphCompletionContextExtensionRetriever(GraphCompletionRetriever):
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
