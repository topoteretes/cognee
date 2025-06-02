from typing import Any, Optional, List, Type
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

logger = get_logger()


class GraphCompletionCotRetriever(GraphCompletionRetriever):
    """
    Handles graph completion by generating responses based on a series of interactions with
    a language model. This class extends from GraphCompletionRetriever and is designed to
    manage the retrieval and validation process for user queries, integrating follow-up
    questions based on reasoning. The public methods are:

    - get_completion

    Instance variables include:
    - validation_system_prompt_path
    - validation_user_prompt_path
    - followup_system_prompt_path
    - followup_user_prompt_path
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        validation_user_prompt_path: str = "cot_validation_user_prompt.txt",
        validation_system_prompt_path: str = "cot_validation_system_prompt.txt",
        followup_system_prompt_path: str = "cot_followup_system_prompt.txt",
        followup_user_prompt_path: str = "cot_followup_user_prompt.txt",
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
        self.validation_system_prompt_path = validation_system_prompt_path
        self.validation_user_prompt_path = validation_user_prompt_path
        self.followup_system_prompt_path = followup_system_prompt_path
        self.followup_user_prompt_path = followup_user_prompt_path

    async def get_completion(
        self, query: str, context: Optional[Any] = None, max_iter=4
    ) -> List[str]:
        """
        Generate completion responses based on a user query and contextual information.

        This method interacts with a language model client to retrieve a structured response,
        using a series of iterations to refine the answers and generate follow-up questions
        based on reasoning derived from previous outputs. It raises exceptions if the context
        retrieval fails or if the model encounters issues in generating outputs.

        Parameters:
        -----------

            - query (str): The user's query to be processed and answered.
            - context (Optional[Any]): Optional context that may assist in answering the query.
              If not provided, it will be fetched based on the query. (default None)
            - max_iter: The maximum number of iterations to refine the answer and generate
              follow-up questions. (default 4)

        Returns:
        --------

            - List[str]: A list containing the generated answer to the user's query.
        """
        llm_client = get_llm_client()
        followup_question = ""
        triplets = []
        answer = [""]

        for round_idx in range(max_iter + 1):
            if round_idx == 0:
                if context is None:
                    context = await self.get_context(query)
            else:
                triplets += await self.get_triplets(followup_question)
                context = await self.resolve_edges_to_text(list(set(triplets)))

            answer = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )
            logger.info(f"Chain-of-thought: round {round_idx} - answer: {answer}")
            if round_idx < max_iter:
                valid_args = {"query": query, "answer": answer, "context": context}
                valid_user_prompt = render_prompt(
                    filename=self.validation_user_prompt_path, context=valid_args
                )
                valid_system_prompt = read_query_prompt(
                    prompt_file_name=self.validation_system_prompt_path
                )

                reasoning = await llm_client.acreate_structured_output(
                    text_input=valid_user_prompt,
                    system_prompt=valid_system_prompt,
                    response_model=str,
                )
                followup_args = {"query": query, "answer": answer, "reasoning": reasoning}
                followup_prompt = render_prompt(
                    filename=self.followup_user_prompt_path, context=followup_args
                )
                followup_system = read_query_prompt(
                    prompt_file_name=self.followup_system_prompt_path
                )

                followup_question = await llm_client.acreate_structured_output(
                    text_input=followup_prompt, system_prompt=followup_system, response_model=str
                )
                logger.info(
                    f"Chain-of-thought: round {round_idx} - follow-up question: {followup_question}"
                )

        return [answer]
