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
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig

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
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        save_interaction: bool = False,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
        )
        self.validation_system_prompt_path = validation_system_prompt_path
        self.validation_user_prompt_path = validation_user_prompt_path
        self.followup_system_prompt_path = followup_system_prompt_path
        self.followup_user_prompt_path = followup_user_prompt_path

    async def get_completion(
        self,
        query: str,
        context: Optional[List[Edge]] = None,
        session_id: Optional[str] = None,
        max_iter=4,
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
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)
            - max_iter: The maximum number of iterations to refine the answer and generate
              follow-up questions. (default 4)

        Returns:
        --------

            - List[str]: A list containing the generated answer to the user's query.
        """
        followup_question = ""
        triplets = []
        completion = ""

        # Retrieve conversation history if session saving is enabled
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        conversation_history = ""
        if session_save:
            conversation_history = await get_conversation_history(session_id=session_id)

        for round_idx in range(max_iter + 1):
            if round_idx == 0:
                if context is None:
                    triplets = await self.get_context(query)
                    context_text = await self.resolve_edges_to_text(triplets)
                else:
                    context_text = await self.resolve_edges_to_text(context)
            else:
                triplets += await self.get_context(followup_question)
                context_text = await self.resolve_edges_to_text(list(set(triplets)))

            completion = await generate_completion(
                query=query,
                context=context_text,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                conversation_history=conversation_history if session_save else None,
            )
            logger.info(f"Chain-of-thought: round {round_idx} - answer: {completion}")
            if round_idx < max_iter:
                valid_args = {"query": query, "answer": completion, "context": context_text}
                valid_user_prompt = render_prompt(
                    filename=self.validation_user_prompt_path, context=valid_args
                )
                valid_system_prompt = read_query_prompt(
                    prompt_file_name=self.validation_system_prompt_path
                )

                reasoning = await LLMGateway.acreate_structured_output(
                    text_input=valid_user_prompt,
                    system_prompt=valid_system_prompt,
                    response_model=str,
                )
                followup_args = {"query": query, "answer": completion, "reasoning": reasoning}
                followup_prompt = render_prompt(
                    filename=self.followup_user_prompt_path, context=followup_args
                )
                followup_system = read_query_prompt(
                    prompt_file_name=self.followup_system_prompt_path
                )

                followup_question = await LLMGateway.acreate_structured_output(
                    text_input=followup_prompt, system_prompt=followup_system, response_model=str
                )
                logger.info(
                    f"Chain-of-thought: round {round_idx} - follow-up question: {followup_question}"
                )

        if self.save_interaction and context and triplets and completion:
            await self.save_qa(
                question=query, answer=completion, context=context_text, triplets=triplets
            )

        # Save to session cache
        if session_save:
            context_summary = await summarize_text(context_text)
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=session_id,
            )

        return [completion]
