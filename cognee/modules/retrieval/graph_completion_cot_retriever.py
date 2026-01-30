import asyncio
import json
from typing import Optional, List, Type, Any
from pydantic import BaseModel
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.shared.logging_utils import get_logger

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import (
    generate_completion,
    summarize_text,
)
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.exceptions.exceptions import CogneeValidationError

logger = get_logger()


def _as_answer_text(completion: Any) -> str:
    """Convert completion to human-readable text for validation and follow-up prompts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, BaseModel):
        # Add notice that this is a structured response
        json_str = completion.model_dump_json(indent=2)
        return f"[Structured Response]\n{json_str}"
    try:
        return json.dumps(completion, indent=2)
    except TypeError:
        return str(completion)


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
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        max_iter: int = 4,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            session_id=session_id,
            response_model=response_model,
        )
        self.validation_system_prompt_path = validation_system_prompt_path
        self.validation_user_prompt_path = validation_user_prompt_path
        self.followup_system_prompt_path = followup_system_prompt_path
        self.followup_user_prompt_path = followup_user_prompt_path
        self.completion = []
        self.max_iter = max_iter

    async def get_retrieved_objects(self, query: str) -> List[Edge]:
        """
        Run chain-of-thought completion with optional structured output.

        Parameters:
        -----------
            - query: User query

        Returns:
        --------
            - List of retrieved edges
        """
        # Check if session saving is enabled
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        # Load conversation history if enabled
        conversation_history = ""
        if session_save:
            conversation_history = await get_conversation_history(session_id=self.session_id)

        completion, context_text, triplets = await self._run_cot_completion(
            query=query,
            conversation_history=conversation_history,
        )

        # Note: completion info is stored to reduce the need to call LLM again in get_completion_from_context
        self.completion = completion

        if self.save_interaction and context_text and triplets and completion:
            await self.save_qa(
                question=query, answer=str(completion), context=context_text, triplets=triplets
            )

        # Save to session cache if enabled
        if session_save:
            context_summary = await summarize_text(context_text)
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=str(completion),
                session_id=self.session_id,
            )

        return triplets

    async def _run_cot_completion(
        self,
        query: str,
        conversation_history: str = "",
    ) -> tuple[Any, str, List[Edge]]:
        """
        Run chain-of-thought completion with optional structured output.

        Parameters:
        -----------
            - query: User query
            - context: Optional pre-fetched context edges
            - conversation_history: Optional conversation history string
            - max_iter: Maximum CoT iterations
            - response_model: Type for structured output (str for plain text)

        Returns:
        --------
            - completion_result: The generated completion (string or structured model)
            - context_text: The resolved context text
            - triplets: The list of triplets used
        """
        followup_question = ""
        triplets = []
        completion = ""

        for round_idx in range(self.max_iter + 1):
            if round_idx == 0:
                triplets = await self.get_triplets(query)
                context_text = await self.resolve_edges_to_text(triplets)
            else:
                triplets += await self.get_triplets(followup_question)
                context_text = await self.resolve_edges_to_text(list(set(triplets)))

            completion = await generate_completion(
                query=query,
                context=context_text,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                conversation_history=conversation_history if conversation_history else None,
                response_model=self.response_model,
            )

            logger.info(f"Chain-of-thought: round {round_idx} - answer: {completion}")

            if round_idx < self.max_iter:
                answer_text = _as_answer_text(completion)
                valid_args = {"query": query, "answer": answer_text, "context": context_text}
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
                followup_args = {"query": query, "answer": answer_text, "reasoning": reasoning}
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

        return completion, context_text, triplets

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: List[Edge],
        context: str,
    ) -> List[Any]:
        """
        Generate completion responses based on a user query and contextual information.

        This method interacts with a language model client to retrieve a structured response,
        using a series of iterations to refine the answers and generate follow-up questions
        based on reasoning derived from previous outputs. It raises exceptions if the context
        retrieval fails or if the model encounters issues in generating outputs. It returns
        structured output using the provided response model.

        Parameters:
        -----------

            - query (str): The user's query to be processed and answered.
            - context (Optional[Any]): Optional context that may assist in answering the query.
              If not provided, it will be fetched based on the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)
            - max_iter: The maximum number of iterations to refine the answer and generate
              follow-up questions. (default 4)
            - response_model (Type): The Pydantic model type for structured output. (default str)

        Returns:
        --------

            - List[str]: A list containing the generated answer to the user's query.
        """
        if not retrieved_objects:
            raise CogneeValidationError("No context retrieved to generate completion.")
        completion = self.completion
        return [completion]
