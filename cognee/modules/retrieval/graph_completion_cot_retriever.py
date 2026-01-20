import asyncio
import json
from typing import Optional, List, Type, Any
from pydantic import BaseModel
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.validate_queries import validate_queries
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

logger = get_logger()


class QueryState:
    """
    Helper class containing all necessary information about the query state.
    Used to keep track of important information in a more readable way, and
    enable as many parallel calls to llms as possible.
    """

    def __init__(self):
        self.completion: str = ""
        self.triplets: List[Edge] = []
        self.context_text: str = ""

        self.answer_text: str = ""
        self.valid_user_prompt: str = ""
        self.valid_system_prompt: str = ""
        self.reasoning: str = ""

        self.followup_question: str = ""
        self.followup_prompt: str = ""
        self.followup_system: str = ""


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
        )
        self.validation_system_prompt_path = validation_system_prompt_path
        self.validation_user_prompt_path = validation_user_prompt_path
        self.followup_system_prompt_path = followup_system_prompt_path
        self.followup_user_prompt_path = followup_user_prompt_path

    async def _run_cot_completion(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        context: Optional[List[Edge] | List[List[Edge]]] = None,
        conversation_history: str = "",
        max_iter: int = 4,
        response_model: Type = str,
    ) -> tuple[List[Any], List[str], List[List[Edge]]]:
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
        followup_question_batch = []
        completion_batch = []
        context_text_batch = []

        if query:
            # Treat a single query as a batch of queries, mainly avoiding massive code duplication
            query_batch = [query]
            if context:
                context = [context]

        triplets_batch = context

        # dict containing query -> QueryState key-value pairs
        # For every query, we save necessary data so we can execute requests in parallel
        query_state_tracker = {}
        for batched_query in query_batch:
            query_state_tracker[batched_query] = QueryState()

        for round_idx in range(max_iter + 1):
            if round_idx == 0:
                if context is None:
                    # Get context, resolve to text, and store info in the query state
                    triplets_batch = await self.get_context(query_batch=query_batch)
                    context_text_batch = await asyncio.gather(
                        *[
                            self.resolve_edges_to_text(batched_triplets)
                            for batched_triplets in triplets_batch
                        ]
                    )
                    for batched_query, batched_triplets, batched_context_text in zip(
                        query_batch, triplets_batch, context_text_batch
                    ):
                        query_state_tracker[batched_query].triplets = batched_triplets
                        query_state_tracker[batched_query].context_text = batched_context_text
                else:
                    # In this case just resolve to text and save to the query state
                    context_text_batch = await asyncio.gather(
                        *[
                            self.resolve_edges_to_text(batched_context)
                            for batched_context in context
                        ]
                    )
                    for batched_query, batched_triplets, batched_context_text in zip(
                        query_batch, context, context_text_batch
                    ):
                        query_state_tracker[batched_query].triplets = batched_triplets
                        query_state_tracker[batched_query].context_text = batched_context_text
            else:
                # Find new triplets, and update existing query states
                triplets_batch = await self.get_context(query_batch=followup_question_batch)

                for batched_query, batched_followup_triplets in zip(query_batch, triplets_batch):
                    query_state_tracker[batched_query].triplets = list(
                        dict.fromkeys(
                            query_state_tracker[batched_query].triplets + batched_followup_triplets
                        )
                    )

                context_text_batch = await asyncio.gather(
                    *[
                        self.resolve_edges_to_text(batched_query_state.triplets)
                        for batched_query_state in query_state_tracker.values()
                    ]
                )

                for batched_query, batched_context_text in zip(query_batch, context_text_batch):
                    query_state_tracker[batched_query].context_text = batched_context_text

            completion_batch = await asyncio.gather(
                *[
                    generate_completion(
                        query=batched_query,
                        context=batched_query_state.context_text,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        system_prompt=self.system_prompt,
                        conversation_history=conversation_history if conversation_history else None,
                        response_model=response_model,
                    )
                    for batched_query, batched_query_state in query_state_tracker.items()
                ]
            )

            for batched_query, batched_completion in zip(query_batch, completion_batch):
                query_state_tracker[batched_query].completion = batched_completion

            logger.info(f"Chain-of-thought: round {round_idx} - answers: {completion_batch}")

            if round_idx < max_iter:
                for batched_query, batched_query_state in query_state_tracker.items():
                    batched_query_state.answer_text = _as_answer_text(
                        batched_query_state.completion
                    )
                    valid_args = {
                        "query": batched_query,
                        "answer": batched_query_state.answer_text,
                        "context": batched_query_state.context_text,
                    }
                    batched_query_state.valid_user_prompt = render_prompt(
                        filename=self.validation_user_prompt_path,
                        context=valid_args,
                    )
                    batched_query_state.valid_system_prompt = read_query_prompt(
                        prompt_file_name=self.validation_system_prompt_path
                    )

                reasoning_batch = await asyncio.gather(
                    *[
                        LLMGateway.acreate_structured_output(
                            text_input=batched_query_state.valid_user_prompt,
                            system_prompt=batched_query_state.valid_system_prompt,
                            response_model=str,
                        )
                        for batched_query_state in query_state_tracker.values()
                    ]
                )

                for batched_query, batched_reasoning in zip(query_batch, reasoning_batch):
                    query_state_tracker[batched_query].reasoning = batched_reasoning

                for batched_query, batched_query_state in query_state_tracker.items():
                    followup_args = {
                        "query": batched_query,
                        "answer": batched_query_state.answer_text,
                        "reasoning": batched_query_state.reasoning,
                    }
                    batched_query_state.followup_prompt = render_prompt(
                        filename=self.followup_user_prompt_path,
                        context=followup_args,
                    )
                    batched_query_state.followup_system = read_query_prompt(
                        prompt_file_name=self.followup_system_prompt_path
                    )

                followup_question_batch = await asyncio.gather(
                    *[
                        LLMGateway.acreate_structured_output(
                            text_input=batched_query_state.followup_prompt,
                            system_prompt=batched_query_state.followup_system,
                            response_model=str,
                        )
                        for batched_query_state in query_state_tracker.values()
                    ]
                )

                for batched_query, batched_followup_question in zip(
                    query_batch, followup_question_batch
                ):
                    query_state_tracker[batched_query].followup_question = batched_followup_question

                logger.info(
                    f"Chain-of-thought: round {round_idx} - follow-up questions: {followup_question_batch}"
                )

        return completion_batch, context_text_batch, triplets_batch

    async def get_completion(
        self,
        query: Optional[str] = None,
        context: Optional[List[Edge] | List[List[Edge]]] = None,
        session_id: Optional[str] = None,
        max_iter=4,
        response_model: Type = str,
        query_batch: Optional[List[str]] = None,
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

        # Check if session saving is enabled
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        if query_batch and session_save:
            raise ValueError("You cannot use batch queries with session saving currently.")
        if query_batch and self.save_interaction:
            raise ValueError("Cannot use batch queries with interaction saving currently.")

        is_query_valid, msg = validate_queries(query, query_batch)
        if not is_query_valid:
            raise ValueError(msg)

        # Load conversation history if enabled
        conversation_history = ""
        if session_save:
            conversation_history = await get_conversation_history(session_id=session_id)

        completion, context_text, triplets = await self._run_cot_completion(
            query=query,
            query_batch=query_batch,
            context=context,
            conversation_history=conversation_history,
            max_iter=max_iter,
            response_model=response_model,
        )

        # TODO: Handle save interaction for batch queries
        if self.save_interaction and context and triplets and completion:
            await self.save_qa(
                question=query,
                answer=str(completion[0]),
                context=context_text[0],
                triplets=triplets[0],
            )

        # TODO: Handle session save interaction for batch queries
        # Save to session cache if enabled
        if session_save:
            context_summary = await summarize_text(context_text)
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=str(completion),
                session_id=session_id,
            )

        return completion
