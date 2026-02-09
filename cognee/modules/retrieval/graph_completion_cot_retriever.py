import asyncio
import json
from typing import Optional, List, Type, Any, Union

from pydantic import BaseModel

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.query_state import QueryState
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import (
    batch_llm_completion,
    generate_completion_batch,
    summarize_text,
)
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.exceptions.exceptions import CogneeValidationError

logger = get_logger()


def _as_answer_text(completion: Any) -> str:
    """Convert completion to human-readable text for validation and follow-up prompts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, BaseModel):
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

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Union[List[Edge], List[List[Edge]]]:
        """
        Run chain-of-thought completion with optional structured output.

        Parameters:
        -----------
            - query: User query

        Returns:
        --------
            - List of retrieved edges
        """
        session_save = self._use_session_cache()
        validate_retriever_input(query, query_batch, session_save)

        conversation_history = ""
        if session_save:
            conversation_history = await get_conversation_history(session_id=self.session_id)

        # Normalize single query to batch for uniform processing
        effective_batch = [query] if query else query_batch

        completion, context_text, triplets = await self._run_cot_completion(
            effective_batch, conversation_history
        )

        # Store completion to avoid re-calling LLM in get_completion_from_context
        self.completion = completion

        if session_save:
            context_summary = await summarize_text(context_text[0])
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=str(completion[0]),
                session_id=self.session_id,
            )

        if query:
            return triplets[0]
        return triplets

    # -- CoT orchestrator --

    async def _run_cot_completion(
        self, query_batch: List[str], conversation_history: str = ""
    ) -> tuple[List[Any], List[str], List[List[Edge]]]:
        """
        Run chain-of-thought completion with optional structured output.

        Parameters:
        -----------
            - query_batch: Batch of user queries
            - conversation_history: Optional conversation history string

        Returns:
        --------
            - completion_result: The generated completion (string or structured model)
            - context_text: The resolved context text
            - triplets: The list of triplets used
        """
        states = {q: QueryState() for q in query_batch}
        await self._fetch_initial_triplets_and_context(states)
        await self._generate_completions(states, conversation_history)

        for _ in range(self.max_iter):
            followup_queries = await self._run_cot_round(states)
            await self._merge_followup_triplets(states, followup_queries)
            await self._generate_completions(states, conversation_history)

        return self._collect_results(states, query_batch)

    # -- Helper methods called by the orchestrator --

    async def _fetch_initial_triplets_and_context(self, states: dict):
        """Fetch triplets and resolve context text for all queries."""
        queries = list(states.keys())
        triplets_batch = await self.get_triplets(query_batch=queries)
        context_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(t) for t in triplets_batch]
        )
        for q, triplets, context in zip(queries, triplets_batch, context_batch):
            states[q].triplets = triplets
            states[q].context_text = context

    async def _generate_completions(self, states: dict, conversation_history: str):
        """Generate completions for all queries in parallel."""
        queries = list(states.keys())
        contexts = [states[q].context_text for q in queries]
        completions = await generate_completion_batch(
            query_batch=queries,
            context=contexts,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
            conversation_history=conversation_history if conversation_history else None,
        )
        for q, comp in zip(queries, completions):
            states[q].completion = comp
        logger.info(f"Chain-of-thought: generated completions for {len(queries)} queries")

    async def _run_cot_round(self, states: dict) -> List[str]:
        """Run one CoT round: validate answers, generate follow-up questions."""
        validation_prompts, validation_system = self._build_validation_prompts(states)
        reasoning_batch = await batch_llm_completion(validation_prompts, validation_system)

        followup_prompts, followup_system = self._build_followup_prompts(states, reasoning_batch)
        followup_questions = await batch_llm_completion(followup_prompts, followup_system)

        logger.info(f"Chain-of-thought: follow-up questions: {followup_questions}")
        return followup_questions

    # -- Prompt builders --

    def _build_cot_prompts(self, template_path, states, extras):
        """Build prompts with common query+answer fields and per-query extras."""
        return [
            render_prompt(
                filename=template_path,
                context={"query": q, "answer": _as_answer_text(states[q].completion), **extra},
            )
            for q, extra in zip(states.keys(), extras)
        ]

    def _build_validation_prompts(self, states):
        """Build validation user prompts and load system prompt."""
        system_prompt = read_query_prompt(prompt_file_name=self.validation_system_prompt_path)
        user_prompts = self._build_cot_prompts(
            self.validation_user_prompt_path,
            states,
            [{"context": s.context_text} for s in states.values()],
        )
        return user_prompts, system_prompt

    def _build_followup_prompts(self, states, reasoning_batch):
        """Build followup user prompts and load system prompt."""
        system_prompt = read_query_prompt(prompt_file_name=self.followup_system_prompt_path)
        user_prompts = self._build_cot_prompts(
            self.followup_user_prompt_path,
            states,
            [{"reasoning": r} for r in reasoning_batch],
        )
        return user_prompts, system_prompt

    async def _merge_followup_triplets(self, states: dict, followup_questions: List[str]):
        """Fetch triplets for follow-up questions and merge with existing state."""
        queries = list(states.keys())
        new_triplets_batch = await self.get_triplets(query_batch=followup_questions)

        for q, new_triplets in zip(queries, new_triplets_batch):
            states[q].merge_triplets(new_triplets)

        context_batch = await asyncio.gather(
            *[self.resolve_edges_to_text(states[q].triplets) for q in queries]
        )
        for q, context in zip(queries, context_batch):
            states[q].context_text = context

    def _collect_results(
        self, states: dict, query_batch: List[str]
    ) -> tuple[List[Any], List[str], List[List[Edge]]]:
        """Extract final completions, context texts, and triplets from states."""
        completions = [states[q].completion for q in query_batch]
        contexts = [states[q].context_text for q in query_batch]
        triplets = [states[q].triplets for q in query_batch]
        return completions, contexts, triplets

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: List[Edge] | List[List[Edge]] = None,
        context: str | List[str] = None,
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
            - query_batch (list[str]): The list of queries to be processed and answered.
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
        if not retrieved_objects or (
            query_batch and all(len(triplet) == 0 for triplet in retrieved_objects)
        ):
            raise CogneeValidationError("No context retrieved to generate completion.")
        return self.completion
