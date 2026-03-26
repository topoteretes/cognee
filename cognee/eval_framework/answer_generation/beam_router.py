"""BEAM question-type router — routes probing questions to appropriate retrievers.

Each BEAM question type maps to a retriever + system prompt strategy.
The router classifies the question (using pre-labeled types from the dataset)
and delegates to the matching retrieval strategy.
"""

import asyncio
from typing import Any, Dict, List, Optional

from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger()


# Prompt templates per question type — instruct the LLM on HOW to answer
_TYPE_PROMPTS: Dict[str, str] = {
    "information_extraction": (
        "You are answering a factual question about a conversation. "
        "Extract the specific information requested. Be precise and concise. "
        "If the information is not in the context, say so."
    ),
    "temporal_reasoning": (
        "You are answering a question that requires reasoning about time. "
        "The context includes [Session X, Turn Y] markers showing the chronological "
        "order of the conversation. Pay attention to these markers, dates, time "
        "references, and the order of events. Reference specific dates or time "
        "periods in your answer."
    ),
    "multi_session_reasoning": (
        "You are answering a question that requires combining information "
        "from multiple conversation sessions. The context includes [Session X, Turn Y] "
        "markers — use them to identify which session each fact comes from. "
        "Think step by step: first identify the relevant pieces from different sessions, "
        "then synthesize them into a coherent answer."
    ),
    "contradiction_resolution": (
        "You are answering a question about contradictory information in a conversation. "
        "The context includes [Session X, Turn Y] markers showing when each piece of "
        "information was stated. When two facts conflict, the one from a later session "
        "or later turn is the correct/updated version. Identify both statements, note "
        "their positions, and use the most recent one as the answer."
    ),
    "event_ordering": (
        "You are answering a question about the order of events in a conversation. "
        "The context includes [Session X, Turn Y] markers showing chronological position. "
        "Sort events by their session and turn numbers. Reference specific sessions, "
        "turn numbers, or time anchors when available."
    ),
    "knowledge_update": (
        "You are answering a question about updated information. "
        "The context includes [Session X, Turn Y] markers showing when each piece of "
        "information was stated. The conversation may contain both old and new versions "
        "of a fact. Always use the information from the latest session/turn unless "
        "specifically asked about history."
    ),
    "summarization": (
        "You are summarizing part of a conversation. "
        "The context includes [Session X, Turn Y] markers — use them to organize "
        "your summary by session when multiple sessions are involved. "
        "Cover all key points. Be comprehensive but concise."
    ),
    "abstention": (
        "You are answering a question where the correct response may be to abstain. "
        "ONLY answer if the context contains direct, explicit evidence. If the context "
        "does not contain the specific information asked about — even if it contains "
        "related information — respond with: 'This information was not discussed in "
        "the conversation.' Do NOT guess or infer from partial evidence."
    ),
    "preference_following": (
        "You are answering a question about user preferences expressed in the conversation. "
        "The context includes [Session X, Turn Y] markers — preferences may change over time, "
        "so use the most recent preference unless asked about history. Pay attention to "
        "explicitly stated preferences and implicit ones inferred from behavior."
    ),
    "instruction_following": (
        "You are answering a question about instructions given during the conversation. "
        "The context includes [Session X, Turn Y] markers — use them to locate the original "
        "instruction and any follow-up actions. Check whether the instruction was followed "
        "consistently. Reference the original instruction and evidence of compliance."
    ),
}

# Map question types to retriever classes.
# Retriever choice rationale:
#   GraphCompletionRetriever      – direct triplet lookup, best for factual queries
#   GraphCompletionContextExtensionRetriever – iteratively widens evidence window
#   CompletionRetriever           – RAG over raw chunks, better coverage for summaries
_TYPE_RETRIEVERS: Dict[str, type] = {
    # Direct factual lookup
    "information_extraction": GraphCompletionRetriever,
    "preference_following": GraphCompletionRetriever,
    # RAG over raw chunks — raw context better for abstention (clearer what was/wasn't discussed)
    "abstention": CompletionRetriever,
    # RAG over raw chunks — graph loses temporal/event ordering, chunks preserve message flow
    "temporal_reasoning": CompletionRetriever,
    "event_ordering": CompletionRetriever,
    "knowledge_update": GraphCompletionContextExtensionRetriever,
    "multi_session_reasoning": GraphCompletionContextExtensionRetriever,
    "contradiction_resolution": GraphCompletionContextExtensionRetriever,
    "instruction_following": GraphCompletionRetriever,
    # RAG over raw chunks — better detail coverage than graph summaries
    "summarization": CompletionRetriever,
}

# Per-type top_k overrides (default is 10)
_TYPE_TOP_K: Dict[str, int] = {
    "summarization": 35,  # Sessions have ~30 turns, need broad coverage
    "event_ordering": 25,  # Need enough chunks to see full event sequence
    "temporal_reasoning": 20,  # Need enough chunks to find date references
    "abstention": 15,  # Enough context to judge what was/wasn't discussed
    "multi_session_reasoning": 15,  # Evidence spread across multiple sessions
    "knowledge_update": 15,  # Need to see both old and new versions of a fact
    "contradiction_resolution": 15,  # Need to see both conflicting statements
    "information_extraction": 12,  # Slightly more context for precise facts
    "instruction_following": 12,  # Need instruction + evidence of compliance
}

# Per-type context extension rounds (default is 4)
_TYPE_EXTENSION_ROUNDS: Dict[str, int] = {}

_DEFAULT_PROMPT = (
    "You are answering a question about a conversation. "
    "Use the provided context to give an accurate answer."
)


class BEAMRouter:
    """Routes BEAM probing questions to the appropriate retriever and prompt.

    Usage::

        router = BEAMRouter()
        answers = await router.answer_questions(questions)
    """

    def __init__(self, fallback_retriever: Optional[type] = None):
        self._fallback_retriever = fallback_retriever or GraphCompletionRetriever
        self._retriever_cache: Dict[str, BaseRetriever] = {}

    def _get_retriever(self, question_type: str) -> BaseRetriever:
        """Get or create a retriever instance for the given question type."""
        if question_type not in self._retriever_cache:
            retriever_cls = _TYPE_RETRIEVERS.get(question_type, self._fallback_retriever)
            system_prompt = self.get_system_prompt(question_type)
            top_k = _TYPE_TOP_K.get(question_type, 10)
            kwargs: Dict[str, Any] = {"system_prompt": system_prompt, "top_k": top_k}

            # Graph-based retriever tuning
            if retriever_cls in (
                GraphCompletionRetriever,
                GraphCompletionContextExtensionRetriever,
            ):
                kwargs["wide_search_top_k"] = 200

            # More iterations for categories that need broader evidence
            if retriever_cls is GraphCompletionContextExtensionRetriever:
                kwargs["context_extension_rounds"] = _TYPE_EXTENSION_ROUNDS.get(
                    question_type, 4
                )

            self._retriever_cache[question_type] = retriever_cls(**kwargs)

        return self._retriever_cache[question_type]

    @staticmethod
    def get_system_prompt(question_type: str) -> str:
        """Get the system prompt for a question type."""
        return _TYPE_PROMPTS.get(question_type, _DEFAULT_PROMPT)

    def _make_retriever(self, question_type: str) -> BaseRetriever:
        """Create a fresh retriever instance (not cached) for parallel use."""
        retriever_cls = _TYPE_RETRIEVERS.get(question_type, self._fallback_retriever)
        system_prompt = self.get_system_prompt(question_type)
        top_k = _TYPE_TOP_K.get(question_type, 10)
        kwargs: Dict[str, Any] = {"system_prompt": system_prompt, "top_k": top_k}

        if retriever_cls in (
            GraphCompletionRetriever,
            GraphCompletionContextExtensionRetriever,
        ):
            kwargs["wide_search_top_k"] = 200
            kwargs["triplet_distance_penalty"] = 4.0

        if retriever_cls is GraphCompletionContextExtensionRetriever:
            kwargs["context_extension_rounds"] = _TYPE_EXTENSION_ROUNDS.get(
                question_type, 4
            )

        return retriever_cls(**kwargs)

    async def _answer_single(
        self, instance: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Answer a single question with concurrency control."""
        async with semaphore:
            query_text = instance["question"]
            question_type = instance.get("question_type", "information_extraction")
            golden_answer = instance["answer"]

            retriever = self._make_retriever(question_type)

            try:
                retrieved_objects = await retriever.get_retrieved_objects(query=query_text)
                retrieval_context = await retriever.get_context_from_objects(
                    query=query_text, retrieved_objects=retrieved_objects
                )
                search_results = await retriever.get_completion_from_context(
                    query=query_text,
                    retrieved_objects=retrieved_objects,
                    context=retrieval_context,
                )

                if isinstance(search_results, str):
                    search_results = [search_results]

                answer_text = search_results[0] if search_results else ""

            except Exception as e:
                logger.error(f"Failed to answer '{query_text[:80]}...': {e}")
                answer_text = f"ERROR: {e}"
                retrieval_context = ""

            answer = {
                "question": query_text,
                "answer": answer_text,
                "golden_answer": golden_answer,
                "retrieval_context": retrieval_context,
                "question_type": question_type,
            }

            if "rubric" in instance:
                answer["rubric"] = instance["rubric"]
            if "golden_context" in instance:
                answer["golden_context"] = instance["golden_context"]
            if "difficulty" in instance:
                answer["difficulty"] = instance["difficulty"]

            logger.info(
                f"[{question_type}] Answered: '{query_text[:60]}...' "
                f"(retriever: {type(retriever).__name__})"
            )

            return answer

    async def answer_questions(
        self, questions: List[Dict[str, Any]], max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Answer a list of BEAM probing questions using type-based routing.

        Args:
            questions: List of dicts with "question", "answer" (golden),
                "question_type", and optionally "rubric".
            max_concurrent: Max questions to answer in parallel.

        Returns:
            List of answer dicts compatible with the eval framework.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [self._answer_single(q, semaphore) for q in questions]
        return await asyncio.gather(*tasks)
