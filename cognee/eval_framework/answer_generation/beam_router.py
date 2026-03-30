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


# Map question types to retriever classes.
_TYPE_RETRIEVERS: Dict[str, type] = {
    # Direct factual lookup from graph triplets
    "information_extraction": GraphCompletionRetriever,
    "preference_following": GraphCompletionRetriever,
    "instruction_following": GraphCompletionRetriever,
    # Raw chunks — preserves [Session X, Turn Y] markers and dates for temporal reasoning
    "temporal_reasoning": CompletionRetriever,
    "event_ordering": CompletionRetriever,
    # Raw chunks — clearest signal for what was/wasn't discussed
    "abstention": CompletionRetriever,
    # Context extension — needs to connect info across different graph regions
    "knowledge_update": GraphCompletionContextExtensionRetriever,
    "contradiction_resolution": GraphCompletionContextExtensionRetriever,
    "multi_session_reasoning": GraphCompletionContextExtensionRetriever,
    # Raw chunks with high top_k — broad coverage for summarization
    "summarization": CompletionRetriever,
}

# Per-type top_k overrides (retriever defaults used otherwise)
_TYPE_TOP_K: Dict[str, int] = {
    "summarization": 20,
    "temporal_reasoning": 10,
    "event_ordering": 10,
    "abstention": 10,
    "information_extraction": 10,
    "contradiction_resolution": 10,
    "knowledge_update": 10,
}

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

    def _make_retriever(self, question_type: str) -> BaseRetriever:
        """Create a fresh retriever instance for parallel use."""
        retriever_cls = _TYPE_RETRIEVERS.get(question_type, self._fallback_retriever)
        kwargs: Dict[str, Any] = {"system_prompt": _DEFAULT_PROMPT}
        if question_type in _TYPE_TOP_K:
            kwargs["top_k"] = _TYPE_TOP_K[question_type]
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
