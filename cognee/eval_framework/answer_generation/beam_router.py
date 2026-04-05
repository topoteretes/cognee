"""BEAM question-type router — routes probing questions to appropriate retrievers.

Each BEAM question type maps to a retriever + system prompt strategy.
The router classifies the question (using pre-labeled types from the dataset)
and delegates to the matching retrieval strategy.
"""

from typing import Any, Dict, List, Optional

from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
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
        "Pay attention to dates, time references, and the order of events. "
        "Reference specific dates or time periods in your answer."
    ),
    "multi_session_reasoning": (
        "You are answering a question that requires combining information "
        "from multiple conversation sessions. Think step by step: first identify "
        "the relevant pieces of information from different sessions, then "
        "synthesize them into a coherent answer."
    ),
    "contradiction_resolution": (
        "You are answering a question about contradictory information in a conversation. "
        "Identify both the original statement and the contradicting one. "
        "Explain which is more recent or authoritative, and resolve the contradiction."
    ),
    "event_ordering": (
        "You are answering a question about the order of events in a conversation. "
        "List events in chronological order. Reference specific sessions or "
        "time anchors when available."
    ),
    "knowledge_update": (
        "You are answering a question about updated information. "
        "The conversation may contain both old and new versions of a fact. "
        "Always use the most recent information unless specifically asked about history."
    ),
    "summarization": (
        "You are summarizing part of a conversation. "
        "Cover all key points mentioned in the relevant sessions. "
        "Be comprehensive but concise. Use bullet points if appropriate."
    ),
    "abstention": (
        "You are answering a question where the correct response may be to abstain. "
        "If the conversation does not contain enough evidence to answer confidently, "
        "say that the information is not available rather than guessing."
    ),
    "preference_following": (
        "You are answering a question about user preferences expressed in the conversation. "
        "Pay attention to explicitly stated preferences, changes in preference over time, "
        "and implicit preferences inferred from the user's behavior."
    ),
    "instruction_following": (
        "You are answering a question about instructions given during the conversation. "
        "Check whether specific instructions were followed consistently. "
        "Reference the original instruction and evidence of compliance or non-compliance."
    ),
}

# Map question types to retriever classes
_TYPE_RETRIEVERS: Dict[str, type] = {
    "information_extraction": GraphCompletionRetriever,
    "temporal_reasoning": GraphCompletionRetriever,
    "event_ordering": GraphCompletionRetriever,
    "knowledge_update": GraphCompletionRetriever,
    "abstention": GraphCompletionRetriever,
    "preference_following": GraphCompletionRetriever,
    "instruction_following": GraphCompletionRetriever,
    # These require multi-step reasoning
    "multi_session_reasoning": GraphCompletionCotRetriever,
    "contradiction_resolution": GraphCompletionCotRetriever,
    # Summarization benefits from pre-computed summaries
    "summarization": GraphSummaryCompletionRetriever,
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
        self._retriever_cache: Dict[str, BaseRetriever] = {}

    def _get_retriever(self, question_type: str) -> BaseRetriever:
        """Get or create a retriever instance for the given question type."""
        retriever_cls = _TYPE_RETRIEVERS.get(question_type, self._fallback_retriever)
        cls_name = retriever_cls.__name__

        if cls_name not in self._retriever_cache:
            self._retriever_cache[cls_name] = retriever_cls()

        return self._retriever_cache[cls_name]

    @staticmethod
    def get_system_prompt(question_type: str) -> str:
        """Get the system prompt for a question type."""
        return _TYPE_PROMPTS.get(question_type, _DEFAULT_PROMPT)

    async def answer_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answer a list of BEAM probing questions using type-based routing.

        Args:
            questions: List of dicts with "question", "answer" (golden),
                "question_type", and optionally "rubric".

        Returns:
            List of answer dicts compatible with the eval framework.
        """
        answers = []

        for instance in questions:
            query_text = instance["question"]
            question_type = instance.get("question_type", "information_extraction")
            golden_answer = instance["answer"]

            retriever = self._get_retriever(question_type)
            system_prompt = self.get_system_prompt(question_type)

            try:
                retrieved_objects = await retriever.get_retrieved_objects(query=query_text)
                retrieval_context = await retriever.get_context_from_objects(
                    query=query_text, retrieved_objects=retrieved_objects
                )
                search_results = await retriever.get_completion_from_context(
                    query=query_text,
                    retrieved_objects=retrieved_objects,
                    context=retrieval_context,
                    system_prompt=system_prompt,
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

            answers.append(answer)
            logger.info(
                f"[{question_type}] Answered: '{query_text[:60]}...' "
                f"(retriever: {type(retriever).__name__})"
            )

        return answers
