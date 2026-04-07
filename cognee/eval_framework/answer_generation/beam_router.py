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
    "summarization": 50,
    "temporal_reasoning": 20,
    "event_ordering": 20,
    "abstention": 20,
    "information_extraction": 20,
    "contradiction_resolution": 20,
    "knowledge_update": 20,
    "multi_session_reasoning": 20,
    "instruction_following": 20,
    "preference_following": 20,
}

_DEFAULT_PROMPT = (
    "You are answering a question about a conversation. "
    "Use the provided context to give an accurate answer."
)

# Per-type system prompt overrides for question types that benefit from targeted instructions.
_TYPE_PROMPTS: Dict[str, str] = {
    "summarization": (
        "You are summarizing a multi-session conversation. "
        "Provide a comprehensive, detailed summary that covers all key features, decisions, "
        "milestones, challenges, solutions, and outcomes discussed across every session. "
        "Be thorough and specific — include names, dates, technologies, and concrete details. "
        "Do not omit any significant topic or decision."
    ),
    "temporal_reasoning": (
        "You are answering a time-related question about a conversation. "
        "Pay close attention to all dates, deadlines, durations, and time intervals mentioned "
        "in the context. Calculate precisely and show your reasoning when computing differences "
        "between dates or deadlines."
    ),
    "knowledge_update": (
        "You are answering a question about a conversation where facts may have changed "
        "across sessions. Information given earlier may have been corrected, updated, or "
        "superseded in later sessions. Always identify the most recent version of any fact "
        "and use that as the current truth. If a value, status, or decision was revised, "
        "report the updated version, not the original."
    ),
    "contradiction_resolution": (
        "You are answering a question about a conversation where conflicting information "
        "may have been provided across different sessions. Carefully identify all instances "
        "where the same topic was discussed with different details. Determine which version "
        "is the most recent or authoritative, and explain the resolution. If the question "
        "asks about a specific fact, provide the latest correct value."
    ),
    "multi_session_reasoning": (
        "You are answering a question that requires combining information from multiple "
        "conversation sessions. Each session may contain different pieces of the answer. "
        "Carefully synthesize facts, decisions, and details from across all sessions to "
        "form a complete and accurate answer. Do not rely on a single session — look for "
        "connections and dependencies between sessions."
    ),
    "information_extraction": (
        "You are extracting specific factual information from a conversation. "
        "Answer with precise, concrete details — names, numbers, dates, technologies, "
        "or other specific facts exactly as stated in the conversation. Be direct and "
        "do not add qualifications or hedging. If the answer is a list, include all items."
    ),
    "instruction_following": (
        "You are answering a question where the user gave specific instructions or constraints "
        "during the conversation — for example, formatting requirements, preferences, or "
        "conditions. Pay close attention to any explicit instructions the user stated and "
        "ensure your answer reflects those constraints exactly. If the question asks what "
        "the user requested or specified, quote or paraphrase their instructions precisely."
    ),
    "abstention": (
        "You are answering a question about a conversation. If the topic was never discussed "
        "or the information was not mentioned in any session, you must clearly state that it "
        "was not discussed or that the information is not available. Do not guess or infer "
        "answers for topics that were not covered. Only answer based on what is explicitly "
        "present in the context."
    ),
    "event_ordering": (
        "You are answering a question about the chronological order of events in a conversation. "
        "List events in strict chronological order based on when they occurred or were discussed. "
        "Pay close attention to session numbers, turn numbers, dates, and temporal markers "
        "like 'before', 'after', 'first', 'then', 'finally'. Present the events as a clearly "
        "ordered sequence."
    ),
    "preference_following": (
        "You are answering a question about a user's stated preferences, choices, or "
        "likes/dislikes from a conversation. Focus on what the user explicitly expressed "
        "they prefer, want, or chose. Quote or paraphrase their exact preferences rather "
        "than inferring. If the user changed their preference across sessions, report the "
        "most recent one."
    ),
}


class BEAMRouter:
    """Routes BEAM probing questions to the appropriate retriever and prompt.

    Usage::

        router = BEAMRouter()
        answers = await router.answer_questions(questions)

        # For 10M-scale benchmarks with higher retrieval budgets:
        router = BEAMRouter(
            top_k_overrides={"summarization": 150, "DEFAULT": 50},
            context_extension_rounds=8,
        )
    """

    def __init__(
        self,
        fallback_retriever: Optional[type] = None,
        top_k_overrides: Optional[Dict[str, int]] = None,
        context_extension_rounds: Optional[int] = None,
        wide_search_top_k: Optional[int] = None,
        triplet_distance_penalty: Optional[float] = None,
    ):
        self._fallback_retriever = fallback_retriever or GraphCompletionRetriever
        self._top_k_overrides = top_k_overrides or {}
        self._context_extension_rounds = context_extension_rounds
        self._wide_search_top_k = wide_search_top_k
        self._triplet_distance_penalty = triplet_distance_penalty

    def _get_top_k(self, question_type: str) -> Optional[int]:
        """Resolve top_k: instance overrides > per-type defaults."""
        if question_type in self._top_k_overrides:
            return self._top_k_overrides[question_type]
        if "DEFAULT" in self._top_k_overrides:
            return self._top_k_overrides["DEFAULT"]
        return _TYPE_TOP_K.get(question_type)

    def _make_retriever(self, question_type: str) -> BaseRetriever:
        """Create a fresh retriever instance for parallel use."""
        retriever_cls = _TYPE_RETRIEVERS.get(question_type, self._fallback_retriever)
        prompt = _TYPE_PROMPTS.get(question_type, _DEFAULT_PROMPT)
        kwargs: Dict[str, Any] = {"system_prompt": prompt}
        top_k = self._get_top_k(question_type)
        if top_k is not None:
            kwargs["top_k"] = top_k
        # Graph retriever params — only passed when explicitly set (None = use retriever defaults)
        if self._wide_search_top_k is not None and retriever_cls in (
            GraphCompletionRetriever,
            GraphCompletionContextExtensionRetriever,
        ):
            kwargs["wide_search_top_k"] = self._wide_search_top_k
        if self._triplet_distance_penalty is not None and retriever_cls in (
            GraphCompletionRetriever,
            GraphCompletionContextExtensionRetriever,
        ):
            kwargs["triplet_distance_penalty"] = self._triplet_distance_penalty
        if (
            self._context_extension_rounds is not None
            and retriever_cls is GraphCompletionContextExtensionRetriever
        ):
            kwargs["context_extension_rounds"] = self._context_extension_rounds
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
        self, questions: List[Dict[str, Any]], max_concurrent: int = 10
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
