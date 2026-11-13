from typing import Any

from cognee.eval_framework.integrity.status import AnswerStatus
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


async def answer_question(
    retriever: Any, query: str, record_failures: bool = True, **completion_kwargs: Any
) -> dict[str, Any]:
    """Run one question through a retriever and classify the outcome.

    Returns ``answer``, ``retrieval_context`` and ``status`` (plus ``error`` on failure). A
    failure is recorded as a status, never as answer text, so it cannot be scored as an answer.
    Retrieval and completion are separate phases so the status says which one failed.
    With ``record_failures=False`` the retriever's exception propagates instead.
    ``completion_kwargs`` are forwarded to ``get_completion_from_context``.
    """
    try:
        retrieved_objects = await retriever.get_retrieved_objects(query=query)
        retrieval_context = await retriever.get_context_from_objects(
            query=query, retrieved_objects=retrieved_objects
        )
    except Exception as error:
        if not record_failures:
            raise
        logger.exception("Retrieval failed for question %r", query)
        return {
            "answer": "",
            "retrieval_context": "",
            "status": AnswerStatus.RETRIEVAL_ERROR.value,
            "error": _describe(error),
        }

    try:
        search_results = await retriever.get_completion_from_context(
            query=query,
            retrieved_objects=retrieved_objects,
            context=retrieval_context,
            **completion_kwargs,
        )
    except Exception as error:
        if not record_failures:
            raise
        logger.exception("LLM completion failed for question %r", query)
        return {
            "answer": "",
            "retrieval_context": retrieval_context,
            "status": AnswerStatus.LLM_ERROR.value,
            "error": _describe(error),
        }

    if not isinstance(search_results, list):
        search_results = [search_results]
    answer = search_results[0] if search_results else ""
    empty = answer is None or (isinstance(answer, str) and not answer.strip())
    return {
        "answer": "" if empty else answer,
        "retrieval_context": retrieval_context,
        "status": (AnswerStatus.EMPTY if empty else AnswerStatus.OK).value,
    }
