from collections.abc import Callable
from typing import Any

from cognee.eval_framework.integrity.answering import answer_question
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)

retriever_options: dict[str, Any] = {
    "cognee_graph_completion": GraphCompletionRetriever,
    "cognee_graph_completion_cot": GraphCompletionCotRetriever,
    "cognee_graph_completion_context_extension": GraphCompletionContextExtensionRetriever,
    "cognee_completion": CompletionRetriever,
    "graph_summary_completion": GraphSummaryCompletionRetriever,
}


class AnswerGeneratorExecutor:
    async def question_answering_non_parallel(
        self,
        questions: list[dict[str, str]],
        retriever: BaseRetriever,
        on_answer: Callable[[dict[str, Any]], None] | None = None,
        record_failures: bool = False,
    ) -> list[dict[str, Any]]:
        """Answer each question in turn.

        A retrieval or completion error propagates, as it always did. With
        ``record_failures`` the question instead gets a record with a failed ``status`` and the
        error text (never answer text) and the run continues. ``on_answer`` is called with every
        finished record as soon as it exists, which is how the run checkpoints its progress.
        """
        answers = []
        for instance in questions:
            query_text = instance["question"]
            outcome = await answer_question(retriever, query_text, record_failures)
            answer = {
                "question": query_text,
                "answer": outcome["answer"],
                "golden_answer": instance["answer"],
                "retrieval_context": outcome["retrieval_context"],
                "status": outcome["status"],
            }
            if "error" in outcome:
                answer["error"] = outcome["error"]

            if "golden_context" in instance:
                answer["golden_context"] = instance["golden_context"]

            if on_answer is not None:
                on_answer(answer)
            answers.append(answer)

        return answers
