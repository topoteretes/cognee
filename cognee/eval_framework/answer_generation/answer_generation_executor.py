from typing import List, Dict
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)

from cognee.modules.retrieval.base_retriever import BaseRetriever


retriever_options: Dict[str, BaseRetriever] = {
    "cognee_graph_completion": GraphCompletionRetriever,
    "cognee_completion": CompletionRetriever,
    "graph_summary_completion": GraphSummaryCompletionRetriever,
}


class AnswerGeneratorExecutor:
    async def question_answering_non_parallel(
        self,
        questions: List[Dict[str, str]],
        retriever: BaseRetriever,
    ) -> List[Dict[str, str]]:
        answers = []
        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]

            retrieval_context = await retriever.get_context(query_text)
            search_results = await retriever.get_completion(query_text, retrieval_context)

            answer = {
                "question": query_text,
                "answer": search_results[0],
                "golden_answer": correct_answer,
                "retrieval_context": retrieval_context,
            }

            if "golden_context" in instance:
                answer["golden_context"] = instance["golden_context"]

            answers.append(answer)

        return answers
