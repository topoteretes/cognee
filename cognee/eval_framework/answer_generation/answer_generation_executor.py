from typing import List, Dict, Any
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)

from cognee.modules.retrieval.base_retriever import BaseRetriever


retriever_options: Dict[str, Any] = {
    "cognee_graph_completion": GraphCompletionRetriever,
    "cognee_graph_completion_cot": GraphCompletionCotRetriever,
    "cognee_graph_completion_context_extension": GraphCompletionContextExtensionRetriever,
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

            retrieved_objects = await retriever.get_retrieved_objects(query_text)
            retrieval_context = await retriever.get_context_from_objects(
                query_text, retrieved_objects
            )
            search_results = await retriever.get_completion_from_context(
                query_text, retrieved_objects, retrieval_context
            )

            ############
            if isinstance(search_results, str):
                search_results = [search_results]
            #############
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
