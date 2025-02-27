import cognee
from typing import List, Dict, Callable, Awaitable
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.code_retriever import CodeRetriever
from cognee.modules.retrieval.base_retriever import BaseRetriever


retriever_options: Dict[str, BaseRetriever] = {
    "cognee_graph_completion": GraphCompletionRetriever,
    "cognee_completion": CompletionRetriever,
    "cognee_summaries": SummariesRetriever,
    "cognee_insights": InsightsRetriever,
    "cognee_chunks": ChunksRetriever,
    "cognee_code": CodeRetriever,
}


class AnswerGeneratorExecutor:
    async def question_answering_non_parallel(
        self,
        questions: List[Dict[str, str]],
        retriever_cls: BaseRetriever,
    ) -> List[Dict[str, str]]:
        retriever = retriever_cls()
        answers = []
        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]

            retrieval_context = await retriever.get_context(query_text)
            search_results = await retriever.get_completion(query_text, retrieval_context)

            answers.append(
                {
                    "question": query_text,
                    "answer": search_results[0],
                    "golden_answer": correct_answer,
                    "retrieval_context": retrieval_context,
                }
            )

        return answers
