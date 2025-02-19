import cognee
from typing import List, Dict, Callable, Awaitable
from cognee.api.v1.search import SearchType

question_answering_engine_options: Dict[str, Callable[[str], Awaitable[List[str]]]] = {
    "cognee_graph_completion": lambda query: cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query
    ),
    "cognee_completion": lambda query: cognee.search(
        query_type=SearchType.COMPLETION, query_text=query
    ),
    "cognee_summaries": lambda query: cognee.search(
        query_type=SearchType.SUMMARIES, query_text=query
    ),
    "cognee_insights": lambda query: cognee.search(
        query_type=SearchType.INSIGHTS, query_text=query
    ),
    "cognee_chunks": lambda query: cognee.search(query_type=SearchType.CHUNKS, query_text=query),
    "cognee_code": lambda query: cognee.search(query_type=SearchType.CODE, query_text=query),
}


class AnswerGeneratorExecutor:
    async def question_answering_non_parallel(
        self,
        questions: List[Dict[str, str]],
        answer_resolver: Callable[[str], Awaitable[List[str]]],
    ) -> List[Dict[str, str]]:
        answers = []
        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]

            search_results = await answer_resolver(query_text)

            answers.append(
                {
                    "question": query_text,
                    "answer": search_results[0],
                    "golden_answer": correct_answer,
                }
            )

        return answers
