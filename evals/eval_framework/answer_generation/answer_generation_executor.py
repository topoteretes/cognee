import cognee
from typing import List, Dict, Callable, Awaitable
from cognee.api.v1.search import SearchType


class AnswerGeneratorExecutor:
    # Each option is a function that takes a query (str) and returns an awaitable list of answers.
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
        "cognee_chunks": lambda query: cognee.search(
            query_type=SearchType.CHUNKS, query_text=query
        ),
        "cognee_code": lambda query: cognee.search(query_type=SearchType.CODE, query_text=query),
    }

    async def question_answering_non_parallel(
        self, questions: List[Dict[str, str]], qa_engine: str
    ) -> List[Dict[str, str]]:
        if not questions:
            raise ValueError("Questions list cannot be empty")
        if qa_engine not in self.question_answering_engine_options:
            raise ValueError(f"Unsupported QA engine: {qa_engine}")

        answer_resolver = self.question_answering_engine_options[qa_engine]

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
