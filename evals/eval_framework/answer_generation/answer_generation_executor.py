import cognee
from typing import Any
from cognee.api.v1.search import SearchType


class AnswerGeneratorExecutor:
    question_answering_engine_options = {
        "cognee_graph_completion": SearchType.GRAPH_COMPLETION,
        "cognee_completion": SearchType.COMPLETION,
    }

    search_type = None

    async def question_answering_non_parallel(
        self, questions: list[dict[str, Any]], qa_engine=None
    ):
        if not questions:
            raise ValueError("Questions list cannot be empty")
        if qa_engine not in self.question_answering_engine_options:
            raise ValueError(f"Unsupported QA engine: {qa_engine}")

        search_type = self.question_answering_engine_options[qa_engine]

        answers = []

        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]

            search_results = await cognee.search(query_type=search_type, query_text=query_text)

            answers.append(
                {
                    "question": query_text,
                    "answer": search_results[0],
                    "golden_answer": correct_answer,
                }
            )

        return answers
