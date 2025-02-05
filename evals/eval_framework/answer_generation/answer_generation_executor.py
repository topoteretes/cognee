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
        search_type = self.question_answering_engine_options[qa_engine]

        answers = []

        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]
            # Perform async search for the question
            search_results = await cognee.search(search_type, query_text=query_text)

            # Store the results along with the question
            answers.append(
                {
                    "question": query_text,
                    "answer": search_results[0],
                    "golden_answer": correct_answer,
                }
            )

        return answers
