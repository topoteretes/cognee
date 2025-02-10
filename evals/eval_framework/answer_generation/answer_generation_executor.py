import cognee
from typing import Any
from cognee.api.v1.search import SearchType


class AnswerGeneratorExecutor:
    # Note: not all search has generation at the end, only completions
    question_answering_engine_options = {
        "cognee_graph_completion": SearchType.GRAPH_COMPLETION,
        "cognee_completion": SearchType.COMPLETION,
        "cognee_summaries": SearchType.SUMMARIES,
        "cognee_insights": SearchType.INSIGHTS,
        "cognee_chunks": SearchType.CHUNKS,
        "cognee_code": SearchType.CODE,
    }

    search_type = None

    async def question_answering_non_parallel(self, questions: list[dict[str, str]], qa_engine):
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
