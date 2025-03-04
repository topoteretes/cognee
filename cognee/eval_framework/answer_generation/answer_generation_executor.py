import cognee
from typing import List, Dict, Callable, Awaitable
from cognee.api.v1.search import SearchType

question_answering_engine_options: Dict[str, Callable[[str, str], Awaitable[List[str]]]] = {
    "cognee_graph_completion": lambda query, system_prompt_path: cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=query,
        system_prompt_path=system_prompt_path,
    ),
    "cognee_completion": lambda query, system_prompt_path: cognee.search(
        query_type=SearchType.COMPLETION, query_text=query, system_prompt_path=system_prompt_path
    ),
    "graph_summary_completion": lambda query, system_prompt_path: cognee.search(
        query_type=SearchType.GRAPH_SUMMARY_COMPLETION,
        query_text=query,
        system_prompt_path=system_prompt_path,
    ),
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
