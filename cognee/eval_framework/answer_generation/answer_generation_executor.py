from typing import List, Dict

from cognee.eval_framework.answer_generation.registry import get_fixed_retriever_options
from cognee.modules.retrieval.base_retriever import BaseRetriever

retriever_options = get_fixed_retriever_options()


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

            retrieved_objects = await retriever.get_retrieved_objects(query=query_text)
            retrieval_context = await retriever.get_context_from_objects(
                query=query_text, retrieved_objects=retrieved_objects
            )
            search_results = await retriever.get_completion_from_context(
                query=query_text, retrieved_objects=retrieved_objects, context=retrieval_context
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
