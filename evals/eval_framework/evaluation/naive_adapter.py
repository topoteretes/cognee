from typing import Any, Dict, List
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from evals.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter


class CorrectnessEvaluation(BaseModel):
    """Response model containing evaluation score and explanation."""

    score: float
    explanation: str


class NaiveAdapter(BaseEvalAdapter):
    def __init__(self):
        self.llm_client = get_llm_client()

    async def evaluate_correctness(
        self, question: str, answer: str, golden_answer: str
    ) -> Dict[str, Any]:
        system_prompt = """You are helping a reasonable person evaluate and score answers
    •	Compare the provided answer to the golden answer based on common-sense meaning and understanding.
	•	Focus on the meaning, not the exact wording or structure.
	•	If the answer is correct, don’t penalize it for being too short or too long.
	•	Extra details are fine as long as the correct answer is included.
	•	Score should be between 0 and 1.
	"""

        input_prompt = f"""Question: {question}
        Provided Answer: {answer}
        Golden Answer: {golden_answer}
        """

        evaluation = await self.llm_client.acreate_structured_output(
            text_input=input_prompt,
            system_prompt=system_prompt,
            response_model=CorrectnessEvaluation,
        )

        return {"score": evaluation.score, "reason": evaluation.explanation}

    async def evaluate_answers(
        self, answers: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        results = []
        for answer in answers:
            metric_results = {}

            for metric in evaluator_metrics:
                if metric == "correctness":
                    metric_results[metric] = await self.evaluate_correctness(
                        answer["question"], answer["answer"], answer["golden_answer"]
                    )
                else:
                    metric_results[metric] = self.constant_metrics[metric]

            results.append({**answer, "metrics": metric_results})

        return results
