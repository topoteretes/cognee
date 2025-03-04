from typing import Any, Dict, List
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.eval_framework.eval_config import EvalConfig


class CorrectnessEvaluation(BaseModel):
    """Response model containing evaluation score and explanation."""

    score: float
    explanation: str


class DirectLLMEvalAdapter(BaseEvalAdapter):
    def __init__(self):
        """Initialize adapter with prompt paths from config."""
        config = EvalConfig()
        self.system_prompt_path = config.direct_llm_system_prompt
        self.eval_prompt_path = config.direct_llm_eval_prompt
        self.llm_client = get_llm_client()

    async def evaluate_correctness(
        self, question: str, answer: str, golden_answer: str
    ) -> Dict[str, Any]:
        args = {"question": question, "answer": answer, "golden_answer": golden_answer}

        user_prompt = render_prompt(self.eval_prompt_path, args)
        system_prompt = read_query_prompt(self.system_prompt_path)

        evaluation = await self.llm_client.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=CorrectnessEvaluation,
        )

        return {"score": evaluation.score, "reason": evaluation.explanation}

    async def evaluate_answers(
        self, answers: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluate a list of answers using specified metrics."""
        if not answers or not evaluator_metrics:
            return []

        if "correctness" not in evaluator_metrics:
            return [{"metrics": {}, **answer} for answer in answers]

        results = []
        for answer in answers:
            correctness = await self.evaluate_correctness(
                question=answer["question"],
                answer=answer["answer"],
                golden_answer=answer["golden_answer"],
            )
            results.append({**answer, "metrics": {"correctness": correctness}})

        return results
